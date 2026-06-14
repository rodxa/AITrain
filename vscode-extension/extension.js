const http = require("http");
const https = require("https");
const vscode = require("vscode");

const output = vscode.window.createOutputChannel("Local Trained AI");
const chatViewType = "localTrainedAi.chatView";

function activate(context) {
  const chatProvider = new LocalAiChatViewProvider(context.extensionUri);
  context.subscriptions.push(output);
  context.subscriptions.push(vscode.window.registerWebviewViewProvider(chatViewType, chatProvider));
  context.subscriptions.push(vscode.commands.registerCommand("localTrainedAi.ask", askCommand));
  context.subscriptions.push(vscode.commands.registerCommand("localTrainedAi.explainSelection", explainSelectionCommand));
  context.subscriptions.push(vscode.commands.registerCommand("localTrainedAi.replaceSelection", replaceSelectionCommand));
  context.subscriptions.push(vscode.commands.registerCommand("localTrainedAi.insertReply", insertReplyCommand));
  context.subscriptions.push(vscode.commands.registerCommand("localTrainedAi.agentTask", agentTaskCommand));
  context.subscriptions.push(vscode.commands.registerCommand("localTrainedAi.openChat", openChatCommand));
}

function deactivate() {}

async function askCommand() {
  const editor = vscode.window.activeTextEditor;
  const prompt = await vscode.window.showInputBox({
    title: "Ask Local AI",
    prompt: "What should your local model do?",
    ignoreFocusOut: true
  });
  if (!prompt) return;

  const contextText = editor ? buildEditorContext(editor) : "";
  const reply = await askLocalModel(`${prompt}\n\n${contextText}`.trim());
  showReply(reply);
}

async function explainSelectionCommand() {
  const editor = requireEditor();
  if (!editor) return;

  const selected = getSelectedText(editor);
  if (!selected) {
    vscode.window.showInformationMessage("Select some code first.");
    return;
  }

  const reply = await askLocalModel([
    "Explain this code clearly and point out likely issues.",
    buildFileHeader(editor),
    selected
  ].join("\n\n"));
  showReply(reply);
}

async function replaceSelectionCommand() {
  const editor = requireEditor();
  if (!editor) return;

  const selected = getSelectedText(editor);
  if (!selected) {
    vscode.window.showInformationMessage("Select the text you want replaced first.");
    return;
  }

  const instruction = await vscode.window.showInputBox({
    title: "Replace Selection With Local AI",
    prompt: "Describe the change to make to the selected text.",
    ignoreFocusOut: true
  });
  if (!instruction) return;

  const reply = await askLocalModel([
    "Return only the replacement text. Do not wrap it in Markdown fences.",
    `Instruction: ${instruction}`,
    buildFileHeader(editor),
    "Selected text:",
    selected
  ].join("\n\n"));

  await editor.edit(editBuilder => {
    editBuilder.replace(editor.selection, cleanCodeFence(reply));
  });
}

async function insertReplyCommand() {
  const editor = requireEditor();
  if (!editor) return;

  const prompt = await vscode.window.showInputBox({
    title: "Insert Local AI Reply",
    prompt: "What should your local model generate here?",
    ignoreFocusOut: true
  });
  if (!prompt) return;

  const reply = await askLocalModel([
    "Return only the text to insert. Do not wrap it in Markdown fences.",
    `Instruction: ${prompt}`,
    buildEditorContext(editor)
  ].join("\n\n"));

  await editor.edit(editBuilder => {
    editBuilder.insert(editor.selection.active, cleanCodeFence(reply));
  });
}

async function agentTaskCommand() {
  const editor = vscode.window.activeTextEditor;
  const instruction = await vscode.window.showInputBox({
    title: "Local AI Agent Task",
    prompt: "Describe the task. The model may propose file edits or a terminal command.",
    ignoreFocusOut: true
  });
  if (!instruction) return;

  const reply = await askLocalModel([
    "You are controlling VS Code through a safe action API.",
    "Return only JSON. No Markdown. No explanation outside JSON.",
    "Schema:",
    "{\"summary\":\"short summary\",\"actions\":[{\"type\":\"show_message\",\"text\":\"...\"},{\"type\":\"replace_selection\",\"text\":\"...\"},{\"type\":\"insert_at_cursor\",\"text\":\"...\"},{\"type\":\"create_or_update_file\",\"path\":\"relative/path.ext\",\"content\":\"...\"},{\"type\":\"run_terminal_command\",\"command\":\"...\"}]}",
    "Use create_or_update_file only for workspace-relative paths.",
    "Use run_terminal_command only when necessary.",
    `Task: ${instruction}`,
    editor ? buildEditorContext(editor) : "No active editor."
  ].join("\n\n"));

  let plan;
  try {
    plan = extractJsonObject(reply);
  } catch (error) {
    showReply(reply);
    vscode.window.showWarningMessage("The model did not return valid agent JSON. I put the raw reply in the Local Trained AI output panel.");
    return;
  }

  await applyAgentPlan(plan);
}

function openChatCommand() {
  vscode.commands.executeCommand(`${chatViewType}.focus`);
}

class LocalAiChatViewProvider {
  constructor(extensionUri) {
    this.extensionUri = extensionUri;
  }

  resolveWebviewView(webviewView) {
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri]
    };
    webviewView.webview.html = getChatHtml();
    webviewView.webview.onDidReceiveMessage(async message => {
      if (message.type !== "ask") return;
      try {
        const editor = vscode.window.activeTextEditor;
        const contextText = message.includeContext !== false && editor ? buildEditorContext(editor) : "";
        if (message.agentMode) {
          const reply = await askLocalModel(buildAgentPrompt(message.text, contextText), {
            maxNewTokens: message.maxNewTokens,
            temperature: message.temperature
          });
          let plan;
          try {
            plan = extractJsonObject(reply);
          } catch (error) {
            webviewView.webview.postMessage({ type: "reply", text: `I could not read this as agent JSON:\n\n${reply}` });
            return;
          }
          await applyAgentPlan(plan);
          webviewView.webview.postMessage({ type: "reply", text: plan.summary || "Agent actions applied." });
          return;
        }

        const reply = await askLocalModel(`${message.text}\n\n${contextText}`.trim(), {
          maxNewTokens: message.maxNewTokens,
          temperature: message.temperature
        });
        webviewView.webview.postMessage({ type: "reply", text: reply });
      } catch (error) {
        webviewView.webview.postMessage({ type: "reply", text: `Error: ${error.message}` });
      }
    });
  }
}

async function applyAgentPlan(plan) {
  const actions = Array.isArray(plan.actions) ? plan.actions : [];
  if (!actions.length) {
    showReply(plan.summary || "No actions returned.");
    return;
  }

  const summary = plan.summary || `${actions.length} action(s) proposed.`;
  const choice = await vscode.window.showWarningMessage(
    `Local AI wants to apply: ${summary}`,
    { modal: true },
    "Apply",
    "Show JSON"
  );

  if (choice === "Show JSON") {
    showReply(JSON.stringify(plan, null, 2));
    return;
  }
  if (choice !== "Apply") return;

  for (const action of actions) {
    await applyAgentAction(action);
  }
}

async function applyAgentAction(action) {
  if (!action || typeof action.type !== "string") return;

  if (action.type === "show_message") {
    showReply(String(action.text || ""));
    return;
  }

  if (action.type === "replace_selection") {
    const editor = requireEditor();
    if (!editor) return;
    await editor.edit(editBuilder => editBuilder.replace(editor.selection, String(action.text || "")));
    return;
  }

  if (action.type === "insert_at_cursor") {
    const editor = requireEditor();
    if (!editor) return;
    await editor.edit(editBuilder => editBuilder.insert(editor.selection.active, String(action.text || "")));
    return;
  }

  if (action.type === "create_or_update_file") {
    const uri = workspaceRelativeUri(action.path);
    if (!uri) return;
    const content = Buffer.from(String(action.content || ""), "utf8");
    await vscode.workspace.fs.writeFile(uri, content);
    const document = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(document);
    return;
  }

  if (action.type === "run_terminal_command") {
    const command = String(action.command || "").trim();
    if (!command) return;
    const choice = await vscode.window.showWarningMessage(
      `Run terminal command?\n\n${command}`,
      { modal: true },
      "Run"
    );
    if (choice !== "Run") return;
    const terminal = vscode.window.createTerminal("Local AI Agent");
    terminal.show();
    terminal.sendText(command);
  }
}

async function askLocalModel(message, options = {}) {
  const config = vscode.workspace.getConfiguration("localTrainedAi");
  const apiBase = config.get("apiBase", "http://127.0.0.1:5000").replace(/\/$/, "");
  const body = {
    message,
    max_new_tokens: Number(options.maxNewTokens || config.get("maxNewTokens", 2048)),
    temperature: Number(options.temperature ?? config.get("temperature", 0.35)),
    context_limit: config.get("contextLimit", 6),
    context_chars: config.get("contextChars", 1200)
  };

  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Local AI is thinking..." },
    async () => {
      const result = await postJson(`${apiBase}/chat`, body);
      if (result.error) {
        throw new Error(result.error);
      }
      return result.reply || "";
    }
  );
}

function postJson(url, body) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const payload = Buffer.from(JSON.stringify(body), "utf8");
    const transport = parsed.protocol === "https:" ? https : http;

    const request = transport.request({
      method: "POST",
      hostname: parsed.hostname,
      port: parsed.port,
      path: `${parsed.pathname}${parsed.search}`,
      headers: {
        "Content-Type": "application/json",
        "Content-Length": payload.length
      }
    }, response => {
      let data = "";
      response.setEncoding("utf8");
      response.on("data", chunk => {
        data += chunk;
      });
      response.on("end", () => {
        try {
          const parsedBody = data ? JSON.parse(data) : {};
          if (response.statusCode < 200 || response.statusCode >= 300) {
            reject(new Error(parsedBody.error || `Server returned ${response.statusCode}`));
            return;
          }
          resolve(parsedBody);
        } catch (error) {
          reject(new Error(`Could not parse server response: ${data.slice(0, 200)}`));
        }
      });
    });

    request.on("error", error => {
      reject(new Error(`${error.message}. Start the server with "python serve.py" first.`));
    });
    request.write(payload);
    request.end();
  });
}

function buildAgentPrompt(task, contextText) {
  return [
    "You are controlling VS Code through a safe action API.",
    "Return only JSON. No Markdown. No explanation outside JSON.",
    "Schema:",
    "{\"summary\":\"short summary\",\"actions\":[{\"type\":\"show_message\",\"text\":\"...\"},{\"type\":\"replace_selection\",\"text\":\"...\"},{\"type\":\"insert_at_cursor\",\"text\":\"...\"},{\"type\":\"create_or_update_file\",\"path\":\"relative/path.ext\",\"content\":\"...\"},{\"type\":\"run_terminal_command\",\"command\":\"...\"}]}",
    "Rules:",
    "- Use create_or_update_file for creating or replacing complete files.",
    "- Use replace_selection only when the user selected text.",
    "- Use insert_at_cursor only for small insertions.",
    "- Use workspace-relative paths only, such as src/app.js or index.html.",
    "- Do not use absolute paths.",
    "- Use run_terminal_command only when necessary.",
    `Task: ${task}`,
    contextText || "No active editor context."
  ].join("\n\n");
}

function requireEditor() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showInformationMessage("Open a file first.");
    return null;
  }
  return editor;
}

function getSelectedText(editor) {
  return editor.document.getText(editor.selection).trim();
}

function buildEditorContext(editor) {
  const selected = getSelectedText(editor);
  const documentText = editor.document.getText();
  const maxChars = 12000;
  const content = selected || documentText.slice(0, maxChars);
  const label = selected ? "Selected text" : `File content${documentText.length > maxChars ? " (truncated)" : ""}`;
  return [buildFileHeader(editor), `${label}:`, content].join("\n\n");
}

function buildFileHeader(editor) {
  const document = editor.document;
  const language = document.languageId || "unknown";
  return `File: ${document.fileName}\nLanguage: ${language}`;
}

function cleanCodeFence(text) {
  const trimmed = text.trim();
  const fenceMatch = trimmed.match(/^```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```$/);
  return fenceMatch ? fenceMatch[1] : text;
}

function extractJsonObject(text) {
  const cleaned = cleanCodeFence(text).trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) {
      throw new Error("No JSON object found.");
    }
    return JSON.parse(cleaned.slice(start, end + 1));
  }
}

function workspaceRelativeUri(relativePath) {
  const workspaceFolder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  if (!workspaceFolder) {
    vscode.window.showWarningMessage("Open a workspace folder before creating files.");
    return null;
  }

  const normalized = String(relativePath || "").replace(/\\/g, "/").replace(/^\/+/, "");
  if (!normalized || normalized.includes("../") || normalized === ".." || normalized.startsWith("..")) {
    vscode.window.showWarningMessage(`Unsafe workspace path rejected: ${relativePath}`);
    return null;
  }

  return vscode.Uri.joinPath(workspaceFolder.uri, ...normalized.split("/").filter(Boolean));
}

function showReply(reply) {
  output.clear();
  output.appendLine(reply);
  output.show(true);
}

function getChatHtml() {
  const nonce = Date.now().toString(36);
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Co-mate AI Chat</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--vscode-foreground);
      background: var(--vscode-sideBar-background);
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
    }
    .shell {
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100vh;
      min-height: 0;
    }
    .top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px;
      border-bottom: 1px solid var(--vscode-sideBar-border, var(--vscode-panel-border));
    }
    .title { font-weight: 700; }
    .status {
      overflow: hidden;
      color: var(--vscode-descriptionForeground);
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .log {
      display: grid;
      align-content: start;
      gap: 10px;
      min-height: 0;
      overflow: auto;
      padding: 10px;
    }
    .msg {
      max-width: 94%;
      padding: 9px 10px;
      border: 1px solid var(--vscode-panel-border);
      border-radius: 6px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .user {
      justify-self: end;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .assistant {
      justify-self: start;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
    }
    .composer {
      display: grid;
      gap: 8px;
      padding: 10px;
      border-top: 1px solid var(--vscode-sideBar-border, var(--vscode-panel-border));
      background: var(--vscode-sideBar-background);
    }
    textarea {
      width: 100%;
      min-height: 78px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--vscode-input-border, transparent);
      border-radius: 4px;
      padding: 8px;
      color: var(--vscode-input-foreground);
      background: var(--vscode-input-background);
      font-family: inherit;
    }
    textarea:focus {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: -1px;
    }
    .row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    label {
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      color: var(--vscode-descriptionForeground);
      font-size: 12px;
    }
    input { margin: 0; }
    .number {
      width: 68px;
      border: 1px solid var(--vscode-input-border, transparent);
      border-radius: 4px;
      padding: 3px 5px;
      color: var(--vscode-input-foreground);
      background: var(--vscode-input-background);
    }
    button {
      min-height: 30px;
      border: 0;
      border-radius: 4px;
      padding: 6px 10px;
      color: var(--vscode-button-foreground);
      background: var(--vscode-button-background);
      cursor: pointer;
      font-family: inherit;
      font-weight: 700;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button.secondary {
      color: var(--vscode-button-secondaryForeground);
      background: var(--vscode-button-secondaryBackground);
    }
    button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div class="title">Co-mate</div>
      <div id="status" class="status">Ready</div>
    </div>
    <div id="log" class="log"></div>
    <div class="composer">
      <textarea id="input" placeholder="Ask your local model..."></textarea>
      <div class="row">
        <label title="Attach the active file or selected text to each message.">
          <input id="includeContext" type="checkbox" checked>
          file context
        </label>
        <label title="Maximum tokens to generate. Higher values are slower and need model context support.">
          tokens
          <input id="maxTokens" class="number" type="number" min="16" max="8192" step="16" value="2048">
        </label>
      </div>
      <div class="row">
        <label title="Let the model propose file edits, file creation, or terminal commands. VS Code asks before applying.">
          <input id="agentMode" type="checkbox">
          agent mode
        </label>
      </div>
      <div class="row">
        <label title="Lower values are more deterministic.">
          temp
          <input id="temperature" class="number" type="number" min="0" max="2" step="0.05" value="0.35">
        </label>
        <div>
          <button id="continue" class="secondary">Continue</button>
          <button id="clear" class="secondary">Clear</button>
          <button id="send">Send</button>
        </div>
      </div>
    </div>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const log = document.getElementById("log");
    const input = document.getElementById("input");
    const status = document.getElementById("status");
    const sendButton = document.getElementById("send");
    const includeContext = document.getElementById("includeContext");
    const agentMode = document.getElementById("agentMode");
    const maxTokens = document.getElementById("maxTokens");
    const temperature = document.getElementById("temperature");
    let lastAssistantReply = "";
    sendButton.addEventListener("click", send);
    document.getElementById("continue").addEventListener("click", () => {
      if (!lastAssistantReply) return;
      send("Continue from exactly where your last answer stopped. Do not restart. Continue the same code or text.");
    });
    document.getElementById("clear").addEventListener("click", () => {
      log.textContent = "";
      lastAssistantReply = "";
      status.textContent = "Ready";
      input.focus();
    });
    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        send();
      }
    });
    window.addEventListener("message", event => {
      removeThinking();
      lastAssistantReply = event.data.text || "";
      append("assistant", lastAssistantReply);
      status.textContent = "Ready";
      sendButton.disabled = false;
      input.focus();
    });
    function send(overrideText) {
      const text = (overrideText || input.value).trim();
      if (!text) return;
      append("user", text);
      if (!overrideText) input.value = "";
      append("assistant thinking", "Thinking...");
      status.textContent = "Thinking...";
      sendButton.disabled = true;
      vscode.postMessage({
        type: "ask",
        text,
        includeContext: includeContext.checked,
        agentMode: agentMode.checked,
        maxNewTokens: Number(maxTokens.value || 2048),
        temperature: Number(temperature.value || 0.35)
      });
    }
    function append(role, text) {
      const item = document.createElement("div");
      item.className = "msg " + role;
      item.textContent = text;
      log.appendChild(item);
      item.scrollIntoView();
    }
    function removeThinking() {
      const item = log.querySelector(".thinking");
      if (item) item.remove();
    }
  </script>
</body>
</html>`;
}

module.exports = {
  activate,
  deactivate
};
