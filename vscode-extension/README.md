# Local Trained AI Agent

This VS Code extension connects to the Flask app in the parent project and uses the trained adapter through `POST /chat`.

## Run

1. Start the model server from the project root:

   ```powershell
   python serve.py
   ```

2. Open `vscode-extension/` in VS Code.
3. Press `F5` to launch an Extension Development Host.
4. In the new VS Code window, open the project you want to work on.
5. Run commands from the Command Palette:

   - `Local AI: Open Chat`
   - `Local AI: Ask`
   - `Local AI: Explain Selection`
   - `Local AI: Replace Selection`
   - `Local AI: Insert Reply`
   - `Local AI: Agent Task`

The extension also adds a **Local AI** icon to the left Activity Bar. Click it to use the side-window chat. The chat sends your current selected text, or the active file if nothing is selected, when **file context** is checked.

The token box in the sidebar controls `max_new_tokens` for VS Code chat. The website's chat token slider does not affect the extension. If a long answer stops early, click **Continue**.

Turn on **agent mode** in the sidebar when you want the model to create files, replace selected text, insert at the cursor, update workspace files, or suggest terminal commands. VS Code asks before applying the proposed actions.

## Settings

The extension defaults to:

```text
http://127.0.0.1:5000
```

Change `localTrainedAi.apiBase` in VS Code settings if the Flask app is hosted elsewhere.

For longer code output, raise `localTrainedAi.maxNewTokens` or use the token box in the sidebar. The extension allows up to `8192`, but the actual maximum still depends on the base model, GPU memory, and context length.

## Notes

The model does not directly edit files. VS Code performs edits after the extension receives a reply from the local model. This keeps file access inside the editor instead of exposing the filesystem to the model server.

`Local AI: Agent Task` asks the model for JSON actions, then asks before applying them. Supported actions are:

- `show_message`
- `replace_selection`
- `insert_at_cursor`
- `create_or_update_file`
- `run_terminal_command`
