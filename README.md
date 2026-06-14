# AI Training Control

Small Flask app for training a local Qwen-style LoRA/QLoRA adapter from files you choose in the browser.

The page is the control surface. Uploaded files plus the visible form values are what a training run uses: base model, system prompt, parsing mode, filters, chunking, eval split, LoRA settings, optimizer settings, and chat-test generation settings.

## Start

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Use The Trained AI In VS Code

This repo includes a local VS Code extension in `vscode-extension/`. It connects to the same Flask server and uses the trained adapter through `/chat`.

```powershell
python serve.py
```

Then open `vscode-extension/` in VS Code and press `F5`. In the Extension Development Host window, use the Command Palette:

- Click the `Local AI` icon in the left Activity Bar to open the side-window chat
- `Local AI: Open Chat`
- `Local AI: Ask`
- `Local AI: Explain Selection`
- `Local AI: Replace Selection`
- `Local AI: Insert Reply`
- `Local AI: Agent Task`

By default the extension calls `http://127.0.0.1:5000`. Change `localTrainedAi.apiBase` in VS Code settings if your server is somewhere else.

The side-window chat has its own token box. The website's chat token slider does not control VS Code. If long code stops early, raise the sidebar token value and click `Continue`.

Turn on `agent mode` in the side-window chat when you want it to create or edit files. VS Code will ask before applying the actions.

Optional:

```powershell
$env:TRAIN_STORAGE_DIR="C:\TrainData"
$env:HF_TOKEN="hf_your_token_here"
python app.py
```

## What Gets Written

- `uploads/`: files copied from the page for the current run
- `training_data.jsonl`: generated train examples
- `eval_data.jsonl`: generated eval examples
- `training.log`: latest training log
- `conversation-ai-lora/`: trained adapter
- `agent_instructions.txt`: latest system prompt submitted from the page

`Clear Data` removes uploaded files and generated datasets. It does not delete the trained adapter.

## Data Modes

- `Auto`: conversation files can produce chat examples, and all readable files can produce context chunks.
- `Chat replies only`: only parsed conversation-style assistant replies become examples.
- `Raw context chunks only`: files become chunked text examples.

Supported conversation-ish files include `.txt`, `.log`, `.chat`, `.md`, `.json`, and `.jsonl`. Binary files are skipped by the trainer.

## Useful Controls

- `System prompt / behavior instructions`: controls the system message inserted into generated examples and chat tests.
- `Assistant speaker name`: tells the parser which speaker in chat logs should become the assistant.
- `Filter low-value replies`: removes empty, deleted, media-placeholder, and link-only replies.
- `Chunk chars` and `Chunk overlap`: control raw text chunking.
- `Eval ratio` and `Dataset seed`: control train/eval splitting and shuffle repeatability.
- `LoRA target modules`: overrides the target preset when filled.
