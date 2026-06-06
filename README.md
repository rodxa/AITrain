# Local Qwen Coder Agent Trainer

Small Flask app for uploading conversations, notes, transcripts, and code/text files, then fine-tuning a local Qwen Coder model with a QLoRA/LoRA adapter.

Recommended setup for this machine:

- Train locally right now with cached `Qwen/Qwen2.5-3B-Instruct` using 4-bit QLoRA.
- Use `Qwen/Qwen2.5-Coder-7B-Instruct` as the better coding target once Hugging Face downloads are stable.
- Try `Qwen/Qwen2.5-Coder-14B-Instruct` later if the large Hugging Face shard download is stable on your connection.
- Run the strongest local VS Code agent with a quantized `Qwen/Qwen3-Coder-30B-A3B-Instruct` through Ollama, LM Studio, llama.cpp, or another OpenAI-compatible local server.
- Keep personality data light: default mix is coding-first, with WhatsApp/personality examples capped to about 10% of the generated dataset.

Training dropdown options:

- `Qwen/Qwen2.5-Coder-7B-Instruct`: recommended
- `Qwen/Qwen2.5-Coder-3B-Instruct`: fast test
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`: very fast test
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`: smoke test
- `Qwen/Qwen2.5-Coder-14B-Instruct`: ambitious on this connection
- `Qwen/Qwen2.5-Coder-32B-Instruct` and `Qwen/Qwen3-Coder-30B-A3B-Instruct`: inference-only here

## Start

```powershell
python -m pip install -r requirements.txt
$env:TRAIN_STORAGE_DIR="C:\TrainData"
python serve.py
```

Open `http://127.0.0.1:5000`.

Fastest start: choose `Qwen2.5 3B Instruct - cached, train now`. It should use the local Hugging Face cache, then train a coding/personality adapter from your uploaded files.

## Files

The app accepts readable text-like files. Conversation formats such as `.txt`, `.log`, `.chat`, `.md`, and `.jsonl` can be up to 5 MB each.

Training writes:

- `uploads/`: copied upload files
- `training_data.jsonl`: generated text dataset
- `training.log`: training progress
- `conversation-ai-lora/`: trained LoRA adapter

By default these files are written in the project folder. Set `TRAIN_STORAGE_DIR` to save them somewhere else, such as `C:\TrainData` on Windows or `/data` in a container.

See `DEPLOY.md` for Docker and online deployment notes.

Do not use an Ollama manifest path like `C:\Users\...\.ollama\models\manifests\...` as the model path. This trainer needs a Hugging Face/Transformers model folder or the downloadable default model.

## Useful Environment Knobs

```powershell
$env:QWEN_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
$env:ALLOW_MODEL_DOWNLOAD="1"
$env:HF_TOKEN="hf_your_token_here"
$env:USE_QLORA="1"
$env:MAX_LENGTH="1536"
$env:GRAD_ACCUM_STEPS="16"
$env:LORA_R="16"
$env:LORA_TARGET_MODE="all"
python app.py
```
