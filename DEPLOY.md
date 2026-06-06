# Deployment

This app can run locally on Windows or in a container. Training large models online requires a GPU host; CPU-only web hosts are only useful for testing the UI and upload flow.

## Local Windows

Save all app data outside the project folder:

```powershell
python -m pip install -U -r requirements.txt
$env:TRAIN_STORAGE_DIR="C:\TrainData"
python serve.py
```

Open:

```text
http://127.0.0.1:5000
```

The app will save uploads, generated datasets, logs, instructions, and LoRA adapters inside `C:\TrainData`.

## Docker

Build:

```powershell
docker build -t local-qwen-trainer .
```

Run with a persistent folder:

```powershell
docker run --rm -p 5000:5000 -v C:\TrainData:/data local-qwen-trainer
```

On Linux:

```bash
docker run --rm -p 5000:5000 -v /srv/train-data:/data local-qwen-trainer
```

## Online Hosting

Set:

```text
TRAIN_STORAGE_DIR=/data
PORT=5000
```

Mount a persistent disk/volume at `/data`. Without persistent storage, uploaded files, logs, datasets, and adapters can disappear when the host restarts.

For public internet testing, protect the app behind authentication or a private tunnel. Uploaded chat exports and training data are sensitive.
