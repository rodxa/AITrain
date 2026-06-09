from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path, *, override: bool, protected_keys: set[str]) -> None:
    if not path.exists():
        return

    current_key = None
    current_value = None
    parsed: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
            if current_key is not None:
                parsed.append((current_key, current_value or ""))
            key, value = stripped.split("=", 1)
            current_key = key.strip()
            current_value = value.strip().strip('"').strip("'")
            continue
        if current_key is not None:
            current_value = f"{current_value}\n{line.rstrip()}"

    if current_key is not None:
        parsed.append((current_key, current_value or ""))

    for key, value in parsed:
        if override and (key not in protected_keys or not os.environ.get(key)):
            os.environ[key] = value
        elif not os.environ.get(key):
            os.environ[key] = value


PROTECTED_ENV_KEYS = set(os.environ)
load_env_file(BASE_DIR / ".env.example", override=False, protected_keys=PROTECTED_ENV_KEYS)
load_env_file(BASE_DIR / ".env", override=True, protected_keys=PROTECTED_ENV_KEYS)

STORAGE_DIR = Path(os.environ.get("TRAIN_STORAGE_DIR", BASE_DIR)).resolve()
UPLOAD_DIR = STORAGE_DIR / "uploads"
LOG_FILE = STORAGE_DIR / "training.log"
TRAIN_SCRIPT = BASE_DIR / "train_model.py"
ADAPTER_DIR = STORAGE_DIR / "conversation-ai-lora"
CHAT_CONTEXT_FILE = STORAGE_DIR / "training_data.jsonl"
EVAL_CONTEXT_FILE = STORAGE_DIR / "eval_data.jsonl"
AGENT_INSTRUCTIONS_FILE = STORAGE_DIR / "agent_instructions.txt"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
training_process: subprocess.Popen | None = None
chat_lock = threading.Lock()
chat_state = {"model": None, "tokenizer": None, "torch": None, "adapter_dir": None}
context_cache = {"mtime": None, "rows": []}
adapter_scale_state = {"value": None}


def env_hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


def env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "on", "yes"}


def current_agent_instructions() -> str:
    env_instructions = os.environ.get("AGENT_INSTRUCTIONS", "").strip()
    if env_instructions:
        return env_instructions
    if AGENT_INSTRUCTIONS_FILE.exists():
        return AGENT_INSTRUCTIONS_FILE.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def form_defaults() -> dict:
    return {
        "model_path": env_value("QWEN_MODEL_PATH", "MODEL_PATH", default="Qwen/Qwen2.5-3B-Instruct"),
        "assistant_speaker_name": env_value("ASSISTANT_SPEAKER_NAME"),
        "agent_instructions": current_agent_instructions(),
        "data_mode": env_value("DATA_MODE", default="auto"),
        "dataset_seed": env_value("DATASET_SEED", default="42"),
        "eval_ratio": env_value("EVAL_RATIO", default="0.08"),
        "max_file_bytes": env_value("MAX_FILE_BYTES", default="200000"),
        "max_conversation_bytes": env_value("MAX_CONVERSATION_BYTES", default=str(5 * 1024 * 1024)),
        "chunk_chars": env_value("CHUNK_CHARS", default="3500"),
        "chunk_overlap": env_value("CHUNK_OVERLAP", default="500"),
        "max_length": env_value("MAX_LENGTH", default="1024"),
        "chat_context_min": env_value("CHAT_CONTEXT_MIN", "WHATSAPP_CONTEXT_MIN", default="1"),
        "chat_context_max": env_value("CHAT_CONTEXT_MAX", "WHATSAPP_CONTEXT_MAX", default="15"),
        "batch_size": env_value("BATCH_SIZE", default="1"),
        "grad_accum_steps": env_value("GRAD_ACCUM_STEPS", default="16"),
        "epochs": env_value("EPOCHS", default="1"),
        "max_steps": env_value("MAX_STEPS"),
        "learning_rate": env_value("LEARNING_RATE", default="5e-5"),
        "optim": env_value("OPTIM"),
        "lora_rank": env_value("LORA_R", default="16"),
        "lora_alpha": env_value("LORA_ALPHA", default="16"),
        "lora_dropout": env_value("LORA_DROPOUT", default="0.05"),
        "lora_target_mode": env_value("LORA_TARGET_MODE", default="all"),
        "lora_target_modules": env_value("LORA_TARGET_MODULES"),
        "hf_token": env_hf_token(),
        "allow_model_download": env_bool("ALLOW_MODEL_DOWNLOAD", True),
        "use_qlora": env_bool("USE_QLORA", True),
        "gpu_train": env_bool("GPU_TRAIN", True),
        "filter_low_value_replies": env_bool("FILTER_LOW_VALUE_REPLIES", True),
        "keep_short_replies": env_bool("KEEP_SHORT_REPLIES", True),
    }


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


def safe_upload_path(filename: str) -> Path:
    parts = [secure_filename(part) for part in Path(filename).parts]
    parts = [part for part in parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("Invalid filename")
    return UPLOAD_DIR.joinpath(*parts)


def clear_training_inputs() -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    for path in (CHAT_CONTEXT_FILE, EVAL_CONTEXT_FILE):
        if path.exists():
            path.unlink()
    context_cache.update({"mtime": None, "rows": []})


def unload_chat_model() -> None:
    with chat_lock:
        torch = chat_state.get("torch")
        chat_state.update({"model": None, "tokenizer": None, "torch": None, "adapter_dir": None})
        adapter_scale_state["value"] = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def bool_form(name: str, default: bool = False) -> bool:
    value = request.form.get(name)
    if value is None:
        return default
    return value in {"1", "true", "on", "yes"}


def set_child_env(child_env: dict[str, str], form_name: str, env_name: str, default: str = "") -> None:
    value = request.form.get(form_name, "").strip()
    if value or default:
        child_env[env_name] = value or default


def load_adapter_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No trained adapter found at {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"No base model found in {config_path}")
    return base_model


def get_chat_model(adapter_dir: Path):
    adapter_dir = adapter_dir.resolve()
    if chat_state["model"] is not None and chat_state.get("adapter_dir") == str(adapter_dir):
        return chat_state

    with chat_lock:
        if chat_state["model"] is not None and chat_state.get("adapter_dir") == str(adapter_dir):
            return chat_state
        torch = chat_state.get("torch")
        chat_state.update({"model": None, "tokenizer": None, "torch": None, "adapter_dir": None})
        adapter_scale_state["value"] = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_model = load_adapter_base_model(adapter_dir)
        cuda_available = torch.cuda.is_available()
        bf16_available = cuda_available and torch.cuda.is_bf16_supported()
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if cuda_available and os.environ.get("CHAT_USE_4BIT", "1") == "1":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=os.environ.get("BNB_4BIT_QUANT_TYPE", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16_available else torch.float16,
                bnb_4bit_use_double_quant=os.environ.get("BNB_4BIT_USE_DOUBLE_QUANT", "1") == "1",
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if bf16_available else torch.float16 if cuda_available else None

        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        chat_state.update({"model": model, "tokenizer": tokenizer, "torch": torch, "adapter_dir": str(adapter_dir)})
        return chat_state


def model_input_device(model):
    for device in (getattr(model, "hf_device_map", None) or {}).values():
        if str(device) not in {"cpu", "disk", "meta"}:
            return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return "cpu"


def set_adapter_scale(model, scale: float) -> None:
    scale = max(0.0, min(scale, 2.0))
    if adapter_scale_state.get("value") == scale:
        return
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        base = getattr(module, "_base_scaling", None)
        if base is None:
            base = dict(scaling)
            setattr(module, "_base_scaling", base)
        for adapter_name, value in base.items():
            module.scaling[adapter_name] = value * scale
    adapter_scale_state["value"] = scale


def load_training_context() -> list[dict[str, str]]:
    if not CHAT_CONTEXT_FILE.exists():
        return []
    mtime = CHAT_CONTEXT_FILE.stat().st_mtime
    if context_cache["mtime"] == mtime:
        return context_cache["rows"]

    rows = []
    with CHAT_CONTEXT_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("path") and item.get("text"):
                rows.append({"path": item["path"], "chunk": item.get("chunk"), "text": item["text"]})
    context_cache.update({"mtime": mtime, "rows": rows})
    return rows


def retrieve_context(message: str, limit: int, max_chars: int) -> tuple[str, list[str]]:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", message)
        if term.lower() not in {"the", "and", "for", "with", "that", "this", "from", "what", "how", "can"}
    }
    scored = []
    for row in load_training_context():
        haystack = f"{row['path']}\n{row['text']}".lower()
        score = sum(haystack.count(term) for term in terms) or 1
        scored.append((score, row))

    snippets = []
    sources = []
    for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
        text = row["text"][:max_chars].rstrip()
        if len(row["text"]) > max_chars:
            text += "\n..."
        source = f"{row['path']}#chunk-{row['chunk']}" if row.get("chunk") else row["path"]
        sources.append(source)
        snippets.append(f"Source: {source}\n{text}")
    return "\n\n---\n\n".join(snippets), sources


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ping")
def ping():
    hf_token = env_hf_token()
    return jsonify({
        "ok": True,
        "storage_dir": str(STORAGE_DIR),
        "adapter_ready": (ADAPTER_DIR / "adapter_config.json").exists(),
        "agent_instructions": current_agent_instructions(),
        "defaults": form_defaults(),
        "env_files": {
            ".env": (BASE_DIR / ".env").exists(),
            ".env.example": (BASE_DIR / ".env.example").exists(),
        },
        "has_hf_token": bool(hf_token),
        "hf_token_length": len(hf_token),
    })


@app.route("/clear-memory", methods=["POST"])
def clear_memory_route():
    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is running. Stop it before clearing memory."}), 409
    unload_chat_model()
    clear_training_inputs()
    return jsonify({"message": "Cleared uploaded files and generated datasets. Adapter files were left alone."})


@app.route("/train", methods=["POST"])
def train():
    global training_process

    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is already running"}), 409

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400

    clear_training_inputs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for uploaded_file in uploaded_files:
        if not uploaded_file.filename:
            continue
        file_path = safe_upload_path(uploaded_file.filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded_file.save(file_path)
        saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "No valid files uploaded"}), 400

    unload_chat_model()

    child_env = os.environ.copy()
    child_env["TRAIN_STORAGE_DIR"] = str(STORAGE_DIR)
    child_env["TRAIN_OUTPUT_DIR"] = str(ADAPTER_DIR)
    child_env["AGENT_INSTRUCTIONS"] = request.form.get("agent_instructions", "").strip()
    AGENT_INSTRUCTIONS_FILE.write_text(child_env["AGENT_INSTRUCTIONS"], encoding="utf-8")

    controls = {
        "model_path": ("QWEN_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"),
        "assistant_speaker_name": ("ASSISTANT_SPEAKER_NAME", ""),
        "chat_context_min": ("CHAT_CONTEXT_MIN", "1"),
        "chat_context_max": ("CHAT_CONTEXT_MAX", "15"),
        "data_mode": ("DATA_MODE", "auto"),
        "dataset_seed": ("DATASET_SEED", "42"),
        "eval_ratio": ("EVAL_RATIO", "0.08"),
        "max_file_bytes": ("MAX_FILE_BYTES", "200000"),
        "max_conversation_bytes": ("MAX_CONVERSATION_BYTES", str(5 * 1024 * 1024)),
        "chunk_chars": ("CHUNK_CHARS", "3500"),
        "chunk_overlap": ("CHUNK_OVERLAP", "500"),
        "max_length": ("MAX_LENGTH", "1024"),
        "batch_size": ("BATCH_SIZE", "1"),
        "grad_accum_steps": ("GRAD_ACCUM_STEPS", "16"),
        "epochs": ("EPOCHS", "1"),
        "max_steps": ("MAX_STEPS", ""),
        "learning_rate": ("LEARNING_RATE", "5e-5"),
        "optim": ("OPTIM", ""),
        "lora_rank": ("LORA_R", "16"),
        "lora_alpha": ("LORA_ALPHA", "16"),
        "lora_dropout": ("LORA_DROPOUT", "0.05"),
        "lora_target_mode": ("LORA_TARGET_MODE", "all"),
        "lora_target_modules": ("LORA_TARGET_MODULES", ""),
    }
    for form_name, (env_name, default) in controls.items():
        set_child_env(child_env, form_name, env_name, default)

    child_env["ALLOW_MODEL_DOWNLOAD"] = "1" if bool_form("allow_model_download", True) else "0"
    child_env["USE_QLORA"] = "1" if bool_form("use_qlora", True) else "0"
    child_env["GPU_TRAIN"] = "1" if bool_form("gpu_train", True) else "0"
    child_env["FILTER_LOW_VALUE_REPLIES"] = "1" if bool_form("filter_low_value_replies", True) else "0"
    child_env["KEEP_SHORT_REPLIES"] = "1" if bool_form("keep_short_replies", True) else "0"
    child_env.setdefault("HF_HUB_DISABLE_XET", "1")
    child_env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    child_env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    hf_token = request.form.get("hf_token", "").strip()
    if hf_token:
        child_env["HF_TOKEN"] = hf_token
        child_env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    log_handle = LOG_FILE.open("w", encoding="utf-8")
    training_process = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=BASE_DIR,
        env=child_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()

    return jsonify({"message": "Training started", "files": saved_count})


@app.route("/stop", methods=["POST"])
def stop():
    global training_process
    if not training_process or training_process.poll() is not None:
        return jsonify({"message": "No training process is running"})
    training_process.terminate()
    try:
        training_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        training_process.kill()
        training_process.wait(timeout=10)
    return jsonify({"message": "Training stopped"})


@app.route("/status")
def status():
    running = training_process is not None and training_process.poll() is None
    exit_code = None if running or training_process is None else training_process.returncode
    log = LOG_FILE.read_text(encoding="utf-8", errors="replace") if LOG_FILE.exists() else ""
    return jsonify({
        "running": running,
        "exit_code": exit_code,
        "log": log[-10000:],
        "storage_dir": str(STORAGE_DIR),
        "adapter_ready": (ADAPTER_DIR / "adapter_config.json").exists(),
    })


@app.route("/chat", methods=["POST"])
def chat():
    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is running. Stop or finish training before chatting."}), 409

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        state = get_chat_model(ADAPTER_DIR)
        model = state["model"]
        tokenizer = state["tokenizer"]
        torch = state["torch"]
        set_adapter_scale(model, float(payload.get("adapter_scale") or 1.0))

        memory_context, sources = retrieve_context(
            message,
            int(payload.get("context_limit") or 6),
            int(payload.get("context_chars") or 1000),
        )
        messages = []
        instructions = current_agent_instructions()
        if instructions:
            messages.append({"role": "system", "content": instructions})
        for item in payload.get("history") or []:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        if memory_context:
            messages.append({"role": "user", "content": f"Relevant training data:\n\n{memory_context}\n\nMessage: {message}"})
        else:
            messages.append({"role": "user", "content": message})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model_input_device(model))
        temperature = float(payload.get("temperature") or 0.7)
        args = {
            **inputs,
            "max_new_tokens": int(payload.get("max_new_tokens") or 160),
            "do_sample": temperature > 0,
            "repetition_penalty": float(payload.get("repetition_penalty") or 1.1),
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            args["temperature"] = temperature
            args["top_p"] = float(payload.get("top_p") or 0.9)

        with chat_lock:
            with torch.no_grad():
                generated = model.generate(**args)
        reply = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        return jsonify({"reply": reply, "used_context": bool(memory_context), "sources": sources})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
