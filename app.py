from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import threading

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path, *, override: bool, protected_keys: set[str]) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        protected_value = os.environ.get(key) if key in protected_keys else None
        if override and (key not in protected_keys or not protected_value):
            os.environ[key] = value
        else:
            if not os.environ.get(key):
                os.environ[key] = value


PROTECTED_ENV_KEYS = set(os.environ)
load_env_file(BASE_DIR / ".env.example", override=False, protected_keys=PROTECTED_ENV_KEYS)
load_env_file(BASE_DIR / ".env", override=True, protected_keys=PROTECTED_ENV_KEYS)

STORAGE_DIR = Path(os.environ.get("TRAIN_STORAGE_DIR", BASE_DIR)).resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = STORAGE_DIR / "uploads"
LOG_FILE = STORAGE_DIR / "training.log"
TRAIN_SCRIPT = BASE_DIR / "train_model.py"
ADAPTER_DIR = STORAGE_DIR / "conversation-ai-lora"
CHAT_CONTEXT_FILE = STORAGE_DIR / "training_data.jsonl"
EVAL_CONTEXT_FILE = STORAGE_DIR / "eval_data.jsonl"
AGENT_INSTRUCTIONS_FILE = STORAGE_DIR / "agent_instructions.txt"
DEFAULT_TRAIN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RECOMMENDED_INFERENCE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
INFERENCE_ONLY_MODELS = {
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}
XAVIER_SYSTEM_PROMPT = (
    "You are Xavier. You answer helpfully, but your tone is casual, slightly chaotic, short, "
    "and WhatsApp-like. Use words like lol, lel, düd, wtf, ok, damn, nah, ye sometimes. "
    "Do not sound formal. Still answer the question properly."
)
DEFAULT_ADAPTER_SCALE = float(os.environ.get("ADAPTER_SCALE", "0.75"))
DEFAULT_CHAT_MAX_NEW_TOKENS = int(os.environ.get("CHAT_MAX_NEW_TOKENS", "160"))
DEFAULT_AGENT_INSTRUCTIONS = os.environ.get(
    "AGENT_INSTRUCTIONS",
    "Code professionally first. Be concise, practical, and direct. Preserve my casual writing style where it feels natural, "
    "but do not sacrifice correctness. Prioritize coding ability over personality.",
)
XAVIER_SYSTEM_PROMPT = DEFAULT_AGENT_INSTRUCTIONS
DEFAULT_AGENT_INSTRUCTIONS = os.environ.get("AGENT_INSTRUCTIONS", "")
XAVIER_SYSTEM_PROMPT = DEFAULT_AGENT_INSTRUCTIONS

app = Flask(__name__)
training_process = None
chat_state = {"model": None, "tokenizer": None, "torch": None}
chat_lock = threading.Lock()
context_cache = {"mtime": None, "rows": []}
adapter_scale_state = {"value": None}


def current_agent_instructions() -> str:
    if AGENT_INSTRUCTIONS_FILE.exists():
        text = AGENT_INSTRUCTIONS_FILE.read_text(encoding="utf-8", errors="replace").strip()
        return text
    return ""


def env_agent_instructions() -> str:
    return current_agent_instructions()


def env_hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def safe_upload_path(filename: str, upload_root: Path = UPLOAD_DIR) -> Path:
    parts = [secure_filename(part) for part in Path(filename).parts]
    parts = [part for part in parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("Invalid filename")
    return upload_root.joinpath(*parts)


def clear_memory() -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    if CHAT_CONTEXT_FILE.exists():
        CHAT_CONTEXT_FILE.unlink()
    if EVAL_CONTEXT_FILE.exists():
        EVAL_CONTEXT_FILE.unlink()
    context_cache.update({"mtime": None, "rows": []})


def unload_chat_model() -> None:
    with chat_lock:
        torch = chat_state.get("torch")
        chat_state.update({"model": None, "tokenizer": None, "torch": None})
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_adapter_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"No base model found in {config_path}")
    return base_model


def get_chat_model():
    if chat_state["model"] is not None and chat_state["tokenizer"] is not None:
        return chat_state

    with chat_lock:
        if chat_state["model"] is not None and chat_state["tokenizer"] is not None:
            return chat_state

        if not ADAPTER_DIR.exists():
            raise FileNotFoundError(f"Adapter folder does not exist: {ADAPTER_DIR}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_model = load_adapter_base_model(ADAPTER_DIR)
        cuda_available = torch.cuda.is_available()
        bf16_available = cuda_available and torch.cuda.is_bf16_supported()

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        except Exception as exc:
            if "vocab" not in str(exc).lower() or "merges" not in str(exc).lower():
                raise
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
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
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model.eval()

        if cuda_available:
            offloaded_devices = {
                str(device)
                for device in getattr(model, "hf_device_map", {}).values()
                if str(device) in {"cpu", "disk"}
            }
            if offloaded_devices:
                chat_state.update({"model": None, "tokenizer": None, "torch": None})
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "The adapter base model is still being offloaded to CPU/disk, so chat would be extremely slow. "
                    "Use a smaller trained adapter/base model for chat, or keep CHAT_USE_4BIT=1 and retrain/select a model that fits in VRAM."
                )

        chat_state.update({"model": model, "tokenizer": tokenizer, "torch": torch})
        return chat_state


def model_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None) or {}
    for device in hf_device_map.values():
        device_text = str(device)
        if device_text not in {"cpu", "disk", "meta"}:
            return device_text

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    return "cuda:0" if chat_state.get("torch") is not None and chat_state["torch"].cuda.is_available() else "cpu"


def blocked_token_ids(tokenizer):
    blocked_tokens = [
        token
        for token in ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"]
        if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id
    ]
    return [tokenizer(token, add_special_tokens=False).input_ids for token in blocked_tokens]


def adapter_scaling_items(model):
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter_name, value in list(scaling.items()):
                yield module, adapter_name, value


def set_adapter_scale(model, scale: float) -> None:
    scale = max(0.0, min(scale, 2.0))
    current = adapter_scale_state.get("value")
    if current == scale:
        return

    for module, adapter_name, value in adapter_scaling_items(model):
        base_scaling = getattr(module, "_xavier_base_scaling", None)
        if base_scaling is None:
            base_scaling = {}
            setattr(module, "_xavier_base_scaling", base_scaling)
        if adapter_name not in base_scaling:
            base_scaling[adapter_name] = value
        module.scaling[adapter_name] = base_scaling[adapter_name] * scale

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


def query_terms(message: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", message)
        if term.lower() not in {"the", "and", "for", "with", "that", "this", "from", "what", "how", "can"}
    }


def is_generated_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    if normalized in {"app.py", "train_model.py", "test_model.py"}:
        return True

    generated_parts = {
        ".dart_tool",
        "dart_tool",
        ".gradle",
        ".idea",
        "idea",
        "build",
        "bin",
        "obj",
        "ephemeral",
        "intermediates",
        "cmakefiles",
        "ds4windows_3.3.3_x64",
    }
    parts = {part.lower() for part in re.split(r"[\\/]+", normalized)}
    return bool(generated_parts & parts)


def clean_training_text(text: str) -> str:
    marker = "\nPath: "
    if marker in text:
        _, rest = text.split(marker, 1)
        if "\n\n" in rest:
            _, content = rest.split("\n\n", 1)
            return content

    marker = "\n\nFile: "
    if marker not in text:
        return text

    _, rest = text.split(marker, 1)
    if "\n\n" not in rest:
        return rest
    _, code = rest.split("\n\n", 1)
    return code


def retrieve_context(message: str, limit: int = 8, max_chars: int = 1200) -> tuple[str, list[str]]:
    rows = load_training_context()
    if not rows:
        return "", []

    terms = query_terms(message)
    scored = []
    for row in rows:
        if is_generated_path(row["path"]):
            continue
        clean_text = clean_training_text(row["text"])
        haystack = f"{row['path']}\n{clean_text}".lower()
        score = sum(haystack.count(term) for term in terms)
        lower_path = row["path"].lower()
        if lower_path.endswith(("pubspec.yaml", "package.json", "pyproject.toml", "requirements.txt")):
            score += 5
        if "/lib/" in lower_path or lower_path.startswith("lib/"):
            score += 3
        if score:
            scored.append((score, row))

    if not scored:
        scored = [(1, row) for row in rows if not is_generated_path(row["path"])]

    snippets = []
    sources = []
    for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]:
        text = clean_training_text(row["text"])
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n..."
        source = f"{row['path']}#chunk-{row['chunk']}" if row.get("chunk") else row["path"]
        sources.append(source)
        snippets.append(f"Memory source: {source}\n{text}")

    return "\n\n---\n\n".join(snippets), sources


def pack_chat_history(history: list[dict[str, str]], max_chars: int = 12000) -> list[dict[str, str]]:
    packed = []
    used_chars = 0

    for item in reversed(history):
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue

        cost = len(content) + len(role) + 8
        if packed and used_chars + cost > max_chars:
            break

        packed.append({"role": role, "content": content})
        used_chars += cost

    return list(reversed(packed))


def xavier_chat_messages(history: list[dict[str, str]], message: str, memory_context: str = "") -> list[dict[str, str]]:
    messages = []
    agent_instructions = current_agent_instructions()
    if agent_instructions:
        messages.append({"role": "system", "content": agent_instructions})
    messages.extend(pack_chat_history(history))
    if memory_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Relevant uploaded data snippets. Use them only if they help answer the message.\n\n"
                    f"{memory_context}\n\n"
                    f"Message: {message}"
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": message})
    return messages


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/ping")
def ping():
    hf_token = env_hf_token()
    return jsonify({
        "ok": True,
        "agent_instructions": env_agent_instructions(),
        "hf_token": hf_token,
        "has_hf_token": bool(hf_token),
        "hf_token_length": len(hf_token),
    })


@app.route("/clear-memory", methods=["POST"])
def clear_memory_route():
    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is running. Stop it before clearing memory."}), 409

    clear_memory()
    return jsonify({"message": "Cleared uploaded text memory. The trained adapter was not deleted."})


@app.route("/clear-project", methods=["POST"])
def clear_project():
    return clear_memory_route()


@app.route("/chat", methods=["POST"])
def chat():
    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is running. Stop or finish training before chatting."}), 409

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        state = get_chat_model()
        model = state["model"]
        tokenizer = state["tokenizer"]
        torch = state["torch"]
        set_adapter_scale(model, float(payload.get("adapter_scale") or DEFAULT_ADAPTER_SCALE))

        memory_context, sources = retrieve_context(message)
        messages = xavier_chat_messages(history, message, memory_context)

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model_input_device(model))
        temperature = float(payload.get("temperature") or 0.7)
        generation_args = {
            **inputs,
            "max_new_tokens": int(payload.get("max_new_tokens") or DEFAULT_CHAT_MAX_NEW_TOKENS),
            "do_sample": temperature > 0,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
            "bad_words_ids": blocked_token_ids(tokenizer) or None,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
            generation_args["top_p"] = float(payload.get("top_p") or 0.9)

        with chat_lock:
            with torch.no_grad():
                generated = model.generate(**generation_args)

        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return jsonify({"reply": reply, "used_context": bool(memory_context), "sources": sources})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/train", methods=["POST"])
def train():
    global training_process

    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is already running"}), 409

    file_groups = [
        ("personality", request.files.getlist("personality_files")),
        ("general", request.files.getlist("general_files")),
        ("general", request.files.getlist("files")),
    ]
    if not any(files for _, files in file_groups):
        return jsonify({"error": "No files uploaded"}), 400

    clear_memory()
    UPLOAD_DIR.mkdir(exist_ok=True)
    saved_count = 0

    for group_name, files in file_groups:
        group_dir = UPLOAD_DIR / group_name
        for uploaded_file in files:
            if not uploaded_file.filename:
                continue

            file_path = safe_upload_path(uploaded_file.filename, group_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            uploaded_file.save(file_path)
            saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "No valid files uploaded"}), 400

    unload_chat_model()

    child_env = os.environ.copy()
    child_env["TRAIN_STORAGE_DIR"] = str(STORAGE_DIR)
    agent_instructions = request.form.get("agent_instructions", "").strip()
    AGENT_INSTRUCTIONS_FILE.write_text(agent_instructions, encoding="utf-8")
    child_env["AGENT_INSTRUCTIONS"] = agent_instructions

    assistant_speaker_name = request.form.get("assistant_speaker_name", "").strip()
    child_env["ASSISTANT_SPEAKER_NAME"] = assistant_speaker_name
    child_env["PERSONALITY_WEIGHT"] = request.form.get("personality_weight", "40").strip() or "40"
    child_env["GENERAL_WEIGHT"] = request.form.get("general_weight", "60").strip() or "60"
    child_env["CHAT_CONTEXT_MIN"] = request.form.get("chat_context_min", "1").strip() or "1"
    child_env["CHAT_CONTEXT_MAX"] = request.form.get("chat_context_max", "15").strip() or "15"

    hf_token = request.form.get("hf_token", "").strip()
    if hf_token:
        child_env["HF_TOKEN"] = hf_token
        child_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    elif child_env.get("HF_TOKEN"):
        child_env["HUGGING_FACE_HUB_TOKEN"] = child_env["HF_TOKEN"]
    elif child_env.get("HUGGING_FACE_HUB_TOKEN"):
        child_env["HF_TOKEN"] = child_env["HUGGING_FACE_HUB_TOKEN"]

    model_path = request.form.get("model_path", "").strip()
    if model_path in INFERENCE_ONLY_MODELS:
        return jsonify({"error": f"{model_path} is not practical for local QLoRA training on this GPU. Choose 7B, 3B, 1.5B, 0.5B, or a local custom model."}), 400
    if model_path:
        child_env["QWEN_MODEL_PATH"] = model_path
        if request.form.get("allow_model_download") == "1" and "/" in model_path:
            child_env["ALLOW_MODEL_DOWNLOAD"] = "1"
    elif request.form.get("allow_model_download") == "1":
        child_env["QWEN_MODEL_PATH"] = DEFAULT_TRAIN_MODEL
        child_env["ALLOW_MODEL_DOWNLOAD"] = "1"
    child_env.setdefault("TRAIN_OUTPUT_DIR", str(ADAPTER_DIR))

    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    child_env.setdefault("HF_HUB_DISABLE_XET", "1")
    child_env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    child_env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    child_env.setdefault("USE_QLORA", "1")
    child_env.setdefault("LORA_TARGET_MODE", "all")

    if request.form.get("quick_train") == "1":
        child_env["QUICK_TRAIN"] = "1"
        child_env["MAX_STEPS"] = "5"
        child_env["MAX_LENGTH"] = "256"
        child_env["LORA_R"] = "4"
        child_env["LORA_ALPHA"] = "8"
        child_env["GRAD_ACCUM_STEPS"] = "1"
        child_env["EPOCHS"] = "1"

    if request.form.get("gpu_train") == "1" and request.form.get("quick_train") != "1":
        child_env["GPU_TRAIN"] = "1"
        child_env["MAX_LENGTH"] = request.form.get("max_length", "").strip() or child_env.get("MAX_LENGTH", "1024")
        child_env.setdefault("BATCH_SIZE", "1")
        child_env.setdefault("GRAD_ACCUM_STEPS", "16")
        child_env["LORA_R"] = request.form.get("lora_rank", "").strip() or child_env.get("LORA_R", "16")
        child_env["LORA_ALPHA"] = request.form.get("lora_alpha", "").strip() or child_env.get("LORA_ALPHA", child_env["LORA_R"])
        child_env.setdefault("LORA_DROPOUT", "0.05")
        child_env["LEARNING_RATE"] = request.form.get("learning_rate", "").strip() or child_env.get("LEARNING_RATE", "5e-5")
        child_env["EPOCHS"] = request.form.get("epochs", "").strip() or child_env.get("EPOCHS", "1")
    elif request.form.get("gpu_train") == "1":
        child_env["GPU_TRAIN"] = "1"
    elif request.form.get("quick_train") != "1":
        child_env["MAX_LENGTH"] = request.form.get("max_length", "").strip() or child_env.get("MAX_LENGTH", "1536")
        child_env["LORA_R"] = request.form.get("lora_rank", "").strip() or child_env.get("LORA_R", "16")
        child_env["LORA_ALPHA"] = request.form.get("lora_alpha", "").strip() or child_env.get("LORA_ALPHA", child_env["LORA_R"])
        child_env["LEARNING_RATE"] = request.form.get("learning_rate", "").strip() or child_env.get("LEARNING_RATE", "5e-5")
        child_env["EPOCHS"] = request.form.get("epochs", "").strip() or child_env.get("EPOCHS", "1")

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
    return jsonify({"running": running, "exit_code": exit_code, "log": log[-8000:], "storage_dir": str(STORAGE_DIR)})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
