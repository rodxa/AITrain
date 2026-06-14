from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
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
export_jobs: dict[str, dict] = {}
imported_adapter_hf_token = {"value": ""}


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
        "model_path": env_value("BASE_MODEL_PATH", "QWEN_MODEL_PATH", "MODEL_PATH", default="google/gemma-4-12B-it"),
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
        "chat_context_min": env_value("CHAT_CONTEXT_MIN", default="1"),
        "chat_context_max": env_value("CHAT_CONTEXT_MAX", default="15"),
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


def detected_gpu_vram_gb() -> tuple[str, float | None]:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.name, props.total_memory / 1024**3
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "No CUDA GPU detected", None

    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "GPU detection unavailable", None

    first_line = output.splitlines()[0] if output else ""
    if not first_line:
        return "GPU detection unavailable", None
    name, _, memory_mb = first_line.rpartition(",")
    try:
        return name.strip() or "CUDA GPU", float(memory_mb.strip()) / 1024
    except ValueError:
        return name.strip() or "CUDA GPU", None


def spec_preset() -> dict:
    gpu_name, vram_gb = detected_gpu_vram_gb()
    values = {
        "gpu_train": vram_gb is not None,
        "use_qlora": True,
        "allow_model_download": True,
        "batch_size": "1",
        "grad_accum_steps": "32",
        "epochs": "1",
        "max_steps": "",
        "learning_rate": "5e-5",
        "optim": "paged_adamw_8bit" if vram_gb is not None else "adamw_torch",
        "max_length": "512",
        "chunk_chars": "2500",
        "chunk_overlap": "300",
        "lora_rank": "8",
        "lora_alpha": "16",
        "lora_dropout": "0.05",
        "lora_target_mode": "qv",
        "lora_target_modules": "",
    }

    if vram_gb is None:
        message = "Applied a conservative CPU/no-GPU preset."
    elif vram_gb < 8:
        message = f"Applied a low-VRAM preset for {gpu_name} ({vram_gb:.1f} GiB VRAM)."
    elif vram_gb < 12:
        values.update({
            "grad_accum_steps": "24",
            "max_length": "768",
            "chunk_chars": "3000",
            "chunk_overlap": "400",
            "lora_target_mode": "attention",
        })
        message = f"Applied a balanced preset for {gpu_name} ({vram_gb:.1f} GiB VRAM)."
    else:
        values.update({
            "grad_accum_steps": "16",
            "max_length": "1024",
            "chunk_chars": "3500",
            "chunk_overlap": "500",
            "lora_rank": "16",
            "lora_target_mode": "all",
        })
        message = f"Applied a roomier preset for {gpu_name} ({vram_gb:.1f} GiB VRAM)."

    return {"message": message, "gpu_name": gpu_name, "vram_gb": vram_gb, "values": values}


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


def ensure_adapter_idle() -> tuple[dict, int] | None:
    if training_process and training_process.poll() is None:
        return {"error": "Training is running. Stop it before changing adapter files."}, 409
    return None


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = target_dir / member.filename
            resolved = member_path.resolve()
            if target_dir != resolved and target_dir not in resolved.parents:
                raise ValueError(f"Unsafe zip path: {member.filename}")
        archive.extractall(target_dir)


def find_adapter_root(path: Path) -> Path:
    direct_config = path / "adapter_config.json"
    if direct_config.exists():
        return path

    matches = [item.parent for item in path.rglob("adapter_config.json")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError("Uploaded zip does not contain adapter_config.json.")
    raise ValueError("Uploaded zip contains multiple adapter_config.json files.")


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
        hf_token = imported_adapter_hf_token.get("value") or env_hf_token() or None
        cuda_available = torch.cuda.is_available()
        bf16_available = cuda_available and torch.cuda.is_bf16_supported()
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=hf_token)
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

        model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token, **model_kwargs)
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


def generate_chat_reply(payload: dict) -> dict:
    message = (payload.get("message") or "").strip()
    if not message:
        raise ValueError("Message is required")

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
    return {"reply": reply, "used_context": bool(memory_context), "sources": sources}


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
        "chat_model_loaded": chat_state["model"] is not None,
        "agent_instructions": current_agent_instructions(),
        "defaults": form_defaults(),
        "env_files": {
            ".env": (BASE_DIR / ".env").exists(),
            ".env.example": (BASE_DIR / ".env.example").exists(),
        },
        "has_hf_token": bool(hf_token),
        "hf_token_length": len(hf_token),
    })


@app.route("/spec-preset")
def spec_preset_route():
    return jsonify(spec_preset())


@app.route("/clear-memory", methods=["POST"])
def clear_memory_route():
    if training_process and training_process.poll() is None:
        return jsonify({"error": "Training is running. Stop it before clearing memory."}), 409
    unload_chat_model()
    clear_training_inputs()
    return jsonify({"message": "Cleared uploaded files and generated datasets. Adapter files were left alone."})


@app.route("/adapter/export")
def export_adapter_route():
    idle_error = ensure_adapter_idle()
    if idle_error:
        payload, status_code = idle_error
        return jsonify(payload), status_code
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        return jsonify({"error": "No adapter found to export."}), 404

    tmp_dir = Path(tempfile.mkdtemp(prefix="adapter-export-"))
    archive_path = Path(shutil.make_archive(str(tmp_dir / ADAPTER_DIR.name), "zip", ADAPTER_DIR))
    response = send_file(
        archive_path,
        as_attachment=True,
        download_name=f"{ADAPTER_DIR.name}.zip",
        mimetype="application/zip",
        conditional=False,
        max_age=0,
    )

    @response.call_on_close
    def cleanup_export_archive():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return response


@app.route("/adapter/export/start", methods=["POST"])
def start_adapter_export_route():
    idle_error = ensure_adapter_idle()
    if idle_error:
        payload, status_code = idle_error
        return jsonify(payload), status_code
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        return jsonify({"error": "No adapter found to export."}), 404

    job_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.mkdtemp(prefix="adapter-export-"))
    export_jobs[job_id] = {"status": "running", "tmp_dir": tmp_dir, "archive_path": None, "error": ""}

    def build_export() -> None:
        try:
            archive_path = Path(shutil.make_archive(str(tmp_dir / ADAPTER_DIR.name), "zip", ADAPTER_DIR))
            export_jobs[job_id].update({"status": "ready", "archive_path": archive_path})
        except Exception as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            export_jobs[job_id].update({"status": "error", "error": str(exc)})

    threading.Thread(target=build_export, daemon=True).start()
    return jsonify({"job_id": job_id, "message": "Adapter export is being prepared."})


@app.route("/adapter/export/status/<job_id>")
def adapter_export_status_route(job_id: str):
    job = export_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Export job not found."}), 404
    payload = {"status": job["status"]}
    if job["status"] == "ready":
        payload["download_url"] = f"/adapter/export/download/{job_id}"
    if job["status"] == "error":
        payload["error"] = job.get("error") or "Could not prepare adapter export."
    return jsonify(payload)


@app.route("/adapter/export/download/<job_id>")
def download_adapter_export_route(job_id: str):
    job = export_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Export job not found."}), 404
    if job["status"] != "ready" or not job.get("archive_path"):
        return jsonify({"error": "Export is not ready yet."}), 409

    archive_path = Path(job["archive_path"])
    response = send_file(
        archive_path,
        as_attachment=True,
        download_name=f"{ADAPTER_DIR.name}.zip",
        mimetype="application/zip",
        conditional=False,
        max_age=0,
    )

    @response.call_on_close
    def cleanup_export_job():
        finished = export_jobs.pop(job_id, None)
        if finished:
            shutil.rmtree(finished["tmp_dir"], ignore_errors=True)

    return response


@app.route("/adapter/import", methods=["POST"])
def import_adapter_route():
    idle_error = ensure_adapter_idle()
    if idle_error:
        payload, status_code = idle_error
        return jsonify(payload), status_code

    uploaded_file = request.files.get("adapter")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "Choose an adapter zip first."}), 400
    if not uploaded_file.filename.lower().endswith(".zip"):
        return jsonify({"error": "Adapter import expects a .zip file."}), 400

    hf_token = request.form.get("hf_token", "").strip()
    unload_chat_model()
    with tempfile.TemporaryDirectory(prefix="adapter-import-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "adapter.zip"
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        uploaded_file.save(zip_path)
        try:
            safe_extract_zip(zip_path, extract_dir)
            adapter_root = find_adapter_root(extract_dir)
        except (OSError, ValueError, zipfile.BadZipFile, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

        backup_dir = ADAPTER_DIR.with_name(f"{ADAPTER_DIR.name}.backup-import")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if ADAPTER_DIR.exists():
            ADAPTER_DIR.rename(backup_dir)
        try:
            shutil.copytree(adapter_root, ADAPTER_DIR)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
        except Exception:
            if ADAPTER_DIR.exists():
                shutil.rmtree(ADAPTER_DIR)
            if backup_dir.exists():
                backup_dir.rename(ADAPTER_DIR)
            raise

    imported_adapter_hf_token["value"] = hf_token
    return jsonify({
        "message": f"Imported adapter to {ADAPTER_DIR}.",
        "has_hf_token": bool(hf_token),
    })


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
        "model_path": ("BASE_MODEL_PATH", "google/gemma-4-12B-it"),
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
    imported_adapter_hf_token["value"] = hf_token
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
        "chat_model_loaded": chat_state["model"] is not None,
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
        return jsonify(generate_chat_reply(payload))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
