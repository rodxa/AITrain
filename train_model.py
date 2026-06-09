from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.environ.get("TRAIN_STORAGE_DIR", BASE_DIR)).resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = STORAGE_DIR / "uploads"
DATASET_FILE = STORAGE_DIR / "training_data.jsonl"
EVAL_DATASET_FILE = STORAGE_DIR / "eval_data.jsonl"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
OUTPUT_DIR = Path(os.environ.get("TRAIN_OUTPUT_DIR", STORAGE_DIR / "conversation-ai-lora"))
DEFAULT_TRAIN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RECOMMENDED_INFERENCE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", os.environ.get("MODEL_PATH", DEFAULT_TRAIN_MODEL))
ALLOW_MODEL_DOWNLOAD = os.environ.get("ALLOW_MODEL_DOWNLOAD", "0") == "1"
USE_QLORA = os.environ.get("USE_QLORA", "1") == "1"
REQUIRED_MODEL_FILES = {"config.json"}
TOKENIZER_FILES = {"tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"}
LORA_TARGET_PRESETS = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qv": ["q_proj", "v_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

BINARY_EXTENSIONS = {
    ".7z",
    ".a",
    ".apk",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".ico",
    ".jar",
    ".jpg",
    ".jpeg",
    ".keystore",
    ".mp3",
    ".mp4",
    ".o",
    ".obj",
    ".pdf",
    ".png",
    ".so",
    ".ttf",
    ".webp",
    ".woff",
    ".woff2",
    ".zip",
}
CONVERSATION_EXTENSIONS = {".txt", ".log", ".chat", ".md", ".json", ".jsonl"}
DEFAULT_MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_BYTES", "200000"))
MAX_CONVERSATION_BYTES = int(os.environ.get("MAX_CONVERSATION_BYTES", str(5 * 1024 * 1024)))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "3500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "500"))
ASSISTANT_SPEAKER_NAME = os.environ.get("ASSISTANT_SPEAKER_NAME", "").strip()
AGENT_INSTRUCTIONS = os.environ.get("AGENT_INSTRUCTIONS", "").strip()
CONTEXT_MIN_MESSAGES = int(os.environ.get("CHAT_CONTEXT_MIN", os.environ.get("WHATSAPP_CONTEXT_MIN", "1")))
CONTEXT_MAX_MESSAGES = int(os.environ.get("CHAT_CONTEXT_MAX", os.environ.get("WHATSAPP_CONTEXT_MAX", "15")))
EVAL_RATIO = float(os.environ.get("EVAL_RATIO", "0.08"))
DATASET_SEED = int(os.environ.get("DATASET_SEED", "42"))
DATA_MODE = os.environ.get("DATA_MODE", "auto").lower()
FILTER_LOW_VALUE_REPLIES = os.environ.get("FILTER_LOW_VALUE_REPLIES", "1") == "1"
KEEP_SHORT_TARGETS = os.environ.get("KEEP_SHORT_REPLIES", "1") == "1"
LORA_TARGET_MODE = os.environ.get("LORA_TARGET_MODE", "all").lower()
STYLE_SYSTEM_PROMPT = AGENT_INSTRUCTIONS
HELPFUL_SYSTEM_PROMPT = AGENT_INSTRUCTIONS
USELESS_REPLY_EXACT = {
    "",
    "?",
    "??",
    "???",
    "ok",
    "okay",
    "k",
    "kk",
    "haha",
    "ahah",
    "hahaha",
}
KEEP_SHORT_REPLIES = {"lol", "lel", "dÃ¼d", "dud", "wtf", "ye", "yes", "nah", "no", "ok wait", "damn", "lmao", "lmfao"}
MEDIA_OR_DELETED_PATTERNS = [
    "media omitted",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "this message was deleted",
    "message deleted",
    "deleted message",
    "<media omitted>",
]
URL_ONLY_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
WHATSAPP_LINE_RES = [
    re.compile(
        r"^\[?\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\]?\s*[-â€“]\s*(?P<speaker>[^:]+):\s*(?P<text>.*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\s*-\s*(?P<speaker>[^:]+):\s*(?P<text>.*)$",
        re.IGNORECASE,
    ),
]


def read_training_file(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return None
    max_bytes = MAX_CONVERSATION_BYTES if path.suffix.lower() in CONVERSATION_EXTENSIONS else DEFAULT_MAX_FILE_BYTES
    if path.stat().st_size > max_bytes:
        return None

    try:
        data = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in data[:4096]:
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def json_to_conversation_text(value, depth: int = 0) -> str:
    if depth > 8:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)

    if isinstance(value, list):
        parts = [json_to_conversation_text(item, depth + 1) for item in value]
        return "\n\n".join(part for part in parts if part)

    if isinstance(value, dict):
        role = value.get("role") or value.get("speaker") or value.get("author") or value.get("name") or value.get("from")
        content = (
            value.get("content")
            or value.get("message")
            or value.get("text")
            or value.get("body")
            or value.get("value")
            or value.get("response")
        )
        if content is not None:
            content_text = json_to_conversation_text(content, depth + 1)
            return f"{role}: {content_text}" if role else content_text

        parts = []
        for key, item in value.items():
            item_text = json_to_conversation_text(item, depth + 1)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "\n".join(parts)

    return ""


def parse_json_records(content: str) -> list[dict]:
    try:
        value = json.loads(content)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [value]
    except json.JSONDecodeError:
        pass

    records = []
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def message_content(message: dict) -> str:
    content = message.get("content") or message.get("text") or message.get("message") or message.get("body")
    return str(content).strip() if content is not None else ""


def normalize_speaker(value: str) -> str:
    value = value.strip()
    if not value:
        return "user"
    return "assistant" if value.lower() == ASSISTANT_SPEAKER_NAME.lower() else "user"


def clean_message_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u200e", "").replace("\ufeff", "")).strip()


def is_bad_message(text: str) -> bool:
    normalized = clean_message_text(text).lower()
    if not normalized:
        return True
    if URL_ONLY_RE.match(normalized):
        return True
    return any(pattern in normalized for pattern in MEDIA_OR_DELETED_PATTERNS)


def is_useless_target_reply(text: str) -> bool:
    if not FILTER_LOW_VALUE_REPLIES:
        return False
    normalized = clean_message_text(text).lower().strip(" .,!;:-_")
    if normalized in USELESS_REPLY_EXACT:
        return True
    if URL_ONLY_RE.match(normalized):
        return True
    return any(pattern in normalized for pattern in MEDIA_OR_DELETED_PATTERNS)


def is_kept_short_reply(text: str) -> bool:
    if not KEEP_SHORT_TARGETS:
        return False
    normalized = clean_message_text(text).lower().strip(" .,!;:-_")
    return normalized in KEEP_SHORT_REPLIES


def has_understandable_short_reply_context(history: list[dict[str, str]]) -> bool:
    if len(history) < 2:
        return False

    recent = history[-5:]
    recent_user_text = " ".join(item["content"] for item in recent if item["role"] == "user")
    if len(recent_user_text) < 12:
        return False

    # Keep reaction-style Xavier replies when there is enough immediately preceding chat to react to.
    has_real_sentence = any(len(item["content"].split()) >= 3 for item in recent if item["role"] == "user")
    has_multi_turn_context = sum(1 for item in recent if item["role"] == "user") >= 2 or any(
        item["role"] == "assistant" for item in recent[:-1]
    )
    return has_real_sentence and has_multi_turn_context


def role_from_json_message(message: dict) -> str:
    role = str(
        message.get("role")
        or message.get("speaker")
        or message.get("author")
        or message.get("name")
        or message.get("from")
        or ""
    ).strip()
    if role.lower() in {"assistant", "model"}:
        return "assistant"
    if role.lower() in {"user", "human"}:
        return "user"
    if role.lower() == "system":
        return "system"
    return normalize_speaker(role)


def parse_json_chat_messages(content: str) -> list[dict[str, str]]:
    records = parse_json_records(content)
    messages: list[dict[str, str]] = []

    for record in records:
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            continue

        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            content_text = clean_message_text(message_content(item))
            if is_bad_message(content_text):
                continue
            role = role_from_json_message(item)
            if role not in {"system", "user", "assistant"}:
                continue
            messages.append({"role": role, "content": content_text})

    return messages


def parse_whatsapp_txt_messages(content: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = None
        for pattern in WHATSAPP_LINE_RES:
            match = pattern.match(line)
            if match:
                break

        if match:
            if current and not is_bad_message(current["content"]):
                messages.append(current)
            current = {
                "role": normalize_speaker(match.group("speaker")),
                "content": clean_message_text(match.group("text")),
            }
            continue

        if current:
            current["content"] = clean_message_text(f"{current['content']}\n{line}")

    if current and not is_bad_message(current["content"]):
        messages.append(current)

    return messages


def coalesce_consecutive_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    coalesced: list[dict[str, str]] = []
    for message in messages:
        content = clean_message_text(message["content"])
        if is_bad_message(content):
            continue
        if coalesced and coalesced[-1]["role"] == message["role"]:
            coalesced[-1]["content"] = clean_message_text(f"{coalesced[-1]['content']}\n{content}")
        else:
            coalesced.append({"role": message["role"], "content": content})
    return coalesced


def build_style_examples_from_messages(path: str, messages: list[dict[str, str]], source_kind: str) -> list[dict]:
    rows: list[dict] = []
    messages = coalesce_consecutive_messages(messages)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        history = messages[max(0, index - CONTEXT_MAX_MESSAGES) : index]
        if not history:
            continue
        if not any(item["role"] == "user" for item in history):
            continue

        short_reply = is_kept_short_reply(message["content"])
        if short_reply:
            if not has_understandable_short_reply_context(history):
                continue
        elif is_useless_target_reply(message["content"]):
            continue
        elif len(history) < CONTEXT_MIN_MESSAGES and index >= CONTEXT_MIN_MESSAGES:
            continue

        row_messages = with_optional_system(history, message, STYLE_SYSTEM_PROMPT)
        rows.append(
            {
                "path": path,
                "chunk": len(rows) + 1,
                "source": f"{source_kind}_conversation",
                "messages": row_messages,
                "text": messages_to_plain_text(row_messages),
            }
        )

    return rows


def build_reply_examples(path: str, content: str, suffix: str, source_kind: str) -> list[dict]:
    if DATA_MODE == "context":
        return []

    messages: list[dict[str, str]] = []
    if suffix in {".json", ".jsonl"}:
        messages = parse_json_chat_messages(content)
    elif suffix == ".txt":
        messages = parse_whatsapp_txt_messages(content)

    if not messages:
        return []

    return build_style_examples_from_messages(path, messages, source_kind)


def messages_to_plain_text(messages: list[dict[str, str]]) -> str:
    names = {"system": "System", "user": "User", "assistant": ASSISTANT_SPEAKER_NAME or "Assistant"}
    return "\n".join(f"{names.get(item['role'], item['role'])}: {item['content']}" for item in messages)


def with_optional_system(
    history: list[dict[str, str]],
    assistant_message: dict[str, str],
    system_prompt: str = "",
) -> list[dict[str, str]]:
    messages = []
    system_prompt = clean_message_text(system_prompt)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append(assistant_message)
    return messages


def legacy_helpful_examples_unused() -> list[dict]:
    examples = [
        (
            "How do I fix this Dart error: The argument type 'String?' can't be assigned to the parameter type 'String'?",
            "That means Dart thinks the value might be null, but the function needs a guaranteed String. Best fixes: handle null with a fallback like `name ?? ''`, check it before calling, or change the receiving function to accept `String?` if null is valid. Avoid just using `!` unless you're 100% sure it cannot be null, cuz that can crash at runtime.",
        ),
        (
            "Explain LoRA simply",
            "LoRA is a way to fine-tune a model by training small extra matrices instead of changing all the original weights. So the base model keeps its general brain, and the adapter learns the new behavior cheaply. Think of it as adding a tiny style/skill patch instead of rebuilding the whole model, lel.",
        ),
        (
            "why is my AI just repeating me?",
            "Usually it learned from examples where the target reply looks too much like the prompt, or the training labels include user text instead of masking it. Use chat-template formatting, mask prompt tokens so loss is only on assistant replies, filter low-value examples, lower the learning rate, and train fewer epochs. Also test with eval prompts so you catch repetition early.",
        ),
        (
            "wanna play gartic?",
            "yeah sure, send the room link. I can play for a bit lol",
        ),
        (
            "Write a Python function that checks whether a string is a palindrome.",
            "Here you go:\n\n```python\ndef is_palindrome(text: str) -> bool:\n    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n    return cleaned == cleaned[::-1]\n```\n\nIt ignores spaces, punctuation, and casing, so `Race car` works too.",
        ),
        (
            "How should I train a style LoRA without making the model useless?",
            "Mix mostly helpful instruction data with a smaller amount of style data. For your case, 70% helpful examples and 30% Xavier chat style is a good starting point. Keep 1-2 epochs, low LR like `5e-5` to `1e-4`, LoRA rank `8` or `16`, and remove target replies that are only `ok`, `lol`, `?`, links, deleted messages, or media placeholders. The model should learn the vibe without forgetting how to answer, dÃ¼d.",
        ),
        (
            "What is overfitting in fine-tuning?",
            "Overfitting is when the model memorizes the training examples too hard and gets worse at general prompts. In a chat-style LoRA it can show up as repeating exact phrases, replying too short, copying the user, or using slang everywhere. Reduce epochs, lower LR, add eval samples, and mix in normal assistant data.",
        ),
        (
            "Give me a quick checklist for debugging a Python script that crashes.",
            "Start with the exact traceback, then find the first line that belongs to your code. Print or inspect the values going into that line, check types, check missing files/env vars, and try the smallest input that still breaks. Once you fix it, add a tiny test so it doesn't come back, dÃ¼d.",
        ),
        (
            "How do I make a LoRA sound like me without copying my private chats too hard?",
            "Use chats as style examples, not as the whole dataset. Filter boring one-word replies, remove private/sensitive stuff, add a helpful system prompt, and mix in normal assistant tasks. Train lightly: low LR, 1 epoch, small rank. The model should learn rhythm and slang, not memorize your conversations.",
        ),
        (
            "Explain gradient accumulation.",
            "Gradient accumulation lets you simulate a bigger batch size by doing several small forward/backward passes before one optimizer step. So if your GPU only fits batch size 1, `grad_accum=8` behaves more like batch size 8. It is slower, but saves VRAM.",
        ),
        (
            "What should I do if my model answers with only 'ok'?",
            "That usually means your target examples contain too many low-information replies. Filter targets that are only `ok`, `lol`, `?`, links, deleted messages, or media placeholders. Also add helpful Q&A/code examples and use a system prompt that says to answer usefully first.",
        ),
        (
            "Can you review this error: ModuleNotFoundError: No module named 'peft'?",
            "Python can't find the `peft` package in the environment you're running. Install it with `python -m pip install peft`, but make sure you're using the same Python/venv that runs the script. Quick check: `python -m pip show peft`.",
        ),
        (
            "What is the difference between base model and adapter?",
            "The base model is the full pretrained model with the general knowledge and language ability. The adapter is the small trained LoRA layer you add on top for your task/style. At inference you load both: base first, adapter second.",
        ),
        (
            "How can I stop my fine-tuned model from repeating the prompt?",
            "Use proper chat formatting, mask the prompt tokens in the loss, avoid examples where the assistant mirrors the user, add a repetition penalty during generation, and train less aggressively. Repetition is often a data/labeling issue, not just a decoding issue.",
        ),
        (
            "Write a short JavaScript debounce helper.",
            "Sure:\n\n```js\nfunction debounce(fn, delay = 250) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n}\n```\n\nUse it for search boxes, resize handlers, stuff that fires too often lol.",
        ),
        (
            "What does learning rate do?",
            "Learning rate controls how big each training update is. Too high and the model can forget useful behavior or become unstable. Too low and it barely learns. For a style LoRA, start low, like `5e-5` to `1e-4`, because you want a nudge, not a personality wipe.",
        ),
    ]

    built_in_rows = [
        {
            "path": "built-in/helpful_examples",
            "chunk": index,
            "source": "helpful",
            "messages": [
                {"role": "system", "content": HELPFUL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "text": messages_to_plain_text(
                [
                    {"role": "system", "content": HELPFUL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            ),
        }
        for index, (prompt, response) in enumerate(examples, start=1)
    ]

    if not HELPFUL_DATA_FILE:
        return built_in_rows

    path = Path(HELPFUL_DATA_FILE)
    if not path.exists():
        print(f"Helpful data file not found, using built-in examples only: {path}", flush=True)
        return built_in_rows

    extra_rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if isinstance(messages, list) and messages and messages[-1].get("role") == "assistant":
            extra_rows.append(
                {
                    "path": str(path),
                    "chunk": line_number,
                    "source": "helpful",
                    "messages": messages,
                    "text": messages_to_plain_text(messages),
                }
            )

    return [*built_in_rows, *extra_rows]


def helpful_examples() -> list[dict]:
    examples = [
        (
            "How do I fix this Dart error: The argument type 'String?' can't be assigned to the parameter type 'String'?",
            "dart is saying that value can be null but the function wants a real `String`.\n\nuse fallback: `name ?? ''`, or check it first:\n\n```dart\nif (name != null) {\n  doThing(name);\n}\n```\n\nonly use `name!` if ur sure, otherwise boom lol",
        ),
        (
            "Explain LoRA simply",
            "LoRA is a tiny add-on brain for the model.\n\nyou don't retrain the whole thing, just small adapter weights. cheaper, faster, less VRAM. base model stays smart, adapter learns the vibe/task. like a patch, lel",
        ),
        (
            "why is my AI repeating me?",
            "usually labels/data are cooked.\n\ncheck this:\n- prompt tokens should be masked\n- train only on assistant reply\n- use Qwen chat template\n- remove mirror-y examples\n- lower LR/epochs\n\nrepetition is often training setup, not just generation. wtf but fixable",
        ),
        (
            "why is my AI just repeating me?",
            "same thing: it probably learned user text as target text.\n\nmask the prompt, keep loss only on assistant output, filter examples where reply copies the user, and train lighter. add repetition penalty too, but data matters more lol",
        ),
        (
            "wanna play gartic?",
            "ye send room link, i can play for a bit lol",
        ),
        (
            "Write a Python function that checks whether a string is a palindrome.",
            "ye here:\n\n```python\ndef is_palindrome(text: str) -> bool:\n    s = ''.join(c.lower() for c in text if c.isalnum())\n    return s == s[::-1]\n```\n\nignores spaces and punctuation too",
        ),
        (
            "How should I train a style LoRA without making the model useless?",
            "go 50/50 or 60/40 style/helpful, not tiny style crumbs.\n\n1 epoch, low LR like `5e-5` to `1e-4`, LoRA `r=8` or `16`. keep short replies only with context, remove media/deleted/link-only trash. rewrite helpful answers in your tone too, dÃ¼d",
        ),
        (
            "What is overfitting in fine-tuning?",
            "overfitting = model memorizes dataset too hard and gets worse outside it.\n\nfor style LoRA it starts saying only `lol`, copying user, repeating phrases, slang every sentence. fix with fewer epochs, lower LR, eval split, better data mix. damn annoying but ye",
        ),
        (
            "Give me a quick checklist for debugging a Python script that crashes.",
            "ok quick:\n\n- read traceback bottom-up\n- find first line that is your code\n- inspect values there\n- check types/files/env vars\n- reproduce with tiny input\n- add small test after\n\nboring but works lol",
        ),
        (
            "How do I make a LoRA sound like me without copying my private chats too hard?",
            "use chats for rhythm/slang, not the whole brain.\n\nremove private stuff, keep context windows, mix helpful examples rewritten like you, train light. goal is vibe transfer, not memorizing old whatsapp nonsense lol",
        ),
        (
            "Explain gradient accumulation.",
            "gradient accumulation = fake bigger batch.\n\nif GPU fits batch 1, `grad_accum=8` does 8 little steps before updating. acts like batch 8-ish. slower, saves VRAM. ye thats basically it",
        ),
        (
            "What should I do if my model answers with only 'ok'?",
            "it learned too many dead replies.\n\nkeep `lol`/`nah`/`damn` only when context explains them. remove `?`, media omitted, deleted msg, link-only stuff. add helpful Xavier-style answers so it learns to answer, not just react.",
        ),
        (
            "Can you review this error: ModuleNotFoundError: No module named 'peft'?",
            "python can't find `peft` in the env ur running.\n\n```powershell\npython -m pip install peft\npython -m pip show peft\n```\n\nif `show` is empty, wrong venv/python. classic damn",
        ),
        (
            "What is the difference between base model and adapter?",
            "base model = big original brain.\n\nadapter = small LoRA add-on with your style/task.\n\nfor chat you load base first, then adapter on top. adapter alone is not the full model, dÃ¼d",
        ),
        (
            "How can I stop my fine-tuned model from repeating the prompt?",
            "mask prompt tokens. seriously.\n\nloss should be only on assistant reply. also use chat template, remove examples where assistant mirrors user, lower LR/epochs, and use repetition penalty at inference. but labels are the big one lol",
        ),
        (
            "Write a short JavaScript debounce helper.",
            "ye:\n\n```js\nfunction debounce(fn, delay = 250) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n}\n```\n\nuse for search inputs / resize spam / clicky buttons lol",
        ),
        (
            "What does learning rate do?",
            "learning rate is how hard each training step pushes.\n\ntoo high = weird/forgets stuff. too low = learns nothing. for style LoRA use low-ish, `5e-5` to `1e-4`. personality nudge, not brain damage lol",
        ),
        (
            "what do you think of this idea?",
            "depends on the idea lol, but check 3 things: does it solve a real problem, can you prototype it fast, and would you still care when hype is gone. if yes, build the tiny ugly version first.",
        ),
        (
            "In C#, write a safe async method that fetches JSON from an API and handles failure.",
            "Use one `HttpClient`, pass a cancellation token, and fail with useful info:\n\n```csharp\npublic static async Task<T?> GetJsonAsync<T>(HttpClient client, string url, CancellationToken ct)\n{\n    using var response = await client.GetAsync(url, ct);\n    if (!response.IsSuccessStatusCode)\n    {\n        var body = await response.Content.ReadAsStringAsync(ct);\n        throw new HttpRequestException($\"GET {url} failed: {(int)response.StatusCode} {body}\");\n    }\n\n    return await response.Content.ReadFromJsonAsync<T>(cancellationToken: ct);\n}\n```\n\nDo not create a new `HttpClient` per request unless you enjoy socket pain lol.",
        ),
        (
            "Unity Rigidbody character keeps jittering when I move it. What should I change?",
            "If it has a `Rigidbody`, move it in `FixedUpdate`, not `Update`, and use physics-friendly calls:\n\n```csharp\nvoid FixedUpdate()\n{\n    var input = new Vector3(move.x, 0f, move.y);\n    var next = rb.position + input.normalized * speed * Time.fixedDeltaTime;\n    rb.MovePosition(next);\n}\n```\n\nAlso check interpolation on the Rigidbody, freeze rotations if needed, and avoid setting `transform.position` directly. That fights physics, damn.",
        ),
        (
            "Create a typed React component for a user card in TypeScript.",
            "Keep the props explicit and render from data, not mystery globals:\n\n```tsx\ntype UserCardProps = {\n  name: string;\n  email: string;\n  isOnline?: boolean;\n};\n\nexport function UserCard({ name, email, isOnline = false }: UserCardProps) {\n  return (\n    <article className=\"user-card\">\n      <strong>{name}</strong>\n      <span>{email}</span>\n      <small>{isOnline ? 'Online' : 'Offline'}</small>\n    </article>\n  );\n}\n```\n\nTiny, typed, predictable. ye.",
        ),
        (
            "How do I fetch data in a Next.js App Router server component?",
            "Make the component async and fetch on the server:\n\n```tsx\ntype Product = { id: string; name: string };\n\nasync function getProducts(): Promise<Product[]> {\n  const res = await fetch('https://api.example.com/products', {\n    next: { revalidate: 60 },\n  });\n\n  if (!res.ok) throw new Error('Failed to load products');\n  return res.json();\n}\n\nexport default async function ProductsPage() {\n  const products = await getProducts();\n  return products.map(p => <div key={p.id}>{p.name}</div>);\n}\n```\n\nUse client components only when you need browser state/events.",
        ),
        (
            "Write a TypeScript helper that groups items by key.",
            "Here, generic and boring in the good way:\n\n```ts\nexport function groupBy<T, K extends PropertyKey>(items: T[], getKey: (item: T) => K): Record<K, T[]> {\n  return items.reduce((groups, item) => {\n    const key = getKey(item);\n    (groups[key] ??= []).push(item);\n    return groups;\n  }, {} as Record<K, T[]>);\n}\n```\n\nWorks for strings, numbers, symbols. no chaos needed lol.",
        ),
        (
            "In C++, how do I avoid leaking memory when storing objects in a collection?",
            "Prefer values or smart pointers. If polymorphism is needed, use `std::unique_ptr`:\n\n```cpp\nstd::vector<std::unique_ptr<Entity>> entities;\nentities.push_back(std::make_unique<Player>());\nentities.push_back(std::make_unique<Enemy>());\n```\n\nNo raw owning `new`/`delete`. RAII handles cleanup when the vector dies.",
        ),
        (
            "I committed to the wrong Git branch. What now?",
            "If the commit should move to another branch:\n\n```bash\ngit switch correct-branch\ngit cherry-pick <commit-sha>\ngit switch wrong-branch\ngit reset --hard HEAD~1\n```\n\nOnly do the reset if the wrong branch commit is local or you are sure rewriting is okay. If it was pushed, use `git revert` instead.",
        ),
        (
            "Write a SQL query to find duplicate emails in a users table.",
            "Use `GROUP BY` and `HAVING`:\n\n```sql\nSELECT email, COUNT(*) AS count\nFROM users\nWHERE email IS NOT NULL\nGROUP BY email\nHAVING COUNT(*) > 1\nORDER BY count DESC;\n```\n\nIf emails should be case-insensitive, group by `LOWER(email)`.",
        ),
        (
            "Design a simple REST API endpoint for creating tasks.",
            "Use `POST /tasks` with validation and a clear response:\n\n```http\nPOST /tasks\nContent-Type: application/json\n\n{\n  \"title\": \"Fix login bug\",\n  \"dueDate\": \"2026-06-10\"\n}\n```\n\nReturn `201 Created` with the created task. Validate required fields, reject unknown weird stuff if your API is strict, and keep IDs server-generated.",
        ),
        (
            "How should I debug a production bug without making it worse?",
            "First stabilize: reproduce, check logs/metrics, identify the last change, and make a tiny rollback plan. Then add temporary logging if needed, patch the smallest risky bit, and write a regression test after. Do not freestyle huge refactors during an incident, wtf.",
        ),
    ]

    rows = []
    for index, (prompt, response) in enumerate(examples, start=1):
        system_prompt = XAVIER_SYSTEM_PROMPT if index % 5 == 0 else CODING_SYSTEM_PROMPT
        source = "xavier_coding_style" if index % 5 == 0 else "coding"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        rows.append(
            {
            "path": "built-in/coder_examples",
            "chunk": index,
            "source": source,
            "messages": messages,
            "training_intent": BASE_PRESERVATION_NOTE,
            "text": messages_to_plain_text(messages),
        }
        )

    if not HELPFUL_DATA_FILE:
        return rows

    path = Path(HELPFUL_DATA_FILE)
    if not path.exists():
        print(f"Helpful data file not found, using built-in Xavier examples only: {path}", flush=True)
        return rows

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        normalized_messages = (
            [{"role": "system", "content": XAVIER_SYSTEM_PROMPT}, *messages[1:]]
            if messages[0].get("role") == "system"
            else [{"role": "system", "content": XAVIER_SYSTEM_PROMPT}, *messages]
        )
        rows.append(
            {
                "path": str(path),
                "chunk": line_number,
                "source": "xavier_helpful",
                "messages": normalized_messages,
                "training_intent": BASE_PRESERVATION_NOTE,
                "text": messages_to_plain_text(normalized_messages),
            }
        )

    return rows


def mix_helpful_and_style_examples(style_rows: list[dict]) -> list[dict]:
    helpful_rows = helpful_examples()
    rng = random.Random(DATASET_SEED)
    selected_style = style_rows[:]
    rng.shuffle(selected_style)
    if MAX_STYLE_EXAMPLES > 0:
        selected_style = selected_style[:MAX_STYLE_EXAMPLES]

    target_helpful_count = max(
        len(helpful_rows),
        int(round(len(selected_style) * (1 - STYLE_RATIO) / max(STYLE_RATIO, 0.01))),
    )
    mixed_helpful = [helpful_rows[index % len(helpful_rows)] for index in range(target_helpful_count)]
    rows = [*selected_style, *mixed_helpful]
    rng.shuffle(rows)
    return rows


def split_eval_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(rows) < 8 or EVAL_RATIO <= 0:
        return rows, []

    rng = random.Random(DATASET_SEED)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_count = max(1, int(round(len(shuffled) * EVAL_RATIO)))
    return shuffled[eval_count:], shuffled[:eval_count]


def mix_data_lanes(personality_rows: list[dict], general_rows: list[dict]) -> list[dict]:
    weighted_rows = [*personality_rows, *general_rows]

    rng = random.Random(DATASET_SEED)
    rng.shuffle(weighted_rows)
    return weighted_rows


def validate_dataset_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Dataset row {index} must contain at least user and assistant messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Dataset row {index} must end with an assistant message")
        if not any(message.get("role") == "user" for message in messages[:-1]):
            raise ValueError(f"Dataset row {index} must include a user message before the assistant reply")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"Dataset row {index} has invalid role: {message.get('role')}")
            if not clean_message_text(str(message.get("content", ""))):
                raise ValueError(f"Dataset row {index} has an empty message")


def normalize_training_content(path: Path, content: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json_to_conversation_text(json.loads(content)) or content
        except json.JSONDecodeError:
            return content

    if suffix == ".jsonl":
        lines = []
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                lines.append(json_to_conversation_text(json.loads(line)))
            except json.JSONDecodeError:
                lines.append(line.strip())
        return "\n\n".join(line for line in lines if line)

    return content


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def build_dataset() -> tuple[list[dict], list[dict]]:
    if not UPLOAD_DIR.exists():
        raise FileNotFoundError(f"No upload directory found at {UPLOAD_DIR}")

    personality_rows: list[dict] = []
    general_rows: list[dict] = []
    for path in sorted(UPLOAD_DIR.rglob("*")):
        if not path.is_file():
            continue

        content = read_training_file(path)
        if not content or not content.strip():
            continue

        relative_path = path.relative_to(UPLOAD_DIR).as_posix()
        top_level = relative_path.split("/", 1)[0].lower()
        source_kind = "personality" if top_level == "personality" else "general"
        target_rows = personality_rows if source_kind == "personality" else general_rows
        file_kind = "conversation/text log" if path.suffix.lower() in CONVERSATION_EXTENSIONS else "file"
        reply_examples = build_reply_examples(relative_path, content, path.suffix.lower(), source_kind)
        if reply_examples:
            target_rows.extend(reply_examples)
            if DATA_MODE == "chat":
                continue

        content = normalize_training_content(path, content)
        if not content or not content.strip():
            continue
        chunks = chunk_text(content)
        for index, chunk in enumerate(chunks, start=1):
            prompt = (
                "Uploaded data:\n\n"
                f"Source type: {file_kind}\nPath: {relative_path}\nChunk: {index} of {len(chunks)}"
            )
            messages = with_optional_system(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                {
                    "role": "assistant",
                    "content": chunk,
                },
                HELPFUL_SYSTEM_PROMPT,
            )
            target_rows.append(
                {
                    "path": relative_path,
                    "chunk": index,
                    "source": f"{source_kind}_context",
                    "messages": messages,
                    "text": messages_to_plain_text(messages),
                }
            )

    if not personality_rows and not general_rows:
        raise ValueError("No readable text or conversation files were found in uploads")

    rows = mix_data_lanes(personality_rows, general_rows)
    validate_dataset_rows(rows)
    train_rows, eval_rows = split_eval_rows(rows)

    with DATASET_FILE.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with EVAL_DATASET_FILE.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    personality_count = sum(1 for row in rows if str(row.get("source", "")).startswith("personality_"))
    general_count = sum(1 for row in rows if str(row.get("source", "")).startswith("general_"))
    print(
        f"Built {len(train_rows)} train and {len(eval_rows)} eval examples "
        f"({personality_count} conversation-style, {general_count} context/general; "
        f"data_mode={DATA_MODE}, seed={DATASET_SEED})",
        flush=True,
    )
    return train_rows, eval_rows


def require_training_dependencies():
    try:
        import torch
        from datasets import Dataset
        from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install them with:\n"
            "  python -m pip install -U -r requirements.txt\n\n"
            "QLoRA 4-bit training requires bitsandbytes>=0.46.1. If you already installed requirements, run:\n"
            "  python -m pip install -U \"bitsandbytes>=0.46.1\"\n\n"
            "If installation fails on Python 3.14, create a Python 3.11 or 3.12 virtual environment "
            "and install the requirements there."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "hf_hub_download": hf_hub_download,
        "list_repo_files": list_repo_files,
        "snapshot_download": snapshot_download,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def validate_model_path() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        if ALLOW_MODEL_DOWNLOAD or "/" in MODEL_PATH:
            return
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

    normalized = str(model_path).lower()
    if ".ollama" in normalized or "models\\manifests" in normalized or "models/manifests" in normalized:
        raise ValueError(
            "That path is an Ollama manifest, not a trainable Hugging Face/Transformers model folder.\n\n"
            "Use a folder that contains files like config.json, tokenizer.json, and model .safetensors files.\n"
            "For example, download Qwen/Qwen2.5-Coder-14B-Instruct in Hugging Face format, "
            "then put that folder path in the Local model path box.\n\n"
            "Ollama models are GGUF/runtime models; this trainer cannot fine-tune them directly."
        )

    if not model_path.is_dir():
        raise ValueError(f"Model path must be a folder, got: {MODEL_PATH}")

    present_files = {path.name for path in model_path.iterdir() if path.is_file()}
    missing_required = REQUIRED_MODEL_FILES - present_files
    has_tokenizer = bool(TOKENIZER_FILES & present_files)

    if missing_required or not has_tokenizer:
        raise ValueError(
            "Model folder is missing Hugging Face/Transformers files.\n\n"
            f"Checked: {MODEL_PATH}\n"
            "Expected at least config.json plus tokenizer.json or tokenizer.model.\n"
            "Do not use C:\\Users\\...\\.ollama\\models\\manifests; use the downloaded Hugging Face model folder instead."
        )


def lora_target_modules() -> list[str]:
    custom_targets = os.environ.get("LORA_TARGET_MODULES", "").strip()
    if custom_targets:
        return [item.strip() for item in custom_targets.split(",") if item.strip()]
    return LORA_TARGET_PRESETS.get(LORA_TARGET_MODE, LORA_TARGET_PRESETS["attention"])


def is_complete_model_snapshot(path: str | Path) -> bool:
    snapshot = Path(path)
    if not snapshot.exists():
        return False

    names = {item.name for item in snapshot.iterdir() if item.is_file()}
    has_config = "config.json" in names
    has_tokenizer = bool(TOKENIZER_FILES & names)
    has_weights = any(
        name.endswith((".safetensors", ".bin"))
        and not name.endswith(".index.json")
        for name in names
    )
    return has_config and has_tokenizer and has_weights


def resolve_model_path(deps) -> str:
    model_path = Path(MODEL_PATH)
    if model_path.exists() or "/" not in MODEL_PATH:
        return MODEL_PATH

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    try:
        local_path = deps["snapshot_download"](
            repo_id=MODEL_PATH,
            token=token,
            local_files_only=True,
        )
        if is_complete_model_snapshot(local_path):
            print(f"Using cached model snapshot: {local_path}", flush=True)
            return local_path
        print(f"Ignoring incomplete cached model snapshot: {local_path}", flush=True)
        if not ALLOW_MODEL_DOWNLOAD:
            raise RuntimeError(f"Cached model snapshot is incomplete: {local_path}")
    except Exception:
        if not ALLOW_MODEL_DOWNLOAD:
            raise

    print(
        "Pre-downloading model from Hugging Face file-by-file "
        "(single worker, Xet disabled for Windows reliability)...",
        flush=True,
    )

    files = deps["list_repo_files"](MODEL_PATH, token=token)
    wanted_suffixes = (
        ".json",
        ".model",
        ".txt",
        ".jinja",
        ".safetensors",
        ".bin",
        ".py",
    )
    files = [
        name
        for name in files
        if not name.startswith(".") and name.endswith(wanted_suffixes)
    ]
    if not files:
        raise RuntimeError(f"No downloadable model files found for {MODEL_PATH}")

    local_path = None
    for index, filename in enumerate(files, start=1):
        print(f"Downloading {index}/{len(files)}: {filename}", flush=True)
        downloaded = deps["hf_hub_download"](
            repo_id=MODEL_PATH,
            filename=filename,
            token=token,
            local_files_only=not ALLOW_MODEL_DOWNLOAD,
        )
        if local_path is None:
            local_path = str(Path(downloaded).parent)

    if local_path is None:
        raise RuntimeError(f"Could not resolve downloaded model snapshot for {MODEL_PATH}")

    print(f"Model snapshot ready: {local_path}", flush=True)
    return local_path


def train_model() -> None:
    deps = require_training_dependencies()
    torch = deps["torch"]
    train_rows, eval_rows = build_dataset()
    validate_model_path()
    resolved_model_path = resolve_model_path(deps)
    quick_train = os.environ.get("QUICK_TRAIN") == "1"
    cuda_available = torch.cuda.is_available()
    bf16_available = cuda_available and torch.cuda.is_bf16_supported()

    print(f"Prepared {len(train_rows)} training examples at {DATASET_FILE}", flush=True)
    if eval_rows:
        print(f"Prepared {len(eval_rows)} evaluation examples at {EVAL_DATASET_FILE}", flush=True)
    print(f"Loading base model: {MODEL_PATH}", flush=True)
    if resolved_model_path != MODEL_PATH:
        print(f"Using local snapshot: {resolved_model_path}", flush=True)
    print("Training device: CUDA/GPU" if cuda_available else "Training device: CPU only", flush=True)
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"GPU: {props.name} ({total_gb:.1f} GiB VRAM)", flush=True)
        print(f"Precision: {'BF16' if bf16_available else 'FP16'}", flush=True)
    if quick_train:
        print("Quick CPU test mode enabled: max 5 steps, 256-token context, small LoRA adapter.", flush=True)
    print(
        "Training settings: "
        f"max_length={os.environ.get('MAX_LENGTH', '1536')}, "
        f"batch_size={os.environ.get('BATCH_SIZE', '1')}, "
        f"grad_accum={os.environ.get('GRAD_ACCUM_STEPS', '16')}, "
        f"lora_r={os.environ.get('LORA_R', '16')}, "
        f"lora_targets={','.join(lora_target_modules())}, "
        f"qlora={USE_QLORA}, "
        f"epochs={os.environ.get('EPOCHS', '1')}, "
        f"learning_rate={os.environ.get('LEARNING_RATE', '5e-5')}",
        flush=True,
    )

    try:
        tokenizer = deps["AutoTokenizer"].from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        if "vocab" not in str(exc).lower() or "merges" not in str(exc).lower():
            raise
        print(
            "Fast tokenizer failed on cached vocab/merges files; retrying with use_fast=False...",
            flush=True,
        )
        tokenizer = deps["AutoTokenizer"].from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": not ALLOW_MODEL_DOWNLOAD,
        "device_map": "auto",
    }
    if USE_QLORA and cuda_available:
        model_kwargs["quantization_config"] = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type=os.environ.get("BNB_4BIT_QUANT_TYPE", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_available else torch.float16,
            bnb_4bit_use_double_quant=os.environ.get("BNB_4BIT_USE_DOUBLE_QUANT", "1") == "1",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16_available else torch.float16 if cuda_available else None

    model_kwargs["local_files_only"] = True
    model = deps["AutoModelForCausalLM"].from_pretrained(resolved_model_path, **model_kwargs)
    model.config.use_cache = False
    if USE_QLORA and cuda_available:
        model = deps["prepare_model_for_kbit_training"](model)

    lora_config = deps["LoraConfig"](
        r=int(os.environ.get("LORA_R", "16")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", "16")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(),
    )
    model = deps["get_peft_model"](model, lora_config)
    if cuda_available:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    dataset = deps["Dataset"].from_list(train_rows)
    eval_dataset = deps["Dataset"].from_list(eval_rows) if eval_rows else None
    max_length = int(os.environ.get("MAX_LENGTH", "1536"))

    def tokenize(batch):
        input_ids = []
        attention_mask = []
        labels = []

        for messages in batch["messages"]:
            prompt_messages = messages[:-1]
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            ids = tokenizer(full, add_special_tokens=False).input_ids
            label_ids = [-100] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids) :]

            if len(ids) > max_length:
                overflow = len(ids) - max_length
                ids = ids[overflow:]
                label_ids = label_ids[overflow:]
                if all(label == -100 for label in label_ids):
                    label_ids[-1] = ids[-1]

            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            labels.append(label_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized_eval = (
        eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names) if eval_dataset is not None else None
    )

    def collate_batch(features):
        pad_id = tokenizer.pad_token_id
        max_size = max(len(feature["input_ids"]) for feature in features)
        if cuda_available and max_size % 8:
            max_size += 8 - (max_size % 8)

        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for feature in features:
            pad_size = max_size - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_size)
            batch_attention.append(feature["attention_mask"] + [0] * pad_size)
            batch_labels.append(feature["labels"] + [-100] * pad_size)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    training_args = deps["TrainingArguments"](
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM_STEPS", "16")),
        num_train_epochs=float(os.environ.get("EPOCHS", "1")),
        max_steps=int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else -1,
        learning_rate=float(os.environ.get("LEARNING_RATE", "5e-5")),
        bf16=bf16_available,
        fp16=cuda_available and not bf16_available,
        gradient_checkpointing=cuda_available,
        optim=os.environ.get("OPTIM", "paged_adamw_8bit" if USE_QLORA and cuda_available else "adamw_torch"),
        logging_steps=1,
        eval_strategy="epoch" if tokenized_eval is not None else "no",
        save_strategy="epoch",
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized_eval,
        data_collator=collate_batch,
    )

    print("Training started...", flush=True)
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Training completed. Conversation adapter saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

