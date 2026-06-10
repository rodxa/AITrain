from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER = BASE_DIR / "conversation-ai-lora"
DEFAULT_TEST_PROMPTS = [
    "How do I fix this TypeScript error?",
    "Write a Unity movement script.",
    "Review this SQL query for duplicates.",
    "Explain LoRA simply.",
    "Why is my fine-tuned AI repeating the prompt?",
]
SYSTEM_PROMPT = (
    "You answer helpfully, but your tone is casual, slightly chaotic, short, "
    "and WhatsApp-like. Use words like lol, lel, dud, wtf, ok, damn, nah, ye sometimes. "
    "Do not sound formal. Still answer the question properly."
)
DEFAULT_ADAPTER_SCALE = 0.75
SYSTEM_PROMPT = (
    os.environ.get(
        "AGENT_INSTRUCTIONS",
        "Code professionally first. Be concise, practical, and direct. Preserve my casual writing style where it feels natural, "
        "but do not sacrifice correctness. Prioritize coding ability over personality.",
    )
)


def load_base_model_name(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"No base_model_name_or_path found in {config_path}")
    return base_model


def adapter_scaling_items(model):
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter_name, value in list(scaling.items()):
                yield module, adapter_name, value


def set_adapter_scale(model, scale: float) -> None:
    scale = max(0.0, min(scale, 2.0))
    for module, adapter_name, value in adapter_scaling_items(model):
        base_scaling = getattr(module, "_adapter_base_scaling", None)
        if base_scaling is None:
            base_scaling = {}
            setattr(module, "_adapter_base_scaling", base_scaling)
        if adapter_name not in base_scaling:
            base_scaling[adapter_name] = value
        module.scaling[adapter_name] = base_scaling[adapter_name] * scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the trained conversation/text LoRA adapter.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "Reply in the style learned from my uploaded conversations and texts."
        ),
    )
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="Path to the LoRA adapter folder.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base-model", default="", help="Override the base model name/path.")
    parser.add_argument("--base-only", action="store_true", help="Test the base model without the LoRA adapter.")
    parser.add_argument("--adapter-scale", type=float, default=DEFAULT_ADAPTER_SCALE, help="LoRA strength, 0.0-2.0.")
    parser.add_argument("--smoke", action="store_true", help="Run the standard helpful/style test prompts.")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter).resolve()
    base_model = args.base_model or load_base_model_name(adapter_dir)
    cuda_available = torch.cuda.is_available()
    bf16_available = cuda_available and torch.cuda.is_bf16_supported()

    print(f"Loading base model: {base_model}", flush=True)
    if args.base_only:
        print("Adapter: disabled", flush=True)
    else:
        print(f"Loading adapter: {adapter_dir}", flush=True)
    print("Device: CUDA/GPU" if cuda_available else "Device: CPU", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception as exc:
        if "vocab" not in str(exc).lower() or "merges" not in str(exc).lower():
            raise
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16_available else torch.float16 if cuda_available else None,
        device_map="auto",
    )
    if not args.base_only:
        model = PeftModel.from_pretrained(model, adapter_dir)
        set_adapter_scale(model, args.adapter_scale)
    model.eval()

    blocked_tokens = [
        token
        for token in ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"]
        if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id
    ]
    bad_words_ids = [tokenizer(token, add_special_tokens=False).input_ids for token in blocked_tokens]

    def generate_reply(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generation_args = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "repetition_penalty": 1.18,
            "no_repeat_ngram_size": 4,
            "bad_words_ids": bad_words_ids or None,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_args["temperature"] = args.temperature
            generation_args["top_p"] = 0.9

        with torch.no_grad():
            generated = model.generate(**generation_args)

        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    prompts = DEFAULT_TEST_PROMPTS if args.smoke else [args.prompt]
    for prompt in prompts:
        print(f"\n--- Prompt ---\n{prompt}")
        print("\n--- Response ---\n")
        print(generate_reply(prompt))


if __name__ == "__main__":
    main()
