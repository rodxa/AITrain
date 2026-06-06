from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the mixed Qwen ChatML dataset without training.")
    parser.add_argument("--output", default=str(train_model.DATASET_FILE), help="Training JSONL output path.")
    parser.add_argument("--eval-output", default=str(train_model.EVAL_DATASET_FILE), help="Evaluation JSONL output path.")
    parser.add_argument("--preview", type=int, default=3, help="Number of training rows to print.")
    args = parser.parse_args()

    train_rows, eval_rows = train_model.build_dataset()
    train_model.validate_dataset_rows(train_rows)
    if eval_rows:
        train_model.validate_dataset_rows(eval_rows)

    output = Path(args.output)
    eval_output = Path(args.eval_output)
    if output != train_model.DATASET_FILE:
        with output.open("w", encoding="utf-8") as handle:
            for row in train_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if eval_output != train_model.EVAL_DATASET_FILE:
        with eval_output.open("w", encoding="utf-8") as handle:
            for row in eval_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Valid dataset: {len(train_rows)} train rows, {len(eval_rows)} eval rows")
    print(
        "Additive defaults: low LR, attention-only LoRA, and adapter_scale inference keep the base model brain intact."
    )
    for row in train_rows[: max(0, args.preview)]:
        print("\n--- Preview ---")
        print(json.dumps({"source": row.get("source"), "messages": row.get("messages")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
