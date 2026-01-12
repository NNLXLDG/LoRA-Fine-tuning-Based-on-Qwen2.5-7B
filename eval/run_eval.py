#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception as e:
    raise RuntimeError("peft is required. Install via: pip install peft") from e


def parse_args():
    p = argparse.ArgumentParser(description="Run eval on LoRA adapters with fixed prompts (jsonl).")

    # IO / paths
    p.add_argument("--base_model", type=str, required=True, help="Path or HF id of base model.")
    p.add_argument("--runs_dir", type=str, required=True, help="Directory containing run subfolders.")
    p.add_argument(
        "--run_ids",
        type=str,
        default="",
        help="Comma-separated run IDs (subfolder names). If empty, auto-detect all subdirs.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint-62",
        help='Which checkpoint folder under each run dir to load (e.g., "checkpoint-62").',
    )
    p.add_argument("--prompts", type=str, required=True, help="Prompts file in jsonl format.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for eval jsonl files.")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling. Default: False (greedy).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)

    # perf / device
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto", help='Transformers device_map, e.g. "auto".')
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (default 1).")

    # misc
    p.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True.")
    p.add_argument("--use_fast", action="store_true", help="Use fast tokenizer if available.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N prompts.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling (if do_sample).")

    return p.parse_args()


def torch_dtype_from_str(s: str):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32


def load_prompts_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    assert path.exists(), f"Prompt file not found: {path}"
    prompts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
            if limit > 0 and len(prompts) >= limit:
                break
    return prompts


def list_run_ids(runs_dir: Path, run_ids_csv: str) -> List[str]:
    if run_ids_csv.strip():
        return [x.strip() for x in run_ids_csv.split(",") if x.strip()]

    # auto-detect: only immediate subdirs
    ids = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    return ids


def find_adapter_dir(run_dir: Path, checkpoint: str) -> Path:
    """
    Force use a specific checkpoint folder, e.g. run_dir/checkpoint-62
    """
    d = run_dir / checkpoint
    if (d / "adapter_config.json").exists():
        return d
    raise FileNotFoundError(f"Cannot find adapter_config.json in {d}")


def build_inputs(tokenizer, batch_system: List[str], batch_prompt: List[str]):
    messages_batch = []
    for sys, up in zip(batch_system, batch_prompt):
        messages_batch.append(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": up},
            ]
        )

    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs


@torch.no_grad()
def generate_batch(
    tokenizer,
    model,
    batch_system: List[str],
    batch_prompt: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
):
    inputs = build_inputs(tokenizer, batch_system, batch_prompt)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Only pass sampling args if sampling is enabled (avoid confusion)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p, top_k=top_k))

    out = model.generate(**inputs, **gen_kwargs)

    # IMPORTANT: decode only generated part (exclude prompt tokens)
    input_lens = (inputs["attention_mask"].sum(dim=1)).tolist()
    outputs = []
    for seq, in_len in zip(out, input_lens):
        gen_ids = seq[in_len:]
        outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return outputs


def main():
    args = parse_args()

    # seed (only matters if do_sample)
    if args.seed and args.do_sample:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    runs_dir = Path(args.runs_dir)
    prompts_path = Path(args.prompts)
    out_dir = Path(args.out_dir)

    assert runs_dir.exists(), f"runs_dir not found: {runs_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts_jsonl(prompts_path, limit=args.limit)
    print(f"[OK] Loaded prompts: {len(prompts)} from {prompts_path}")

    run_ids = list_run_ids(runs_dir, args.run_ids)
    if not run_ids:
        raise RuntimeError("No run_ids found. Provide --run_ids or check --runs_dir.")
    print(f"[OK] Will evaluate {len(run_ids)} run(s): {run_ids}")

    # tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch_dtype_from_str(args.dtype)

    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            print(f"[SKIP] run dir not found: {run_dir}")
            continue

        try:
            adapter_dir = find_adapter_dir(run_dir, args.checkpoint)
        except Exception as e:
            print(f"[SKIP] {run_id}: {e}")
            continue

        print(f"\n== Evaluating {run_id} ==")
        print(f"  run_dir     = {run_dir}")
        print(f"  adapter_dir = {adapter_dir}")
        print(f"  checkpoint  = {args.checkpoint}")

        # fresh base model each run
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()

        # load LoRA
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        out_path = out_dir / f"{run_id}__{args.checkpoint}.jsonl"
        t0 = time.time()

        bs = max(1, args.batch_size)
        n = len(prompts)

        with out_path.open("w", encoding="utf-8") as wf:
            i = 0
            while i < n:
                batch = prompts[i : i + bs]
                batch_system = [rec.get("system", "You are a helpful assistant.") for rec in batch]
                batch_prompt = [rec["prompt"] for rec in batch]

                try:
                    outs = generate_batch(
                        tokenizer=tokenizer,
                        model=model,
                        batch_system=batch_system,
                        batch_prompt=batch_prompt,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    batch_errs = [""] * len(batch)
                except RuntimeError as e:
                    msg = str(e)
                    # if OOM, record and continue with empty outputs
                    batch_errs = [f"{type(e).__name__}: {msg}"] * len(batch)
                    outs = [""] * len(batch)
                    if "out of memory" in msg.lower():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                for rec, out_text, err in zip(batch, outs, batch_errs):
                    prompt_idx = i + 1
                    out_rec = {
                        "run_id": run_id,
                        "checkpoint": args.checkpoint,
                        "prompt_id": rec.get("id", f"idx_{prompt_idx:04d}"),
                        "category": rec.get("category", ""),
                        "lang": rec.get("lang", "en"),
                        "system": rec.get("system", "You are a helpful assistant."),
                        "prompt": rec["prompt"],
                        "output": out_text,
                        "error": err,
                        "gen_cfg": {
                            "max_new_tokens": args.max_new_tokens,
                            "do_sample": args.do_sample,
                            "temperature": args.temperature if args.do_sample else None,
                            "top_p": args.top_p if args.do_sample else None,
                            "top_k": args.top_k if args.do_sample else None,
                            "batch_size": bs,
                        },
                        "base_model": args.base_model,
                        "adapter_dir": str(adapter_dir),
                    }
                    wf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

                i += bs
                if (i % max(1, 15 * bs) == 0) or (i >= n):
                    print(f"  {min(i, n)}/{n} done")

        dt = time.time() - t0
        print(f"[DONE] wrote: {out_path}  (time: {dt:.1f}s)")

        # free VRAM between runs
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()