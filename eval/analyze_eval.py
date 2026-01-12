#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze JSONL eval outputs produced by eval/run_eval.py.

Input JSONL format (per line):
{
  "run_id": "...",
  "checkpoint": "...",
  "prompt_id": "...",
  "category": "...",
  "lang": "en",
  "system": "...",
  "prompt": "...",
  "output": "...",
  "error": "",
  "gen_cfg": {...}
}

Outputs:
- summary.csv / summary.md: run-level stats
- prompt_level.csv: prompt-level stats
- (optional) aligned_outputs.csv: side-by-side outputs across runs for case study
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# -------------------------
# Heuristics (simple, stable)
# -------------------------
REFUSAL_PATTERNS = [
    r"\bI(?:'| a)m sorry\b",
    r"\bI can(?:not|'t)\b",
    r"\bI (?:won't|will not)\b",
    r"\bI (?:cannot|can't) help with\b",
    r"\bI (?:am unable|cannot) comply\b",
    r"\bI do not have (?:access|the ability)\b",
    r"\bAs an AI\b",
    r"\bI(?:'| a)m just an AI\b",
    r"\bI (?:cannot|can't) provide\b",
    r"\bI (?:don't|do not) (?:have|know)\b",
    r"\bI (?:cannot|can't) assist with that\b",
]

# "Overly boilerplate email" detector (useful given your toy prompt example)
EMAIL_BOILERPLATE_PATTERNS = [
    r"\bDear\b",
    r"\bBest regards\b",
    r"\bSincerely\b",
    r"\bSubject:\b",
]

CODE_BLOCK_PATTERN = re.compile(r"```", re.MULTILINE)
URL_PATTERN = re.compile(r"https?://\S+")
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")  # rough "non-English chars" heuristic


def safe_get(d: dict, k: str, default=None):
    v = d.get(k, default)
    return default if v is None else v


def detect_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 5:
        return False
    low = t.lower()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    # Extra: if output starts with refusal-ish phrase
    if low.startswith("i'm sorry") or low.startswith("i am sorry") or low.startswith("sorry"):
        return True
    return False


def detect_email_boilerplate(text: str) -> bool:
    if not text:
        return False
    hits = 0
    for pat in EMAIL_BOILERPLATE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits += 1
    return hits >= 2


def approx_english_ratio(text: str) -> float:
    """
    crude: ratio of ASCII chars among all non-space chars
    """
    if not text:
        return 0.0
    s = "".join(ch for ch in text if not ch.isspace())
    if not s:
        return 0.0
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / len(s)


def word_count_approx(text: str) -> int:
    if not text:
        return 0
    # split by whitespace, keep simple
    return len(text.strip().split())


def char_count(text: str) -> int:
    return 0 if not text else len(text)


def has_code_block(text: str) -> bool:
    if not text:
        return False
    return bool(CODE_BLOCK_PATTERN.search(text))


def has_url(text: str) -> bool:
    if not text:
        return False
    return bool(URL_PATTERN.search(text))


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_run_key(path: Path, sample_row: dict) -> Tuple[str, str]:
    """
    Determine (run_id, checkpoint) for grouping.
    Prefer explicit fields; fallback to filename pattern: {run_id}__{checkpoint}.jsonl
    """
    run_id = safe_get(sample_row, "run_id", "")
    ckpt = safe_get(sample_row, "checkpoint", "")

    if run_id and ckpt:
        return run_id, ckpt

    name = path.name
    # e.g. 20260106_145132__checkpoint-62.jsonl
    m = re.match(r"(.+?)__([^\.]+)\.jsonl$", name)
    if m:
        run_id = run_id or m.group(1)
        ckpt = ckpt or m.group(2)
    return run_id or "unknown_run", ckpt or "unknown_ckpt"


def build_frames(files: List[Path]) -> pd.DataFrame:
    all_rows = []
    for fp in files:
        rows = load_jsonl(fp)
        if not rows:
            continue
        run_id, ckpt = infer_run_key(fp, rows[0])
        for r in rows:
            out = safe_get(r, "output", "")
            err = safe_get(r, "error", "")
            prompt = safe_get(r, "prompt", "")
            system = safe_get(r, "system", "")
            prompt_id = safe_get(r, "prompt_id", "")
            category = safe_get(r, "category", "")
            lang = safe_get(r, "lang", "")

            rec = {
                "file": str(fp),
                "run_id": run_id,
                "checkpoint": ckpt,
                "prompt_id": prompt_id,
                "category": category,
                "lang": lang,
                "system": system,
                "prompt": prompt,
                "output": out,
                "error": err,
                "output_chars": char_count(out),
                "output_words": word_count_approx(out),
                "is_empty": int(len(out.strip()) == 0),
                "has_error": int(bool(err)),
                "is_refusal": int(detect_refusal(out)),
                "email_boilerplate": int(detect_email_boilerplate(out)),
                "english_ratio": approx_english_ratio(out),
                "has_codeblock": int(has_code_block(out)),
                "has_url": int(has_url(out)),
            }
            all_rows.append(rec)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def summarize_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    gcols = ["run_id", "checkpoint"]
    agg = df.groupby(gcols).agg(
        n=("prompt_id", "count"),
        empty_rate=("is_empty", "mean"),
        error_rate=("has_error", "mean"),
        refusal_rate=("is_refusal", "mean"),
        email_boilerplate_rate=("email_boilerplate", "mean"),
        avg_chars=("output_chars", "mean"),
        med_chars=("output_chars", "median"),
        avg_words=("output_words", "mean"),
        med_words=("output_words", "median"),
        avg_english_ratio=("english_ratio", "mean"),
        codeblock_rate=("has_codeblock", "mean"),
        url_rate=("has_url", "mean"),
    ).reset_index()

    # nice formatting (keep numeric raw in csv, format later in md)
    return agg.sort_values(["run_id", "checkpoint"])


def to_markdown_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No data."

    fmt = summary.copy()
    pct_cols = [
        "empty_rate", "error_rate", "refusal_rate", "email_boilerplate_rate", "codeblock_rate", "url_rate"
    ]
    for c in pct_cols:
        if c in fmt.columns:
            fmt[c] = (fmt[c] * 100.0).map(lambda x: f"{x:.1f}%")
    if "avg_english_ratio" in fmt.columns:
        fmt["avg_english_ratio"] = (fmt["avg_english_ratio"] * 100.0).map(lambda x: f"{x:.1f}%")

    # round numeric stats
    for c in ["avg_chars", "med_chars", "avg_words", "med_words"]:
        if c in fmt.columns:
            fmt[c] = fmt[c].map(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and not math.isnan(x) else str(x))

    # select columns (order)
    cols = [
        "run_id", "checkpoint", "n",
        "avg_words", "med_words", "avg_chars", "med_chars",
        "refusal_rate", "empty_rate", "error_rate",
        "avg_english_ratio", "email_boilerplate_rate",
        "codeblock_rate", "url_rate",
    ]
    cols = [c for c in cols if c in fmt.columns]
    return fmt[cols].to_markdown(index=False)


def build_aligned_outputs(df: pd.DataFrame, run_keys: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Produce a side-by-side table: one row per prompt_id, with outputs per run.
    """
    if df.empty:
        return pd.DataFrame()

    # Keep stable prompt meta
    base_cols = ["prompt_id", "category", "lang", "system", "prompt"]
    base = df[base_cols].drop_duplicates(subset=["prompt_id"]).set_index("prompt_id")

    out = base.copy()
    for (rid, ckpt) in run_keys:
        sub = df[(df["run_id"] == rid) & (df["checkpoint"] == ckpt)].copy()
        sub = sub.set_index("prompt_id")
        col_name = f"{rid}__{ckpt}"
        out[col_name] = sub["output"]
        out[col_name + "__error"] = sub["error"]

    out = out.reset_index()
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_dir",
        type=str,
        default="/hy-tmp/align-lab/outputs/eval",
        help="Directory containing eval JSONL files.",
    )
    ap.add_argument(
        "--files",
        type=str,
        default="",
        help="Comma-separated list of JSONL files. If empty, will scan eval_dir for *.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/hy-tmp/align-lab/outputs/eval/analysis",
        help="Output directory for summary tables.",
    )
    ap.add_argument(
        "--aligned",
        action="store_true",
        help="Also export aligned_outputs.csv (side-by-side outputs across runs).",
    )
    ap.add_argument(
        "--min_prompts",
        type=int,
        default=100,
        help="Warn if any run has fewer than this number of prompts.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.files.strip():
        files = [Path(p.strip()) for p in args.files.split(",") if p.strip()]
    else:
        files = sorted(eval_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No JSONL files found. eval_dir={eval_dir}")

    df = build_frames(files)
    if df.empty:
        raise RuntimeError("Loaded 0 records from jsonl files.")

    # Prompt-level export (single table for debugging & deeper analysis)
    prompt_csv = out_dir / "prompt_level.csv"
    df.to_csv(prompt_csv, index=False)
    print(f"[OK] wrote: {prompt_csv}")

    # Run-level summary
    summary = summarize_run(df)
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] wrote: {summary_csv}")

    summary_md = out_dir / "summary.md"
    md = to_markdown_table(summary)
    summary_md.write_text(md + "\n", encoding="utf-8")
    print(f"[OK] wrote: {summary_md}")

    # basic warnings
    for _, row in summary.iterrows():
        if int(row["n"]) < args.min_prompts:
            print(f"[WARN] run {row['run_id']} {row['checkpoint']} has only n={row['n']} prompts")

    # Optional aligned outputs
    if args.aligned:
        run_keys = [(r["run_id"], r["checkpoint"]) for _, r in summary.iterrows()]
        aligned = build_aligned_outputs(df, run_keys)
        aligned_csv = out_dir / "aligned_outputs.csv"
        aligned.to_csv(aligned_csv, index=False)
        print(f"[OK] wrote: {aligned_csv}")

    print("\n[DONE] Analysis complete.")
    print(f"Open: {summary_md}")


if __name__ == "__main__":
    main()