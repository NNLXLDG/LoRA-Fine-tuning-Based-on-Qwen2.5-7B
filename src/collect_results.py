import os, re, csv, json, glob
from pathlib import Path

RUNS_DIR = Path("/hy-tmp/align-lab/outputs/runs")
OUT_CSV  = Path("/hy-tmp/align-lab/outputs/results.csv")

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")

def parse_hyperparams(log_txt: str) -> dict:
    # logs里有一段：[INFO] Hyperparams: { ...json... }
    m = re.search(r"\[INFO\]\s+Hyperparams:\s*\n(\{.*?\})\s*\n", log_txt, flags=re.S)
    if not m:
        return {}
    block = m.group(1)
    # block 是 json 风格（双引号），直接 json.loads
    try:
        return json.loads(block)
    except Exception:
        return {}

def parse_last_metrics(log_txt: str) -> dict:
    # trainer每隔一段会打印 {'loss':..., 'train_runtime':...}
    # 取最后一次出现的 train_runtime/train_loss
    metrics = {}
    # 最后一条 train_loss
    m_loss = list(re.finditer(r"'train_loss':\s*([0-9.]+)", log_txt))
    if m_loss:
        metrics["train_loss"] = float(m_loss[-1].group(1))
    # 最后一条 train_runtime
    m_rt = list(re.finditer(r"'train_runtime':\s*([0-9.]+)", log_txt))
    if m_rt:
        metrics["train_runtime_s"] = float(m_rt[-1].group(1))
    # samples/sec
    m_sps = list(re.finditer(r"'train_samples_per_second':\s*([0-9.]+)", log_txt))
    if m_sps:
        metrics["samples_per_s"] = float(m_sps[-1].group(1))
    # steps/sec
    m_stps = list(re.finditer(r"'train_steps_per_second':\s*([0-9.]+)", log_txt))
    if m_stps:
        metrics["steps_per_s"] = float(m_stps[-1].group(1))
    return metrics

def main():
    rows = []
    for run_dir in sorted(RUNS_DIR.glob("2026*_*")):
        log_path = run_dir / "train.log"
        if not log_path.exists():
            continue
        txt = read_text(log_path)
        hp = parse_hyperparams(txt)
        mt = parse_last_metrics(txt)

        rows.append({
            "run_id": run_dir.name,
            "model": hp.get("model", ""),
            "data_root": hp.get("data_root", ""),
            "limit": hp.get("limit", ""),
            "max_seq_len": hp.get("max_seq_len", ""),
            "bsz": hp.get("bsz", ""),
            "grad_acc": hp.get("grad_acc", ""),
            "lr": hp.get("lr", ""),
            "epochs": hp.get("epochs", ""),
            "lora_r": hp.get("lora_r", ""),
            "lora_alpha": hp.get("lora_alpha", ""),
            "lora_dropout": hp.get("lora_dropout", ""),
            "bf16": hp.get("bf16", ""),
            "grad_ckpt": hp.get("grad_ckpt", ""),
            **mt,
            "log_path": str(log_path),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote: {OUT_CSV}")
    for r in rows:
        print(f"- {r['run_id']}  lora_r={r.get('lora_r')}  train_loss={r.get('train_loss')}  runtime_s={r.get('train_runtime_s')}")

if __name__ == "__main__":
    main()