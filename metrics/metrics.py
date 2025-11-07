"""
metrics.py

Evaluate the fine-tuned model in `models/flan_t5_fee_extractor` against
the dataset at `data/fee_messages_instruction.json`.

Features:
- load model + tokenizer (fallback to hub if not present locally)
- batched generation with configurable beams/batch size
- metrics: exact match, normalized exact match, ROUGE-L (token-level LCS),
  per-field precision/recall/F1 for the structured "key: value | ..." targets
- latency (avg / p50 / p95) and throughput
- save predictions to JSON

Usage:
    python metrics.py --model_dir ./models/flan_t5_fee_extractor --data_path data/fee_messages_instruction.json

Requires: transformers, datasets, torch, tqdm, numpy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_json_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    # Expect each entry to contain at least `input_text` and `target_text`.
    inputs = [d.get("input_text", d.get("input", "")) for d in js]
    targets = [d.get("target_text", d.get("target", "")) for d in js]
    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def normalize_text(s: str) -> str:
    """Lowercase, strip, collapse whitespace for robust comparison."""
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())


def tokenize_kv_text(s: str) -> Dict[str, str]:
    """Parse a target/prediction of form 'k: v | k2: v2' into a dict.

    If parsing fails, returns an empty dict. Values are normalized.
    """
    d: Dict[str, str] = {}
    if not s:
        return d
    # Split on '|' separators
    parts = [p.strip() for p in s.split("|") if p.strip()]
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            d[k.strip().lower()] = normalize_text(v)
    return d


def lcs_length(a: List[str], b: List[str]) -> int:
    # Classic DP O(n*m) LCS length; tokens lists are usually short here
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def rouge_l_score(pred: str, ref: str) -> Tuple[float, float, float]:
    # Tokenize by whitespace
    p_toks = pred.split()
    r_toks = ref.split()
    lcs = lcs_length(p_toks, r_toks)
    if lcs == 0:
        return 0.0, 0.0, 0.0
    prec = lcs / max(1, len(p_toks))
    rec = lcs / max(1, len(r_toks))
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    return prec, rec, f1


def compute_field_metrics(pred_dicts: List[Dict[str, str]], ref_dicts: List[Dict[str, str]]):
    # Collect counts per field
    all_fields = set()
    for rd in ref_dicts:
        all_fields.update(rd.keys())
    for pd in pred_dicts:
        all_fields.update(pd.keys())

    counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in all_fields}
    for pd, rd in zip(pred_dicts, ref_dicts):
        for f in all_fields:
            pval = pd.get(f)
            rval = rd.get(f)
            if rval is None or rval == "":
                # if reference doesn't have the field but prediction does, count as FP
                if pval is not None and pval != "":
                    counts[f]["fp"] += 1
                # else both absent -> ignored
            else:
                # reference has value
                if pval is None or pval == "":
                    counts[f]["fn"] += 1
                else:
                    # compare normalized strings
                    if pval == rval:
                        counts[f]["tp"] += 1
                    else:
                        counts[f]["fp"] += 1
                        counts[f]["fn"] += 1

    metrics = {}
    for f, c in counts.items():
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
        metrics[f] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return metrics


def evaluate_predictions(preds: List[str], refs: List[str]) -> Dict:
    n = len(preds)
    assert n == len(refs)

    exact = 0
    norm_exact = 0
    rouge_f1s = []
    rouge_precs = []
    rouge_recs = []
    pred_dicts = []
    ref_dicts = []

    for p, r in zip(preds, refs):
        if p == r:
            exact += 1
        if normalize_text(p) == normalize_text(r):
            norm_exact += 1
        pnorm = normalize_text(p)
        rnorm = normalize_text(r)
        prec, rec, f1 = rouge_l_score(pnorm, rnorm)
        rouge_precs.append(prec)
        rouge_recs.append(rec)
        rouge_f1s.append(f1)

        pred_dicts.append(tokenize_kv_text(pnorm))
        ref_dicts.append(tokenize_kv_text(rnorm))

    field_metrics = compute_field_metrics(pred_dicts, ref_dicts)

    results = {
        "n": n,
        "exact_match": exact / n,
        "normalized_exact_match": norm_exact / n,
        "rouge_l_prec_mean": float(np.mean(rouge_precs)) if rouge_precs else 0.0,
        "rouge_l_rec_mean": float(np.mean(rouge_recs)) if rouge_recs else 0.0,
        "rouge_l_f1_mean": float(np.mean(rouge_f1s)) if rouge_f1s else 0.0,
        "field_metrics": field_metrics,
    }
    return results


def generate_batches(model, tokenizer, inputs: List[str], device: torch.device, batch_size=8, **gen_kwargs):
    preds: List[str] = []
    latencies: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch = inputs[i : i + batch_size]
            tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            t0 = time.perf_counter()
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / input_ids.size(0))
            for o in out:
                preds.append(tokenizer.decode(o, skip_special_tokens=True))
    return preds, latencies


def summarize_latency(latencies: List[float]):
    arr = np.array(latencies)
    return {"mean_s": float(arr.mean()), "p50_s": float(np.percentile(arr, 50)), "p95_s": float(np.percentile(arr, 95)), "throughput_items_per_s": float(1.0 / arr.mean() if arr.mean() > 0 else math.inf)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models/flan_t5_fee_extractor")
    parser.add_argument("--data_path", type=str, default="data/fee_messages_eval.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_preds", type=str, default="metrics/metrics_summary.json", help="Path to save predictions JSON")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading dataset from {args.data_path}...")
    ds = load_json_dataset(args.data_path)
    # `datasets.Dataset` indexing returns a Column object which is not
    # JSON-serializable. Convert to plain Python lists before use.
    inputs = list(ds["input_text"])
    refs = list(ds["target_text"])

    print(f"Loading tokenizer/model from {args.model_dir} (or hub fallback)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

    print("Running generation...")
    preds, latencies = generate_batches(
        model,
        tokenizer,
        inputs,
        device,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_length=args.max_length,
        early_stopping=True,
    )

    print("Computing metrics...")
    results = evaluate_predictions(preds, refs)
    latency_stats = summarize_latency(latencies) if latencies else {}
    results["latency"] = latency_stats

    # Basic nice printing
    print("\n=== Summary ===")
    print(f"Samples: {results['n']}")
    print(f"Exact match: {results['exact_match']:.4f}")
    print(f"Normalized exact match: {results['normalized_exact_match']:.4f}")
    print(f"ROUGE-L (prec/rec/f1): {results['rouge_l_prec_mean']:.4f} / {results['rouge_l_rec_mean']:.4f} / {results['rouge_l_f1_mean']:.4f}")
    if latency_stats:
        print(f"Latency mean/p50/p95 (s): {latency_stats['mean_s']:.4f} / {latency_stats['p50_s']:.4f} / {latency_stats['p95_s']:.4f}")
        print(f"Throughput items/s: {latency_stats['throughput_items_per_s']:.2f}")

    print("\nPer-field metrics (top fields):")
    # Show a few top fields sorted by f1
    fm = results["field_metrics"]
    sorted_fields = sorted(fm.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    for k, v in sorted_fields[:15]:
        print(f" - {k}: prec={v['precision']:.3f} rec={v['recall']:.3f} f1={v['f1']:.3f} (tp={v['tp']} fp={v['fp']} fn={v['fn']})")

    # Save predictions and results optionally
    if args.save_preds:
        out = {"predictions": preds, "references": refs, "metrics": results}
        with open(args.save_preds, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved predictions+metrics to {args.save_preds}")


if __name__ == "__main__":
    main()
