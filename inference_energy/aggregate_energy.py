#!/usr/bin/env python3
"""Collect the per-inference power logs into data/inference_energy.csv.

Each log is energy_<domain>_<model>_st<samples>.csv as written by
run_inference_energy.sh. The total energy, mean power and elapsed time of an
inference are the accumulated values of its last sample; the peak power is
the maximum instantaneous value.

Usage: python3 inference_energy/aggregate_energy.py <energy_logs dir> [out.csv]
"""
import csv, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["deepseek-r1:70b", "gemma3:12b", "gemma3:27b", "gpt-oss:120b", "gpt-oss:20b",
          "llama3.3", "llama4", "mistral-nemo", "nemotron:70b", "qwen3:235b", "qwen3:32b"]
logs = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "inference_energy.csv")

rows = []
for path in sorted(glob.glob(os.path.join(logs, "energy_*_st*.csv"))):
    body, samples = re.match(r"energy_(.+)_st(\d+)\.csv", os.path.basename(path)).groups()
    model = next(m for m in MODELS if body.endswith("_" + m.replace(":", "_")))
    domain = body[:-len(model) - 1]
    data = [l.split() for l in open(path) if l.strip() and l[0].isdigit()]
    if not data:
        continue
    last = data[-1]
    rows.append([domain, model, int(samples), last[2], last[3], last[5],
                 "%.2f" % max(float(d[1]) for d in data)])

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Domain", "Model", "Stage", "Total Energy (J)", "Avg Power (W)", "Time (s)", "Peak Power (W)"])
    w.writerows(rows)
print("%d inferences written to %s" % (len(rows), out))
