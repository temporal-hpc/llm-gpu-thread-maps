#!/usr/bin/env python3
"""Collect the graded JSON files into the CSV files of data/.

  passk_responses.csv  one row per response (1,188): verdict before and after
                       the repair, index of the first mismatch, repair log
  passk_cells.csv      one row per (model, domain, samples) cell (198): number
                       of correct samples out of 5, greedy pass, pass@k
  coverage.csv         one row per response: Domain Coverage over 10^5 indices

Usage: python3 evaluation/aggregate.py [--graded graded] [--out data]
"""
import argparse, csv, glob, json, os
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--graded", default=os.path.join(ROOT, "graded"))
ap.add_argument("--out", default=os.path.join(ROOT, "data"))
a = ap.parse_args()

MODELS = ["gpt-oss:120b", "gpt-oss:20b", "qwen3:235b", "qwen3:32b", "deepseek-r1:70b",
          "llama3.3", "llama4", "gemma3:27b", "gemma3:12b", "mistral-nemo", "nemotron:70b"]
DOMS = ["2D_Triangular", "3D_Triangular", "2D_Triangulo_Sierpinski",
        "2D_Alfombra_Sierpinski", "3D_Piramide_Sierpinski", "3D_Menger_Sponge"]


def load(sub):
    d = {}
    for p in glob.glob(os.path.join(a.graded, sub, "*.json")):
        r = json.load(open(p)); d[r["file"]] = r
    return d


def pass_at_k(c, n, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


rep, raw, cov = load("repaired"), load("raw"), load("coverage")

# A response whose grader was killed from outside (time limit reached inside a
# call that cannot be interrupted, or out of memory) has no JSON. It is recorded
# as a failure, with the same verdict as a response that reaches the time limit.
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "responses", "*.txt")))
if len(names) != 1188:
    raise SystemExit("expected 1188 responses, found %d" % len(names))
for f in names:
    dom, model, stage, r = f.split("__")
    base = {"file": f, "domain": dom, "model": model.replace("_", ":", 1),
            "stage": int(stage[2:]), "rep": r}
    for table, name in ((raw, "raw"), (rep, "repaired")):
        if f not in table:
            print("no grade for %s (%s): recorded as a failure" % (f, name))
            table[f] = dict(base, status="RUNTIME_ERROR", note="TO()", passed=False, first_fail=0)
    if cov and f not in cov:
        print("no coverage for %s: recorded as 0" % f)
        cov[f] = dict(base, status="EVAL_TIMEOUT", any_order=0.0)
os.makedirs(a.out, exist_ok=True)

rows = []
for f in sorted(rep):
    x, y = rep[f], raw[f]; g = x.get("repair") or {}
    rows.append(dict(domain=x["domain"], model=x["model"], stage=x["stage"], rep=x["rep"],
                     passed_repaired=int(x["passed"]), status_repaired=x["status"],
                     passed_raw=int(y["passed"]), status_raw=y["status"],
                     first_fail=x.get("first_fail"), repair_extract=g.get("extract"),
                     repair_alias=g.get("aliased_from"), repair_dropped=g.get("dropped_toplevel"),
                     note=(x.get("note") or "")[:120]))
with open(os.path.join(a.out, "passk_responses.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

cells = []
for m in MODELS:
    for d in DOMS:
        for s in (20, 50, 100):
            key = lambda r: "%s__%s__st%d__%s" % (d, m.replace(":", "_"), s, r)
            c = sum(1 for r in "12345" if rep[key(r)]["passed"])
            cr = sum(1 for r in "12345" if raw[key(r)]["passed"])
            cells.append({"model": m, "domain": d, "stage": s, "c_of_5": c,
                          "greedy": int(rep[key("greedy")]["passed"]),
                          "pass@1": round(pass_at_k(c, 5, 1), 4),
                          "pass@3": round(pass_at_k(c, 5, 3), 4),
                          "pass@5": round(pass_at_k(c, 5, 5), 4), "c_of_5_raw": cr})
with open(os.path.join(a.out, "passk_cells.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)

if cov:
    with open(os.path.join(a.out, "coverage.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "model", "stage", "rep", "status", "coverage", "n_eval"])
        for f in sorted(cov):
            x = cov[f]
            w.writerow([x["domain"], x["model"], x["stage"], x["rep"], x["status"],
                        "%.6f" % (x.get("any_order") or 0.0), x.get("n_eval", "")])

print("exact after repair: %d, before: %d, of %d responses"
      % (sum(r["passed_repaired"] for r in rows), sum(r["passed_raw"] for r in rows), len(rows)))
