#!/usr/bin/env python3
"""Build one Ollama request per inference of the grid.

Grid: 11 models x 6 domains x 3 sample counts (20, 50, 100) = 198 cells.
Each cell is sampled 5 times with the default decoding parameters of the
model, plus one greedy pass (temperature = 0): 1,188 requests in total.

The prompt is prompt_template.txt with the first m lines of the domain file
in place of __MAPPING_DATA_HERE__. The same prompt is used for the six
requests of a cell.

Usage: python3 inference/make_payloads.py [--domains domains] [--out payloads]
Requests whose response already exists in responses/ are skipped, so the
run can be resumed.
"""
import argparse, glob, json, os

MODELS = ["gpt-oss:120b", "gpt-oss:20b", "qwen3:32b", "deepseek-r1:70b",
          "llama3.3", "llama4", "gemma3:27b", "gemma3:12b", "mistral-nemo",
          "nemotron:70b", "qwen3:235b"]
SAMPLES = [20, 50, 100]
REPS = ["1", "2", "3", "4", "5", "greedy"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--domains", default=os.path.join(ROOT, "domains"))
ap.add_argument("--out", default=os.path.join(ROOT, "payloads"))
ap.add_argument("--responses", default=os.path.join(ROOT, "responses"))
ap.add_argument("--all", action="store_true", help="do not skip existing responses")
a = ap.parse_args()

template = open(os.path.join(ROOT, "prompt_template.txt")).read()
domains = sorted(glob.glob(os.path.join(a.domains, "*.txt")))
if len(domains) != 6:
    raise SystemExit("expected 6 domain files in %s, found %d (run download_domains.sh)"
                     % (a.domains, len(domains)))
os.makedirs(a.out, exist_ok=True)

n = 0
for model in MODELS:
    tag = model.replace(":", "_")
    for dom in domains:
        name = os.path.basename(dom)[:-4]
        lines = open(dom).read().splitlines()
        for m in SAMPLES:
            prompt = template.replace("__MAPPING_DATA_HERE__", "\n".join(lines[:m]))
            for rep in REPS:
                base = "%s__%s__st%d__%s" % (name, tag, m, rep)
                done = os.path.join(a.responses, base + ".txt")
                if not a.all and os.path.exists(done) and os.path.getsize(done) > 0:
                    continue
                body = {"model": model, "prompt": prompt, "stream": False}
                if rep == "greedy":
                    body["options"] = {"temperature": 0}
                with open(os.path.join(a.out, base + ".json"), "w") as f:
                    json.dump(body, f)
                n += 1
print("%d payloads written to %s" % (n, a.out))
