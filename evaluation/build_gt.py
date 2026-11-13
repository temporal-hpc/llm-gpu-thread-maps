#!/usr/bin/env python3
"""Convert the six domain files into flat int32 arrays for the graders.

Each line of a domain file reads "Thread  <n> -> Coordenadas: (x, y[, z])".
The output is <out>/<domain>.bin (n*dim int32 values) and <out>/meta.json
with the number of points and the dimension of each domain.

Usage: python3 evaluation/build_gt.py [--domains domains] [--out gt]
"""
import argparse, array, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--domains", default=os.path.join(ROOT, "domains"))
ap.add_argument("--out", default=os.path.join(ROOT, "gt"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

meta = {}
for path in sorted(glob.glob(os.path.join(a.domains, "*.txt"))):
    name = os.path.basename(path)[:-4]
    flat = array.array("i"); n = 0; dim = None
    for line in open(path):
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        try:
            i = int(left.split()[-1])
            c = right.split(":", 1)[1].strip().strip("()[]")
            t = [int(v) for v in c.split(",")]
        except Exception:
            continue
        if i != n:
            raise SystemExit("non contiguous index in %s at %d" % (name, i))
        if dim is None:
            dim = len(t)
        elif len(t) != dim:
            raise SystemExit("dimension mismatch in %s at %d" % (name, i))
        flat.extend(t); n += 1
    with open(os.path.join(a.out, name + ".bin"), "wb") as g:
        flat.tofile(g)
    meta[name] = {"n": n, "dim": dim}
    print("%-26s %8d points, %dD" % (name, n, dim))
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
