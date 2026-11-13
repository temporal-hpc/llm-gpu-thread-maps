"""Domain Coverage: fraction of the distinct coordinates of the domain that
the function produces at least once over the first 10^5 indices, regardless
of order. It does not abort at the first mismatch; runtime errors are counted
and the evaluation continues.

Usage: GT_DIR=gt OUT_DIR=graded/coverage python3 grade_anyorder.py <response.txt>
"""
import sys, os, re, json, math, array, signal, itertools, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repair as R

GT_DIR = os.environ["GT_DIR"]; OUT_DIR = os.environ["OUT_DIR"]
FN = "map_to_coordinates"
path = sys.argv[1]; base = os.path.basename(path)[:-4]
dom, model, stage, rep = base.split("__")
dst = os.path.join(OUT_DIR, base + ".json")
rec = {"file": base, "domain": dom, "model": model.replace("_", ":", 1),
       "stage": int(stage[2:]), "rep": rep}

def out(d):
    with open(dst, "w") as f: json.dump(d, f)
    os._exit(0)

content = open(path, encoding="utf-8", errors="replace").read()
if not content.strip(): rec.update(status="EMPTY_RESPONSE", any_order=0.0); out(rec)
code, log = R.repair(content)
if not code: rec.update(status="NO_CODE", any_order=0.0); out(rec)

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
def bad(*a, **k): raise RuntimeError("exit()")
from typing import Tuple, List, Dict, Any, Set, Optional
g = {"__builtins__": __builtins__, "math": math, "itertools": itertools,
     "functools": functools, "Tuple": Tuple, "List": List, "Dict": Dict,
     "Any": Any, "Set": Set, "Optional": Optional, "exit": bad, "quit": bad}
signal.alarm(20)
try:
    loc = {}; exec(compile(code, "<model>", "exec"), g, loc)
    fn = loc.get(FN) or g.get(FN); signal.alarm(0)
except BaseException as e:
    rec.update(status="EXEC_ERROR", note=repr(e)[:120], any_order=0.0); out(rec)
if not callable(fn): rec.update(status="NO_FUNCTION", any_order=0.0); out(rec)

meta = json.load(open(os.path.join(GT_DIR, "meta.json")))[dom]
N, D = meta["n"], meta["dim"]
NA = min(N, 100_000)   # coverage over the first 10^5 indices
gt = array.array("i")
with open(os.path.join(GT_DIR, dom + ".bin"), "rb") as f: gt.fromfile(f, NA * D)
gtset = set()
for n in range(NA):
    b = n * D; gtset.add(tuple(gt[b:b + D]))

produced = set(); errors = 0; status = "OK"
signal.alarm(600)
try:
    for n in range(NA):
        try: r = fn(n)
        except BaseException: errors += 1; continue
        if isinstance(r, (tuple, list)) and len(r) == D:
            try: produced.add(tuple(int(v) for v in r))
            except BaseException: errors += 1
        else: errors += 1
    signal.alarm(0)
except TO:
    status = "EVAL_TIMEOUT"
except BaseException as e:
    status = "EVAL_ERROR"; rec["note"] = repr(e)[:120]

hit = len(produced & gtset)
rec.update(status=status, errors=errors, n_eval=NA, n_gt=len(gtset), n_produced=len(produced),
           any_order=hit / len(gtset) if gtset else 0.0)
out(rec)
