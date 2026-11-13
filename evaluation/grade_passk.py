"""Ordered criterion: the response is correct only if map_to_coordinates(n)
equals the ground truth for every n in [0, 10^6). Aborts at the first
mismatch and records its index.

MODE=raw       code taken from the first fenced block, or the whole response
MODE=repaired  code after the syntactic repair of repair.py (default)

Usage: GT_DIR=gt OUT_DIR=graded/repaired python3 grade_passk.py <response.txt>
"""
import sys, os, re, json, math, array, signal, itertools, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repair as R

GT_DIR = os.environ["GT_DIR"]; OUT_DIR = os.environ["OUT_DIR"]
MODE   = os.environ.get("MODE", "repaired")
FN     = "map_to_coordinates"
# seconds allowed to evaluate the 10^6 indices of one response; reaching the
# limit inside the function is recorded as RUNTIME_ERROR with note TO().
# TIME_LIMIT=0 disables it.
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "1200"))

path = sys.argv[1]
base = os.path.basename(path)[:-4]
dom, model, stage, rep = base.split("__")
dst  = os.path.join(OUT_DIR, base + ".json")
rec  = {"file": base, "domain": dom, "model": model.replace("_", ":", 1),
        "stage": int(stage[2:]), "rep": rep, "mode": MODE}

def out(d):
    with open(dst, "w") as f: json.dump(d, f)
    os._exit(0)

content = open(path, encoding="utf-8", errors="replace").read()
if not content.strip():
    rec.update(status="EMPTY_RESPONSE", passed=False, first_fail=0); out(rec)

if MODE == "raw":
    m = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
    code = (m.group(1) if m else content).strip()
    rec["repair"] = None
else:
    code, log = R.repair(content)
    rec["repair"] = log
    if code is None:
        rec.update(status="NO_CODE", passed=False, first_fail=0); out(rec)
if not code:
    rec.update(status="NO_CODE", passed=False, first_fail=0); out(rec)

class TO(Exception): pass
def _to(*a): raise TO()
signal.signal(signal.SIGALRM, _to)

def bad(*a, **k): raise RuntimeError("exit()")
from typing import Tuple, List, Dict, Any, Set, Optional
g = {"__builtins__": __builtins__, "math": math, "itertools": itertools,
     "functools": functools, "Tuple": Tuple, "List": List, "Dict": Dict,
     "Any": Any, "Set": Set, "Optional": Optional, "exit": bad, "quit": bad}

signal.alarm(20)
try:
    loc = {}
    exec(compile(code, "<model>", "exec"), g, loc)
    fn = loc.get(FN) or g.get(FN)
    signal.alarm(0)
except TO:
    rec.update(status="EXEC_TIMEOUT", passed=False, first_fail=0); out(rec)
except SyntaxError as e:
    rec.update(status="SYNTAX_ERROR", note=str(e)[:200], passed=False, first_fail=0); out(rec)
except BaseException as e:
    rec.update(status="EXEC_ERROR", note=repr(e)[:200], passed=False, first_fail=0); out(rec)
if not callable(fn):
    rec.update(status="NO_FUNCTION", passed=False, first_fail=0); out(rec)

meta = json.load(open(os.path.join(GT_DIR, "meta.json")))[dom]
N, D = meta["n"], meta["dim"]
gt = array.array("i")
with open(os.path.join(GT_DIR, dom + ".bin"), "rb") as f: gt.fromfile(f, N * D)

first_fail = None; status = "OK"; note = ""; reached = 0
signal.alarm(TIME_LIMIT)
try:
    for n in range(N):
        try: r = fn(n)
        except BaseException as e:
            first_fail = n; status = "RUNTIME_ERROR"; note = repr(e)[:160]; break
        if not isinstance(r, (tuple, list)) or len(r) != D:
            first_fail = n; status = "BAD_SHAPE"; note = repr(r)[:100]; break
        b = n * D; ok = True
        for j in range(D):
            if r[j] != gt[b + j]: ok = False; break
        if not ok:
            first_fail = n; status = "MISMATCH"
            note = "%r vs %r" % (tuple(r), tuple(gt[b:b + D])); break
        reached = n + 1
    signal.alarm(0)
except TO:
    status = "EVAL_TIMEOUT"; first_fail = reached
except BaseException as e:
    status = "EVAL_ERROR"; note = repr(e)[:160]; first_fail = reached

rec.update(status=status, note=note, first_fail=first_fail, reached=reached,
           passed=(first_fail is None))
out(rec)
