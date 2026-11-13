#!/usr/bin/env python3
"""Structure of the code generated for the 3D Menger sponge.

Classifies the responses of that domain that emit code (after the syntactic
repair) by two textual properties of the source: whether it reduces the
index in radix 20, and whether it iterates over recursion levels, by an
explicit loop or by a self-recursive call.

Usage: python3 evaluation/menger_taxonomy.py
"""
import collections, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repair as R

ROOT = os.path.dirname(HERE)
RADIX = re.compile(r'(%\s*20\b|//\s*20\b|divmod\s*\([^,]+,\s*20\s*\)|\b20\s*\*\*|base\s*=\s*20\b)')
LOOP = re.compile(r'^\s*(while|for)\b', re.M)
DEF = re.compile(r'\bdef\s+(\w+)')

files = sorted(glob.glob(os.path.join(ROOT, "responses", "3D_Menger_Sponge__*.txt")))
stats = collections.Counter(); by_model = collections.Counter(); with_code = 0
for path in files:
    code, _ = R.repair(open(path, encoding="utf-8", errors="replace").read())
    if not code:
        continue
    with_code += 1
    radix = bool(RADIX.search(code))
    recursive = any(len(re.findall(r'\b%s\s*\(' % re.escape(name), code)) > 1
                    for name in set(DEF.findall(code)))
    iterates = bool(LOOP.search(code)) or recursive
    stats["Reduces the index in radix 20"] += radix
    stats["Iterates over recursion levels (loop or self-call)"] += iterates
    stats["Radix 20 but a single level, no iteration"] += radix and not iterates
    stats["Radix 20 and iteration over levels"] += radix and iterates
    if radix:
        by_model[os.path.basename(path).split("__")[1]] += 1

print("responses of the domain: %d, emitting code: %d\n" % (len(files), with_code))
for k, v in stats.items():
    print("  %-52s %3d  %5.1f%%" % (k, v, 100.0 * v / with_code))
print("\nradix 20 by model:", dict(by_model))
