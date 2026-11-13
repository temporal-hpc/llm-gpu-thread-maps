#!/usr/bin/env python3
"""Effect of the syntactic repair over the 1,188 responses.

Reads data/passk_responses.csv and prints the number of responses under each
verdict before and after the repair, plus the checks that the repair is
monotonic (no response that was correct before fails after it).

Usage: python3 evaluation/repair_summary.py
"""
import collections, csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "passk_responses.csv"))))

LABEL = [("OK", "Exact over 10^6 points"), ("MISMATCH", "Ran, but incorrect mapping"),
         ("SYNTAX_ERROR", "Syntax error"), ("NO_FUNCTION", "No map_to_coordinates"),
         ("NO_CODE", "Prose only, no function defined"), ("EMPTY_RESPONSE", "Empty response"),
         ("RUNTIME_ERROR", "Runtime error"), ("BAD_SHAPE", "Malformed return"),
         ("EXEC_ERROR", "Load error")]
raw = collections.Counter(r["status_raw"] for r in rows)
rep = collections.Counter(r["status_repaired"] for r in rows)
other = (set(raw) | set(rep)) - {k for k, _ in LABEL}

print("%-34s %6s %8s" % ("Verdict", "Raw", "Repaired"))
for k, name in LABEL + [(k, k) for k in sorted(other)]:
    print("%-34s %6s %8d" % (name, raw[k] if raw[k] else "---", rep[k]))
print("%-34s %6d %8d" % ("Total", sum(raw.values()), sum(rep.values())))

gained = sum(1 for r in rows if r["passed_raw"] == "0" and r["passed_repaired"] == "1")
lost = sum(1 for r in rows if r["passed_raw"] == "1" and r["passed_repaired"] == "0")
not_run = sum(raw[k] for k in ("SYNTAX_ERROR", "NO_FUNCTION", "EXEC_ERROR"))
print("\ndid not reach execution before the repair: %d of %d (%.0f%%)"
      % (not_run, len(rows), 100.0 * not_run / len(rows)))
print("correct only after the repair: %d;  correct before and wrong after: %d" % (gained, lost))
