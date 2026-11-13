#!/bin/bash
# Grades the 1,188 responses three times:
#   graded/raw       Correctness criterion, code taken as the model emitted it
#   graded/repaired  Correctness criterion, after the syntactic repair (repair.py)
#   graded/coverage  Domain Coverage over the first 10^5 indices, after repair
# One JSON per response in each directory. Resumable. Run from the repository root,
# after download_domains.sh:   bash evaluation/grade_all.sh [parallel jobs]
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${1:-$(nproc)}"
EV="$ROOT/evaluation"; G="$ROOT/graded"
export GT_DIR="$ROOT/gt"

[ -f "$GT_DIR/meta.json" ] || python3 "$EV/build_gt.py"
mkdir -p "$G/raw" "$G/repaired" "$G/coverage"

# grade <output dir> <timeout> <grader>: one response per job, skipping the
# responses already graded, so an interrupted run can be resumed. A grader
# killed by the time limit or by the system leaves no JSON; aggregate.py then
# records that response as a failure.
grade () {
  ls "$ROOT"/responses/*.txt | OUT_DIR="$1" xargs -P "$J" -I{} bash -c \
    'f=$(basename "$1" .txt); [ -s "$OUT_DIR/$f.json" ] || timeout -k 5 '"$2"' python3 "'"$3"'" "$1" || true' _ {} || true
}
echo "[1/3] Correctness, raw";      MODE=raw      grade "$G/raw"      1300 "$EV/grade_passk.py"
echo "[2/3] Correctness, repaired"; MODE=repaired grade "$G/repaired" 1300 "$EV/grade_passk.py"
echo "[3/3] Domain Coverage";                 grade "$G/coverage"  700 "$EV/grade_anyorder.py"

for d in raw repaired coverage; do echo "$d: $(ls "$G/$d" | wc -l) graded"; done
python3 "$EV/aggregate.py" --out "${OUT:-$ROOT/data}"
