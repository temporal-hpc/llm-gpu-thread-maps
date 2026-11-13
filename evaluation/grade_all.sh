#!/bin/bash
# Grades the 1,188 responses three times:
#   graded/raw       Ordered criterion, code taken as the model emitted it
#   graded/repaired  Ordered criterion, after the syntactic repair (repair.py)
#   graded/coverage  Domain Coverage over the first 10^5 indices, after repair
# One JSON per response in each directory. Run from the repository root,
# after download_domains.sh:   bash evaluation/grade_all.sh [parallel jobs]
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${1:-$(nproc)}"
EV="$ROOT/evaluation"; G="$ROOT/graded"
export GT_DIR="$ROOT/gt"

[ -f "$GT_DIR/meta.json" ] || python3 "$EV/build_gt.py"
mkdir -p "$G/raw" "$G/repaired" "$G/coverage"

echo "[1/3] Ordered, raw";      ls "$ROOT"/responses/*.txt | MODE=raw      OUT_DIR="$G/raw" \
    xargs -P "$J" -I{} timeout -k 5 1300 python3 "$EV/grade_passk.py" {} || true
echo "[2/3] Ordered, repaired"; ls "$ROOT"/responses/*.txt | MODE=repaired OUT_DIR="$G/repaired" \
    xargs -P "$J" -I{} timeout -k 5 1300 python3 "$EV/grade_passk.py" {} || true
echo "[3/3] Domain Coverage";   ls "$ROOT"/responses/*.txt | OUT_DIR="$G/coverage" \
    xargs -P "$J" -I{} timeout -k 5 700 python3 "$EV/grade_anyorder.py" {} || true

for d in raw repaired coverage; do echo "$d: $(ls "$G/$d" | wc -l) graded"; done
python3 "$EV/aggregate.py"
