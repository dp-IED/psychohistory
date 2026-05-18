#!/usr/bin/env bash
# Batch chain: 3 cycles of 5 Polymarket questions + reflection
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=============================================="
echo "  BATCH CHAIN — 3 cycles × (5 questions + reflection)"
echo "  Vault: graph-vault"
echo "  Started: $(date)"
echo "=============================================="

for i in 1 2 3; do
    echo ""
    echo "=============================================="
    echo "  CYCLE $i / 3 — BATCH OF 5"
    echo "  $(date)"
    echo "=============================================="

    PYTHONPATH="." python3 -m scripts.run_backtest \
        --source polymarket \
        --max-questions 5 \
        --vault graph-vault \
        --policy graph-vault/_forecast_instructions.md \
        2>&1 && BATCH_OK=0 || BATCH_OK=1

    echo ""
    echo "Batch $i exit code: $BATCH_OK"

    if [ "$BATCH_OK" -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "  CYCLE $i / 3 — REFLECTION"
        echo "  $(date)"
        echo "=============================================="

        PYTHONPATH="." python3 -m scripts.reflect_graph \
            2>&1 && REFLECT_OK=0 || REFLECT_OK=1

        echo "Reflection $i exit code: $REFLECT_OK"
    else
        echo "Skipping reflection — batch had failures"
    fi
done

echo ""
echo "=============================================="
echo "  ALL CYCLES COMPLETE"
echo "  Finished: $(date)"
echo "=============================================="
echo ""
echo "=== Final vault stats ==="
PYTHONPATH="." python3 -c "
from harness.runs import runs_count, mean_brier, brier_by_category
mb = mean_brier('graph-vault')
by_cat = brier_by_category('graph-vault')
print(f'Total runs: {runs_count(\"graph-vault\")}')
print(f'Mean Brier: {mb:.4f}' if mb else 'Mean Brier: N/A')
print('By category:')
for c, b in sorted(by_cat.items()):
    print(f'  {c}: {b:.4f}')
"
