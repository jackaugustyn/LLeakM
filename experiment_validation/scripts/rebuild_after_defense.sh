#!/usr/bin/env bash
set -u
cd /home/paugustynowicz/LLeakM
source .venv/bin/activate 2>/dev/null

DEFENSE_PID=18099
echo "[watch] waiting for defense_eval PID ${DEFENSE_PID} to finish..."
while kill -0 "${DEFENSE_PID}" 2>/dev/null; do
  sleep 60
done
echo "[watch] defense_eval finished; rebuilding artifacts"

# Drop any truncated final JSONL line before aggregating.
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiment_validation/analysis/defense_samples.jsonl')
good = []
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    try:
        json.loads(line); good.append(line)
    except json.JSONDecodeError:
        pass
p.write_text("\n".join(good) + "\n", encoding="utf-8")
print(f"[watch] kept {len(good)} valid samples")
PY

python experiment_validation/scripts/defense_eval.py --aggregate-only
python experiment_validation/scripts/make_figures.py
python experiment_validation/scripts/make_repro_manifest.py

cd article1
pdflatex -interaction=nonstopmode tifs_main.tex > /dev/null 2>&1
bibtex tifs_main > /dev/null 2>&1
pdflatex -interaction=nonstopmode tifs_main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode tifs_main.tex > /dev/null 2>&1
echo "[watch] REBUILD_COMPLETE pages=$(pdfinfo tifs_main.pdf 2>/dev/null | awk '/Pages/{print $2}') pdf_bytes=$(stat -c%s tifs_main.pdf)"
