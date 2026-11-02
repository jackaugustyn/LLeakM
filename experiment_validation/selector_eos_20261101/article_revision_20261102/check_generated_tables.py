#!/usr/bin/env python3
"""Check generated tables with the separate pre-existing numeric parser.

This uses the independent local manuscript checker's implementation. The
canonical input inventory is read only for directory identifiers
and input counts. Full manuscript prose/layout checks are a separate step.
"""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKER = HERE / "check_manuscript_numbers.py"
code = CHECKER.read_text()
namespace = {"__file__": str(CHECKER), "__name__": "independent_table_checker"}
exec(compile(code, str(CHECKER), "exec"), namespace)
namespace.update(HERE=HERE, ROOT=ROOT, TEX=HERE / "tables.tex",
                 FACTS=HERE.parent / "validation", FIGURES=HERE)
sys.argv = [__file__, "--tables-only"]
namespace["main"]()
report_path = HERE / "table_numeric_validation.json"
report = json.loads(report_path.read_text())
report["checker_implementation"] = str(CHECKER.relative_to(ROOT))
report["checker_implementation_sha256"] = hashlib.sha256(CHECKER.read_bytes()).hexdigest()
report["input_artifact"] = "tables.tex generated from the current EOS campaign; not the shared manuscript"
report["metadata_source_inventory_scope"] = "The canonical inventory supplies original victim-response directory names/counts only; no previous reconstruction scores are read."
report_path.write_text(json.dumps(report, indent=2) + "\n")
