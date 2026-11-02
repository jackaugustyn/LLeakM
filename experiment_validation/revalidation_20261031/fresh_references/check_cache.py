"""Check pinned local weights/tokenizer files; optionally finish missing downloads."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "cache_validation.json")
    args = parser.parse_args()
    models = json.loads((HERE / "models_pinned.json").read_text())["models"]
    results = []
    for model in models:
        if args.download:
            snapshot_download(model["model_id"], revision=model["revision"], allow_patterns=model["files"], max_workers=2)
        for name in model["files"]:
            result = dict(model_id=model["model_id"], revision=model["revision"], filename=name, verified=False)
            try:
                path = Path(hf_hub_download(model["model_id"], name, revision=model["revision"], local_files_only=True))
                expected_size = model["file_sizes"][name]
                if path.stat().st_size != expected_size:
                    raise ValueError(f"Size mismatch: {path.stat().st_size} != {expected_size}")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                actual_hash = digest.hexdigest()
                expected_hash = model.get("lfs_sha256", {}).get(name)
                if expected_hash and actual_hash != expected_hash:
                    raise ValueError("Pinned LFS SHA256 mismatch")
                result.update(verified=True, bytes=expected_size, sha256=actual_hash, path=str(path))
            except Exception as exc:
                result.update(error_type=type(exc).__name__, error=str(exc))
            results.append(result)
    report = dict(complete=all(item["verified"] for item in results), checked_utc=datetime.now(timezone.utc).isoformat(), files=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps({"complete": report["complete"], "verified_files": sum(item["verified"] for item in results), "expected_files": len(results), "report": str(args.output)}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
