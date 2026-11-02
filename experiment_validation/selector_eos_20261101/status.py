"""Read durable campaign status without loading a model or modifying files."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os

HERE = Path(__file__).resolve().parent
RUNS = [('archived', 'inference/cuda_full', 2100),
        ('tinyllama', 'inference/fresh_tinyllama_cuda', 300),
        ('phi', 'inference/fresh_phi_cuda', 300),
        ('secondary', 'secondary/cuda_full', 3900)]


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def main():
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'expected': 6600,
              'completed_reconstructions': 0, 'runs': {}}
    queue = read(HERE / 'queue/status.json')
    result['queue'] = queue
    if queue:
        pid = queue.get('supervisor_pid')
        proc = Path(f'/proc/{pid}/cmdline') if pid else None
        result['supervisor_alive'] = bool(proc and proc.exists() and b'run_queue.py' in proc.read_bytes())
    for label, name, expected in RUNS:
        status = read(HERE / name / 'status.json') or {'status': 'pending', 'completed': 0, 'complete': False}
        result['runs'][label] = {'expected': expected, **status}
        result['completed_reconstructions'] += status.get('completed', 0)
    result['all_checks_complete'] = bool(queue and queue.get('complete'))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
