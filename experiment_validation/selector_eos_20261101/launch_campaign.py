"""Start the frozen full campaign as a durable local process, never duplicate it."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    queue = HERE / 'queue'
    queue.mkdir(exist_ok=True)
    status_path = queue / 'status.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if old.get('complete'):
            print(json.dumps({'status': 'already_completed', 'completed_stages': old['completed_stages']}))
            return
        pid = old.get('supervisor_pid')
        proc = Path(f'/proc/{pid}/cmdline') if pid else None
        if proc and proc.exists() and b'run_queue.py' in proc.read_bytes():
            print(json.dumps({'status': 'already_running', 'pid': pid}))
            return
        raise RuntimeError('Previous queue stopped. Inspect its failure and source bindings before explicitly resuming it.')
    command = [sys.executable, str(HERE / 'run_queue.py'), '--plan', str(HERE / 'queue_plan.json')]
    subprocess.run(command + ['--check'], cwd=ROOT, check=True)
    with (queue / 'supervisor.log').open('ab') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    receipt = {'launched_utc': datetime.now(timezone.utc).isoformat(),
               'pid': child.pid, 'command': command, 'working_dir': str(ROOT),
               'scope': 'All 6600 corrected-selector reconstructions and queued independent verification. No manuscript values promoted while incomplete.'}
    (queue / 'launch_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for _ in range(50):
        if child.poll() is not None:
            raise RuntimeError(f'Queue exited on launch with {child.returncode}; inspect supervisor.log')
        if status_path.exists():
            current = json.loads(status_path.read_text())
            if current.get('status') == 'running':
                print(json.dumps({'status': 'running', 'supervisor_pid': child.pid,
                                  'active_stage': current.get('active_stage'),
                                  'status_path': str(status_path)}, indent=2))
                return
        time.sleep(0.1)
    raise RuntimeError('Queue did not create its status promptly; inspect the launched process before retrying')


if __name__ == '__main__':
    main()
