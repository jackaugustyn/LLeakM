"""Run the authorized revalidation stages sequentially, with durable status.

The first stage can attach to an already running inference process. A failed,
incomplete, changed-source or timed-out stage stops the queue. This supervisor
does not edit the manuscript or promote partial results to final tables.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def process_matches(pid, token):
    try:
        # Zombies have exited and no longer hold GPU allocations.
        stat = Path(f'/proc/{pid}/stat').read_text()
        if stat.rsplit(')', 1)[1].split()[0] == 'Z':
            return False
        return token in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def check_success(stage):
    for spec in stage.get('success_json', []):
        path = ROOT / spec['path']
        value = json.loads(path.read_text())
        for key, expected in spec['equals'].items():
            if value.get(key) != expected:
                raise RuntimeError(f"Stage {stage['name']}: {path} has {key}={value.get(key)!r}, expected {expected!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--check', action='store_true', help='Validate commands and source bindings, without execution.')
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.plan.resolve().parent / 'queue'
    names = [s['name'] for s in plan['stages']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Stage names must be nonempty and unique')
    for rel, expected in plan['source_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f'Source changed before queue: {rel}')
    for stage in plan['stages']:
        if bool(stage.get('attach_pid')) == bool(stage.get('command')):
            raise ValueError('Exactly one of command / attach_pid is required')
        if stage.get('command') and (not isinstance(stage['command'], list) or
                                    not all(isinstance(x, str) for x in stage['command'])):
            raise ValueError('Commands must be structured argument lists')
        if stage.get('timeout_hours', 24) <= 0:
            raise ValueError('Stage timeout must be positive')
    if args.check:
        print(json.dumps({'valid': True, 'stages': names}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / 'status.json'
    plan_sha = sha(args.plan)
    status = {'plan_sha256': plan_sha, 'started_utc': now(), 'supervisor_pid': os.getpid(),
              'status': 'running', 'complete': False, 'completed_stages': [], 'stage_results': []}
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior.get('plan_sha256') != plan_sha:
            raise RuntimeError('Queue plan differs from saved status; use a new plan directory')
        prior_pid = prior.get('active_pid')
        if prior_pid and process_start(prior_pid) is not None:
            same_process = prior.get('active_process_start') in {None, process_start(prior_pid)}
            prior_stage = next((s for s in plan['stages'] if s['name'] == prior.get('active_stage')), None)
            same_attachment = (prior_stage and prior_stage.get('attach_pid') == prior_pid and
                               process_matches(prior_pid, prior_stage['process_token']))
            if same_process and not same_attachment:
                raise RuntimeError(f'Previous stage process {prior_pid} is still alive; refusing duplicate execution')
        status['completed_stages'] = prior.get('completed_stages', [])
        status['stage_results'] = prior.get('stage_results', [])
    child = None
    stage = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    write(status_path, status)
    try:
        for stage in plan['stages']:
            if stage['name'] in status['completed_stages']:
                check_success(stage)
                continue
            for rel, expected in plan['source_sha256'].items():
                if sha(ROOT / rel) != expected:
                    raise RuntimeError(f'Source changed between stages: {rel}')
            started = time.monotonic()
            status.update(active_stage=stage['name'], stage_started_utc=now())
            print(f"{now()} START {stage['name']}", flush=True)
            log = None
            if stage.get('attach_pid'):
                pid = stage['attach_pid']
                token = stage['process_token']
                is_alive = lambda: process_matches(pid, token)
            else:
                log = (out / f"{stage['name']}.log").open('a')
                env = os.environ.copy()
                env.update(stage.get('env', {}))
                child = subprocess.Popen(stage['command'], cwd=ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                pid = child.pid
                is_alive = lambda: child.poll() is None
            status['active_pid'] = pid
            status['active_process_start'] = process_start(pid)
            write(status_path, status)
            try:
                while is_alive():
                    elapsed = time.monotonic() - started
                    status.update(heartbeat_utc=now(), stage_elapsed_seconds=elapsed)
                    progress = stage.get('progress_json')
                    if progress and (ROOT / progress).exists():
                        try:
                            status['stage_progress'] = json.loads((ROOT / progress).read_text())
                        except json.JSONDecodeError:
                            status['stage_progress'] = {'read_error': 'incomplete status JSON'}
                    write(status_path, status)
                    if elapsed > stage.get('timeout_hours', 24) * 3600:
                        print(f"{now()} TIMEOUT {stage['name']}; stopping the active stage", flush=True)
                        if child is not None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                        elif process_matches(pid, token):
                            os.kill(pid, signal.SIGTERM)
                        raise TimeoutError(stage['name'])
                    time.sleep(max(0.05, float(plan.get('monitor_interval_seconds', 15))))
                exit_code = child.returncode if child is not None else None
                if exit_code not in (None, 0):
                    raise RuntimeError(f"Stage {stage['name']} exited with code {exit_code}; see queue/{stage['name']}.log")
                check_success(stage)
            finally:
                if log:
                    log.close()
            status['stage_results'].append({'name': stage['name'], 'finished_utc': now(),
                                            'exit_code': exit_code, 'elapsed_seconds': time.monotonic() - started})
            status['completed_stages'].append(stage['name'])
            status.pop('stage_progress', None)
            write(status_path, status)
            print(f"{now()} FINISH {stage['name']}", flush=True)
            child = None
        status.update(status='completed', complete=True, finished_utc=now())
        status.pop('active_stage', None)
        status.pop('active_pid', None)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status.update(status='failed', complete=False, error_type=type(exc).__name__, error=str(exc), failed_utc=now())
        write(status_path, status)
        raise
    write(status_path, status)
    return 0


if __name__ == '__main__':
    sys.exit(main())
