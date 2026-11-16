#!/usr/bin/env python3
"""Launch all 13 fresh runs, download results, then calculate paired CIs."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common import ROOT, load_inputs, read_json, write_json


def launch(modal, runs_root):
    manifest, _, _ = load_inputs()
    runs_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_'),
                                      dir=runs_root)).resolve()
    invocation = directory.name
    (directory / 'logs').mkdir()
    (directory / 'results').mkdir()
    shutil.copyfile(ROOT / 'manifest.json', directory / 'manifest.json')
    shutil.copytree(ROOT / 'data', directory / 'data')
    (directory / 'configs').mkdir()
    for run in manifest['runs']:
        shutil.copyfile(ROOT / run['config'], directory / 'configs' / run['config'])
    print(f'Experiment invocation: {directory}', flush=True)
    active = []
    status = []
    try:
        for run in manifest['runs']:
            label = run['label']
            log = directory / 'logs' / f'{label}.log'
            receipt = directory / 'logs' / f'{label}.receipt.json'
            command = [modal, 'run', str(ROOT / 'modal_app.py'), '--config', str(ROOT / run['config']),
                       '--invocation-id', invocation, '--receipt', str(receipt)]
            with log.open('w') as stream:
                process = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONHASHSEED='0'),
                                           start_new_session=True)
            active.append((run, process, receipt))
            (directory / 'logs' / f'{label}.pid').write_text(f'{process.pid}\n')
            print(f'Started {label}: PID {process.pid}', flush=True)
        for run, process, receipt in active:
            code = process.wait()
            label = run['label']
            (directory / 'logs' / f'{label}.exit_code').write_text(f'{code}\n')
            status.append({'label': label, 'pid': process.pid, 'exit_code': code})
            write_json(directory / 'status.json', status)
            print(f'Finished {label}: exit {code}', flush=True)
        active.clear()
    finally:
        for _, process, _ in active:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if any(row['exit_code'] for row in status):
        raise RuntimeError(f'One or more model runs failed; analysis not started. See {directory}')

    for run in manifest['runs']:
        label = run['label']
        receipt = read_json(directory / 'logs' / f'{label}.receipt.json')
        if receipt['label'] != label or receipt['invocation_id'] != invocation:
            raise ValueError('Receipt does not belong to this invocation')
        destination = directory / 'results' / label
        destination.mkdir()
        with (directory / 'logs' / f'{label}.download.log').open('w') as stream:
            subprocess.run([modal, 'volume', 'get', 'safety-bench-results',
                            receipt['remote_path'], str(destination)], check=True,
                           stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / 'analyze.py'), '--run-dir', str(directory)], check=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modal', default=str(ROOT.parent / 'env/bin/modal'))
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--validate-only', action='store_true', help='Validate inputs without contacting Modal')
    args = parser.parse_args()
    load_inputs()
    if args.validate_only:
        print('Validated 13 configs and the frozen 300-prompt evaluation set. No jobs launched.')
        return
    executable = shutil.which(args.modal)
    if executable is None:
        parser.error('Modal executable not found; use --modal /path/to/modal')
    directory = launch(str(Path(executable).resolve()), args.runs_root)
    print(f'Completed: {directory / "analysis/report.md"}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; remaining local clients stopped.', file=sys.stderr)
        sys.exit(130)
