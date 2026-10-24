#!/usr/bin/env python3
"""Run all manifest configs concurrently, saving separate logs and exit codes."""

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    experiment = Path(__file__).resolve().parent
    project = experiment.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modal', help='Path to Modal executable')
    parser.add_argument('--logs-root', type=Path, default=experiment / 'logs')
    args = parser.parse_args()
    modal = args.modal or str(project / 'env/bin/modal')
    if not Path(modal).is_file():
        modal = shutil.which(modal) or shutil.which('modal')
    if not modal:
        parser.error('Modal executable not found; use --modal /path/to/modal')
    modal = str(Path(modal).resolve())

    with (experiment / 'manifest.csv').open(newline='') as handle:
        configs = [experiment / row['config'] for row in csv.DictReader(handle)]
    for config in configs:
        if not config.is_file():
            parser.error(f'Missing config: {config}')

    from datetime import datetime

    args.logs_root.mkdir(parents=True, exist_ok=True)
    logs = Path(tempfile.mkdtemp(
        prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=args.logs_root,
    )).resolve()
    print(f'Logs: {logs}', flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    failed = 0
    active = []
    try:
        with (logs / 'summary.csv').open('w', newline='') as summary:
            writer = csv.writer(summary)
            writer.writerow(['config', 'pid', 'exit_code', 'log'])
            summary.flush()
            print(f'Starting all {len(configs)} configs concurrently', flush=True)
            for config in configs:
                log_path = logs / f'{config.stem}.log'
                with log_path.open('w') as output:
                    process = subprocess.Popen(
                        [modal, 'run', str(experiment / 'modal_app.py'), '--config', str(config)],
                        cwd=project, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, env=env,
                    )
                active.append((config, process, log_path))
                (logs / f'{config.stem}.pid').write_text(f'{process.pid}\n')
                print(f'  Started {config.name} (PID {process.pid})', flush=True)
            for config, process, log_path in active:
                code = process.wait()
                (logs / f'{config.stem}.exit_code').write_text(f'{code}\n')
                writer.writerow([config.name, process.pid, code, log_path.name])
                summary.flush()
                failed += code != 0
                print(f'  Finished {config.name}: exit {code}', flush=True)
            active.clear()
    finally:
        # Interrupting the launcher also stops any remaining local Modal clients.
        for _, process, _ in active:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    print(f'Finished {len(configs)} configs; {failed} failed. Logs: {logs}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; remaining local clients stopped.', file=sys.stderr)
        sys.exit(130)
