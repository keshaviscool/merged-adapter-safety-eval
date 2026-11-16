"""Experiment 4 only. Importing this file does not launch paid functions."""
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

import modal

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = EXPERIMENT.parent

REMOTE = '/root/safety_bench'
app = modal.App('safety-bench-experiment4')
cache = modal.Volume.from_name('safety-bench-hf-cache', create_if_missing=True)
results = modal.Volume.from_name('safety-bench-results', create_if_missing=True)
adapters = modal.Volume.from_name('safety-bench-adapters', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git')
    .pip_install_from_requirements(str(PROJECT / 'requirements.txt'))
    .env({'HF_HOME': '/hf-cache', 'PYTHONUNBUFFERED': '1'})
    .add_local_dir(PROJECT / 'bench', remote_path=f'{REMOTE}/bench')
    .add_local_file(PROJECT / 'run_benchmark.py', remote_path=f'{REMOTE}/run_benchmark.py')
    .add_local_file(EXPERIMENT / 'benchmark_entrypoint.py', remote_path=f'{REMOTE}/benchmark_entrypoint.py')
    .add_local_file(EXPERIMENT / 'merge_seed.py', remote_path=f'{REMOTE}/merge_seed.py')
    .add_local_file(EXPERIMENT / 'data/harmbench_behaviors_text_all.csv',
                    remote_path=f'{REMOTE}/frozen_data/harmbench_behaviors_text_all.csv')
)


@app.function(image=image, gpu='A100-80GB', timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name('huggingface-secret')],
              volumes={'/hf-cache': cache, '/results': results, '/adapters': adapters})
def benchmark(config_yaml: str, provenance: dict) -> str:
    import hashlib
    import yaml

    dataset = Path(REMOTE) / 'frozen_data/harmbench_behaviors_text_all.csv'
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != provenance['dataset_sha256']:
        raise ValueError('Remote frozen dataset checksum mismatch')
    config = yaml.safe_load(config_yaml)
    root = Path('/results/experiment4') / provenance['invocation_id'] / provenance['label']
    config['output']['root'] = str(root)
    config['datasets']['cache_dir'] = str(dataset.parent)
    config_path = Path(REMOTE) / 'config_modal.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    provenance_path = Path(REMOTE) / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance))
    try:
        subprocess.run(['python', 'benchmark_entrypoint.py', '--config', str(config_path),
                        '--provenance', str(provenance_path)], cwd=REMOTE, check=True)
        completed = list(root.glob('*/summary.json'))
        if len(completed) != 1:
            raise RuntimeError('Expected one completed result')
        return str(completed[0].parent.relative_to('/results'))
    finally:
        results.commit()
        cache.commit()


@app.local_entrypoint()
def main(config: str, invocation_id: str = '', receipt: str = ''):
    # Modal also imports this module inside the GPU container. common.py and
    # the manifest/configs are local launcher inputs, not container dependencies.
    sys.path.insert(0, str(EXPERIMENT))
    from common import digest, load_inputs

    manifest, _, _ = load_inputs()
    path = Path(config).resolve()
    run = next((r for r in manifest['runs'] if EXPERIMENT / r['config'] == path), None)
    if run is None:
        raise ValueError('Config must be one of the Experiment 4 manifest entries')
    invocation_id = invocation_id or uuid.uuid4().hex
    if not re.fullmatch(r'[A-Za-z0-9_-]+', invocation_id):
        raise ValueError('Invalid invocation ID')
    provenance = {
        **run, 'invocation_id': invocation_id,
        'dataset_sha256': manifest['dataset_sha256'],
        'generation_seed': 42, 'config_sha256': digest(path),
        'source_sha256': {
            str(p.relative_to(PROJECT)): digest(p)
            for p in [PROJECT / 'run_benchmark.py', *sorted((PROJECT / 'bench').rglob('*.py')),
                      EXPERIMENT / 'merge_seed.py', EXPERIMENT / 'benchmark_entrypoint.py']
        },
    }
    remote_path = benchmark.remote(path.read_text(), provenance)
    if receipt:
        Path(receipt).write_text(json.dumps({**provenance, 'remote_path': remote_path}, indent=2) + '\n')
    print(f'Results: {remote_path}')
    print(f'modal volume get safety-bench-results {remote_path} ./results/')
