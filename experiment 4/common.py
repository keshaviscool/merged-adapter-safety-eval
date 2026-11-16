"""Experiment-local input validation; no GPU or Modal imports."""
import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DETECTORS = ('llama_guard_3', 'harm_cls')
ADAPTERS = [
    'Akarshika/qwen25-3b-text2sql-lora',
    'madokalif/qwen2.5-3b-translation-warmup-adapter',
    'balaguhanesh/tool-call-ft',
    'Pilestin/qwen2.5-3b-debug-v1-lora',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_inputs(root=ROOT):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    if (manifest.get('schema_version') != 1 or manifest.get('expected_prompts') != 300
            or manifest.get('bootstrap_resamples') != 1000 or manifest.get('bootstrap_seed') != 42):
        raise ValueError('Unexpected experiment/bootstrap specification')
    dataset = root / 'data/harmbench_behaviors_text_all.csv'
    if digest(dataset) != manifest['dataset_sha256']:
        raise ValueError('Frozen dataset checksum mismatch')
    with dataset.open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['FunctionalCategory'] in ('standard', 'contextual')]
    if len(rows) != 300 or len({r['BehaviorID'] for r in rows}) != 300:
        raise ValueError('Expected 300 unique standard/contextual prompts')
    configs = {}
    signature = None
    expected = {('base', None), ('linear', None), ('ties', None)} | {
        (method, seed) for method in ('dare_linear', 'dare_ties') for seed in range(42, 47)
    }
    runs = manifest['runs']
    if (len(runs) != 13 or len({r['label'] for r in runs}) != 13
            or {(r['method'], r['merge_seed']) for r in runs} != expected):
        raise ValueError('Manifest must contain the specified 13 unique runs')
    for run in runs:
        if Path(run['config']).name != run['config'] or Path(run['label']).name != run['label']:
            raise ValueError('Run names must be single path components')
        config_root = root / 'configs' if (root / 'configs').is_dir() else root
        cfg = yaml.safe_load((config_root / run['config']).read_text())
        method = run['method']
        adapters = cfg['model']['adapters']
        expected_adapters = [] if method == 'base' else [
            {'name': f'a{i}', 'path': path, 'weight': 0.25}
            for i, path in enumerate(ADAPTERS, 1)
        ]
        if adapters != expected_adapters or cfg['model']['base_model'] != 'Qwen/Qwen2.5-3B-Instruct':
            raise ValueError(f'Unexpected model/adapters: {run["label"]}')
        expected_merge = {'strategy': 'linear' if method == 'base' else method,
                          'density': None if method in ('base', 'linear') else 0.75,
                          'majority_sign_method': 'total', 'svd_rank': None}
        if cfg['merge'] != expected_merge:
            raise ValueError(f'Unexpected merge settings: {run["label"]}')
        if cfg['datasets']['harmbench'] != {
            'enabled': True, 'functional_categories': ['standard', 'contextual'], 'max_examples': None
        } or any(cfg['datasets'][d]['enabled'] for d in ('advbench', 'alpaca')):
            raise ValueError('Only the complete supported HarmBench set must be enabled')
        if any(cfg['detectors'][d]['enabled'] != (d in ('llama_guard3', 'harm_cls'))
               for d in cfg['detectors']):
            raise ValueError('Only the two ASR classifiers must be enabled')
        current = {k: cfg[k] for k in ('generation', 'detectors', 'datasets')}
        current['model_settings'] = {k: v for k, v in cfg['model'].items() if k != 'adapters'}
        if signature is not None and current != signature:
            raise ValueError('Generation, detector, dataset, and base settings must match')
        signature = current
        if cfg['generation']['seed'] != 42:
            raise ValueError('Generation seed must stay fixed at 42')
        configs[run['label']] = cfg
    return manifest, configs, rows
