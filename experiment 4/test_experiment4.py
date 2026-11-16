"""CPU-only and simulated-process tests. Never contact Modal or download models."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze
from common import ROOT, DETECTORS, digest, load_inputs, read_json, write_json
from merge_seed import isolated_merge_seed
from run_all import launch


def fixtures(destination):
    """Create complete synthetic artifacts with real frozen prompt identities."""
    manifest, configs, rows = load_inputs()
    expected = analyze.expected_prompts(rows)
    for run in manifest['runs']:
        label = run['label']
        root = destination / label
        root.mkdir(parents=True)
        cfg = configs[label]
        provenance = {**run, 'invocation_id': 'synthetic', 'dataset_sha256': manifest['dataset_sha256'],
                      'generation_seed': 42, 'config_sha256': digest(ROOT / run['config']),
                      'source_sha256': {'synthetic.py': 'test-only'}, 'packages': {'synthetic': '1'}}
        write_json(root / 'experiment4_provenance.json', provenance)
        (root / 'config_snapshot.yaml').write_text(yaml.safe_dump({'config': cfg}))
        summary = {'results': {'harmbench': {'asr': {}}}}
        flags = {d: [] for d in DETECTORS}
        for i, prompt in enumerate(expected.values()):
            for detector in DETECTORS:
                # Genuine paired variation, and deliberately unequal detector judgments.
                flags[detector].append((i + (run['merge_seed'] or 0)) % (5 if detector == DETECTORS[0] else 7) == 0)
        for detector in DETECTORS:
            for folder, partition in (('went_wrong', True), ('went_right', False)):
                (root / folder).mkdir(exist_ok=True)
                records = [
                    {'prompt': prompt, 'response': f'Synthetic response {i}.',
                     'verdicts': {d: {'flagged': flags[d][i], 'raw_output': 'test', 'details': {}}
                                  for d in DETECTORS}}
                    for i, prompt in enumerate(expected.values()) if flags[detector][i] == partition
                ]
                write_json(root / folder / f'harmbench__{detector}.json', list(reversed(records)))
            count = sum(flags[detector])
            summary['results']['harmbench']['asr'][detector] = {
                'n': 300, 'n_attack_success': count, 'asr': count / 300}
        write_json(root / 'summary.json', summary)


def synthetic_invocation(directory):
    manifest, _, _ = load_inputs()
    shutil.copyfile(ROOT / 'manifest.json', directory / 'manifest.json')
    shutil.copytree(ROOT / 'data', directory / 'data')
    (directory / 'configs').mkdir()
    (directory / 'logs').mkdir()
    fixtures(directory / 'results')
    for run in manifest['runs']:
        label = run['label']
        shutil.copyfile(ROOT / run['config'], directory / 'configs' / run['config'])
        meta_path = directory / 'results' / label / 'experiment4_provenance.json'
        info = read_json(meta_path)
        info['invocation_id'] = directory.name
        write_json(meta_path, info)
        write_json(directory / 'logs' / f'{label}.receipt.json',
                   {**info, 'remote_path': f'experiment4/{directory.name}/{label}/timestamp'})


class Experiment4Tests(unittest.TestCase):
    def test_container_import_without_local_helpers(self):
        # Import a copy with no common.py, manifest, configs, or project files.
        # Stub only Modal's declarations so this test cannot contact the service.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'modal_app.py'
            shutil.copyfile(ROOT / 'modal_app.py', path)
            script = '''
import builtins, runpy, sys, types
class Declaration:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
    def function(self, **kwargs): return lambda function: function
    def local_entrypoint(self): return lambda function: function
    @classmethod
    def from_name(cls, *args, **kwargs): return cls()
    @classmethod
    def debian_slim(cls, *args, **kwargs): return cls()
sys.modules['modal'] = types.SimpleNamespace(
    App=Declaration, Image=Declaration, Volume=Declaration, Secret=Declaration)
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'common':
        raise ModuleNotFoundError("No module named 'common'")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
module = runpy.run_path(sys.argv[1])
assert callable(module['benchmark'])
assert callable(module['main'])
'''
            result = subprocess.run([sys.executable, '-I', '-c', script, str(path)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_matrix_and_shared_loader(self):
        from bench.config import load_config

        manifest, configs, rows = load_inputs()
        self.assertEqual(len(rows), 300)
        for run in manifest['runs']:
            cfg = asdict(load_config(ROOT / run['config']))
            self.assertEqual(analyze.normalize_config(cfg), analyze.normalize_config(configs[run['label']]))
        self.assertEqual(sum(r['merge_seed'] is not None for r in manifest['runs']), 10)

    def test_paired_bootstrap_and_seed_statistics(self):
        x = np.arange(300) % 2
        indices = np.random.default_rng(42).integers(300, size=(1000, 300))
        identical = analyze.paired_delta(x, x, indices)
        self.assertEqual(identical['delta_pp'], 0)
        self.assertEqual(identical['ci95_delta_pp'], [0, 0])
        constant = analyze.paired_delta(np.ones(300), np.zeros(300), indices)
        self.assertEqual(constant['ci95_delta_pp'], [100, 100])
        self.assertEqual(analyze.paired_delta(1 - x, x, indices),
                         analyze.paired_delta(1 - x, x, np.random.default_rng(42).integers(300, size=(1000, 300))))
        stats = analyze.seed_stats([[0, 0], [0, 1], [1, 1]])
        self.assertEqual(stats['mean_asr_pct'], 50)
        self.assertEqual(stats['sample_sd_asr_pp'], 50)

    def test_alignment_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures(root)
            _, _, rows = load_inputs()
            expected = analyze.expected_prompts(rows)
            result = root / 'linear'
            vectors, _, _ = analyze.load_verdicts(result, expected)
            self.assertEqual(int(vectors['llama_guard_3'].sum()), 60)
            path = result / 'went_wrong/harmbench__llama_guard_3.json'
            original = read_json(path)
            mutations = [original + original[:1], original[1:]]
            changed = deepcopy(original)
            changed[0]['prompt']['text'] = 'mismatch'
            mutations.append(changed)
            changed = deepcopy(original)
            changed[0]['verdicts']['llama_guard_3']['details']['parse_error'] = True
            mutations.append(changed)
            for records in mutations:
                write_json(path, records)
                with self.assertRaises(ValueError):
                    analyze.load_verdicts(result, expected)

    def test_merge_seed_isolation_and_real_peft_pruning(self):
        import torch
        from peft.utils.merge_utils import dare_linear, dare_ties

        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        for merge in (dare_linear, dare_ties):
            def merged(seed):
                with isolated_merge_seed(seed):
                    random.random()
                    np.random.rand()
                    return merge([torch.ones(4096), torch.ones(4096) * 2], torch.tensor([0.25, 0.25]), 0.75)
            self.assertTrue(torch.equal(merged(42), merged(42)))
            self.assertFalse(torch.equal(merged(42), merged(43)))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_complete_analysis_and_provenance_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            synthetic_invocation(directory)
            report = analyze.analyze(directory)
            self.assertEqual(len(report['runs']), 26)
            self.assertEqual(len(report['dare_aggregates']), 4)
            self.assertTrue((directory / 'analysis/report.md').is_file())
            self.assertEqual(report, analyze.analyze(directory))
            meta = directory / 'results/base/experiment4_provenance.json'
            info = read_json(meta)
            info['generation_seed'] = 99
            write_json(meta, info)
            with self.assertRaises(ValueError):
                analyze.analyze(directory)

    def test_launcher_concurrency_download_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures(root / 'fixtures')
            mock = root / 'mock-modal'
            mock.write_text(f'#!{sys.executable}\n' + '''
import json, os, shutil, sys, time
from pathlib import Path
root = Path(os.environ['MOCK_ROOT'])
args = sys.argv[1:]
if args[0] == 'run':
    config = Path(args[args.index('--config') + 1])
    label = config.stem
    invocation = args[args.index('--invocation-id') + 1]
    receipt = Path(args[args.index('--receipt') + 1])
    barrier = root / invocation
    barrier.mkdir(exist_ok=True)
    (barrier / label).touch()
    deadline = time.monotonic() + 10
    while len(list(barrier.iterdir())) < 13:
        if time.monotonic() > deadline:
            sys.exit(9)
        time.sleep(0.02)
    print('mock stdout', flush=True)
    print('mock stderr', file=sys.stderr, flush=True)
    if os.environ.get('MOCK_FAIL') and label == 'linear':
        sys.exit(7)
    info = json.loads((root / 'fixtures' / label / 'experiment4_provenance.json').read_text())
    info['invocation_id'] = invocation
    info['remote_path'] = f'experiment4/{invocation}/{label}/timestamp'
    receipt.write_text(json.dumps(info))
else:
    assert args[:3] == ['volume', 'get', 'safety-bench-results']
    _, invocation, label, _ = args[3].split('/')
    destination = Path(args[4]) / 'timestamp'
    shutil.copytree(root / 'fixtures' / label, destination)
    meta = destination / 'experiment4_provenance.json'
    info = json.loads(meta.read_text())
    info['invocation_id'] = invocation
    meta.write_text(json.dumps(info))
''')
            mock.chmod(0o755)
            previous = os.environ.get('MOCK_ROOT')
            os.environ['MOCK_ROOT'] = str(root)
            try:
                with redirect_stdout(io.StringIO()):
                    directory = launch(str(mock), root / 'runs')
                self.assertTrue((directory / 'analysis/report.json').exists())
                self.assertEqual(len(read_json(directory / 'status.json')), 13)
                for path in (directory / 'logs').glob('*.log'):
                    if not path.name.endswith('.download.log'):
                        self.assertIn('mock stderr', path.read_text())
                        self.assertIn('mock stdout', path.read_text())
                os.environ['MOCK_FAIL'] = '1'
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    launch(str(mock), root / 'runs')
                failed = [p for p in (root / 'runs').iterdir() if p.resolve() != directory][0]
                self.assertFalse((failed / 'analysis').exists())
                self.assertFalse(list((failed / 'logs').glob('*.download.log')))
                self.assertEqual((failed / 'logs/linear.exit_code').read_text().strip(), '7')
            finally:
                os.environ.pop('MOCK_FAIL', None)
                if previous is None:
                    os.environ.pop('MOCK_ROOT', None)
                else:
                    os.environ['MOCK_ROOT'] = previous


if __name__ == '__main__':
    unittest.main()
