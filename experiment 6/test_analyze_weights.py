"""CPU-only tests with synthetic adapters; no Hub or Modal access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

SCRIPT = Path(__file__).with_name('analyze_weights.py')
spec = importlib.util.spec_from_file_location('weight_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def factors(a, b, scale=1):
    a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    return {'A': a, 'B': b, 'scale': scale, 'shape': (b.shape[0], a.shape[1])}


class WeightAnalysisTests(unittest.TestCase):
    def test_metrics_against_dense(self):
        updates = {
            'a': factors([[1, 0, -1], [0, 2, 1]], [[1, 2], [-1, 0]], 0.5),
            'b': factors([[1, -1, 0]], [[2], [-3]], 2),
        }
        dense = {k: v['scale'] * (v['B'] @ v['A']) for k, v in updates.items()}
        magnitudes, pairs = analysis.module_metrics(updates, chunk_rows=1)
        pair = pairs['a', 'b']
        a, b = dense['a'].double(), dense['b'].double()
        self.assertAlmostEqual(magnitudes['a']['frobenius_norm'], torch.linalg.norm(a).item())
        self.assertAlmostEqual(pair['cosine_similarity'], ((a * b).sum() / (a.norm() * b.norm())).item())
        self.assertAlmostEqual(pair['sign_conflict_rate'], (a.sign() != b.sign()).double().mean().item())
        self.assertEqual(analysis.module_metrics(updates, 100), (magnitudes, pairs))

    def test_identical_opposite_zero_and_shape_mismatch(self):
        a = factors([[1, -2]], [[1], [2]])
        opposite = dict(a, scale=-1)
        zero = dict(a, scale=0)
        mismatch = factors([[1, 2, 3]], [[1]])
        _, pairs = analysis.module_metrics({'a': a, 'same': a, 'opposite': opposite, 'zero': zero, 'other': mismatch}, 1)
        self.assertAlmostEqual(pairs['a', 'same']['cosine_similarity'], 1)
        self.assertEqual(pairs['a', 'same']['sign_conflict_rate'], 0)
        self.assertAlmostEqual(pairs['a', 'opposite']['cosine_similarity'], -1)
        self.assertEqual(pairs['a', 'opposite']['sign_conflict_rate'], 1)
        self.assertIsNone(pairs['a', 'zero']['cosine_similarity'])
        self.assertIsNone(pairs['a', 'zero']['nonzero_sign_conflict_rate'])
        self.assertNotIn(('a', 'other'), pairs)

    def test_end_to_end_local_checkpoints(self):
        from PIL import Image
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, str(SCRIPT), '--output-root', str(root / 'outputs'), '--chunk-rows', '1']
            for index in range(4):
                adapter = root / f'a{index + 1}'
                adapter.mkdir()
                config = dict(peft_type='LORA', r=1, lora_alpha=2, base_model_name_or_path='synthetic',
                              use_rslora=index == 1, rank_pattern={'q_proj': 2}, alpha_pattern={'q_proj': 4})
                (adapter / 'adapter_config.json').write_text(json.dumps(config))
                state = {
                    'base_model.model.layers.0.q_proj.lora_A.weight': torch.tensor([[1., 2.], [0., 1.]]),
                    'base_model.model.layers.0.q_proj.lora_B.weight': torch.eye(2) * (index - 1),
                }
                if index != 3:
                    state['base_model.model.layers.1.v_proj.lora_A.weight'] = torch.ones(1, 2)
                    state['base_model.model.layers.1.v_proj.lora_B.weight'] = torch.ones(2, 1)
                save_file(state, str(adapter / 'adapter_model.safetensors'))
                modules, _ = analysis.load_adapter(str(adapter))
                expected = 4 / (2 ** 0.5 if index == 1 else 2)
                self.assertAlmostEqual(modules['base_model.model.layers.0.q_proj']['scale'], expected)
                command += ['--adapter', f'a{index + 1}={adapter}']
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = next((root / 'outputs').iterdir())
            for name in ('update_magnitudes.csv', 'pairwise_metrics.csv', 'metadata.json', 'report.md'):
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertIn('missing_module', (output / 'pairwise_metrics.csv').read_text())
            for name in ('magnitude_heatmap.png', 'cosine_heatmap.png', 'sign_conflict_heatmap.png'):
                with Image.open(output / name) as image:
                    image.verify()


if __name__ == '__main__':
    unittest.main()
