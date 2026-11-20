"""Tiny CPU checks only: python -m unittest pairwise_experiment.test_experiment."""
from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

from ordering_experiment.compose import ties
from pairwise_experiment.analyze import analyze, bootstrap_effect, composition_effect, ranks, spearman
from pairwise_experiment.common import (DETECTORS, METHODS, build_states, load_verdicts,
    multiplier, read_json, retry_batches, save_verdicts, validate_generations, write_csv, write_json)
from pairwise_experiment.geometry import AdapterBank, derived, sufficient_stats, top_mask, validate_adapter_config
from pairwise_experiment.run import (ROOT, check_source_config, effective_config, prepare,
                                      science_config, source_key, source_state_dir)


class MathTests(unittest.TestCase):
    def test_plan(self):
        states = build_states(["a1", "a2", "a3", "a4"])
        self.assertEqual(len(states), 35)
        self.assertEqual(sum(source_key(k, v) is None for k, v in states.items()), 18)
        for method in METHODS:
            self.assertEqual(sum(s["method"] == method for s in states.values()), 6)

    def test_coefficients(self):
        self.assertEqual(multiplier("linear_sum", [3, 4], [6, 8]), 1)
        self.assertEqual(multiplier("linear_average", [3, 4], [6, 8]), .5)
        self.assertAlmostEqual(multiplier("coefficient_normalized", [3, 4], [6, 8]), .2)
        self.assertAlmostEqual(multiplier("frobenius_normalized", [3, 4], [6, 8]), .1)
        with self.assertRaises(ValueError):
            multiplier("frobenius_normalized", [1, 1], [0, 0])

    def test_geometry(self):
        x = torch.tensor([1., -2, 0, 4])
        y = torch.tensor([-1., -2, 3, 0])
        d = derived(sufficient_stats(x, y, torch.ones(4), .5))
        self.assertAlmostEqual(d["cosine"], 3 / np.sqrt(21 * 14))
        self.assertEqual(d["sign_conflict"], .5)
        self.assertAlmostEqual(d["top_jaccard"], 1 / 3)
        self.assertAlmostEqual(d["relative_to_base_i"], np.sqrt(21) / 2)
        empty = derived(sufficient_stats(x, torch.zeros(4), torch.ones(4), .5))
        self.assertIsNone(empty["cosine"])
        self.assertIsNone(empty["norm_balance"])
        self.assertIsNone(empty["sign_conflict"])
        self.assertEqual(top_mask(torch.ones(4), .5).sum().item(), 4)
        self.assertEqual(top_mask(torch.zeros(4), .5).sum().item(), 0)

    def test_ties_symmetry_and_signs(self):
        x, y = torch.tensor([2., -2, 1, 0]), torch.tensor([-3., -4, 1, 0])
        self.assertTrue(torch.equal(ties(x, y, .5), ties(y, x, .5)))
        self.assertTrue(torch.equal(ties(x, y, .5), torch.tensor([-3., -3, 0, 0])))
        self.assertTrue(torch.equal(ties(x, -x, 1), torch.zeros_like(x)))

    def test_full_strength_effect_and_bootstrap_max(self):
        pair, x, y = [1, 1], [1, 0], [0, 1]
        self.assertEqual(composition_effect(pair, x, y), 50)
        self.assertEqual(bootstrap_effect(pair, x, y, samples=2000), [0, 50])
        self.assertEqual(bootstrap_effect([0, 0], [1, 1], [0, 0], 100), [-100, -100])
        with self.assertRaises(ValueError):
            bootstrap_effect([1], [1, 0], [0])

    def test_ranks(self):
        np.testing.assert_equal(ranks([3, 1, 1, 4]), [3, 1.5, 1.5, 4])
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_oom_retries_order_and_single_failure(self):
        class OOM(Exception):
            pass
        sizes, messages = [], []
        def call(chunk):
            sizes.append(len(chunk))
            if len(chunk) > 2:
                raise OOM()
            return [x * 2 for x in chunk]
        result = retry_batches(list(range(9)), 8, call, OOM, lambda: None, messages.append)
        self.assertEqual(result, [x * 2 for x in range(9)])
        self.assertIn(8, sizes)
        with self.assertRaises(OOM):
            retry_batches([1], 1, lambda chunk: (_ for _ in ()).throw(OOM()), OOM, lambda: None, messages.append)

    def test_adapter_metadata_and_scaled_norm(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_json(directory / "adapter_config.json", {"r": 2, "lora_alpha": 4, "peft_type": "LORA"})
            save_file({"base_model.model.model.layers.0.q_proj.lora_A.weight": torch.eye(2),
                       "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.diag(torch.tensor([3., 4.]))},
                      str(directory / "adapter_model.safetensors"))
            bank = AdapterBank([{"name": "a1", "path": temp}], device="cpu")
            delta = bank.delta("a1", "model.layers.0.q_proj")
            self.assertEqual(delta.norm().item(), 10.)  # raw norm=5; alpha/r=2 applied FIRST
            bank.shapes["untouched"] = (2, 2)
            self.assertEqual(bank.delta("a1", "untouched").norm().item(), 0)
        for key in ("use_dora", "fan_in_fan_out", "rank_pattern", "alpha_pattern"):
            with self.assertRaises(ValueError):
                validate_adapter_config({"r": 2, "lora_alpha": 4, key: True})


class CacheAndIntegrationTests(unittest.TestCase):
    def test_stale_cache_and_prompt_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            gen, verdict = Path(temp) / "gen.json", Path(temp) / "verdict.json"
            records = [{"prompt": {"id": "p"}, "response": "hello"}]
            write_json(gen, records)
            save_verdicts(verdict, gen, [{"flagged": True}])
            self.assertTrue(load_verdicts(verdict, gen)[0]["flagged"])
            records[0]["response"] = "changed"
            write_json(gen, records)
            with self.assertRaises(ValueError):
                load_verdicts(verdict, gen)
            with self.assertRaises(ValueError):
                validate_generations(records, [{"id": "other"}])

    def test_profiles_and_resume_contract(self):
        a = effective_config(ROOT / "pairwise_experiment/config.yaml", "rtx5000")
        b = effective_config(ROOT / "pairwise_experiment/config.yaml", "h100")
        self.assertEqual(a["detectors"]["harm_cls"]["batch_size"], 2)
        self.assertEqual(b["detectors"]["harm_cls"]["batch_size"], 16)
        self.assertEqual(science_config(a), science_config(b))
        b["generation"]["do_sample"] = True
        self.assertNotEqual(science_config(a), science_config(b))

    def test_actual_source_import_and_resume(self):
        raw = effective_config(ROOT / "pairwise_experiment/config.yaml", "rtx5000")
        source = ROOT / raw["pairwise"]["source_run"]
        if not source.exists():
            self.skipTest("Completed ordering results not available")
        source_config = yaml.safe_load((source / "config_used.yaml").read_text())
        check_source_config(raw, source_config)
        invalid = copy.deepcopy(raw)
        invalid["generation"]["max_new_tokens"] = 17
        with self.assertRaises(ValueError):
            check_source_config(invalid, source_config)
        states = build_states([a["name"] for a in raw["model"]["adapters"]])
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            prepare(run_dir, raw, states)
            before = (run_dir / "imports.json").stat().st_mtime_ns
            prepare(run_dir, raw, states)
            self.assertEqual(before, (run_dir / "imports.json").stat().st_mtime_ns)
            self.assertEqual(len(read_json(run_dir / "imports.json")["states"]), 17)
            # Reproduce all 102 imported rates exactly from saved per-model summaries.
            for state_id, key in read_json(run_dir / "imports.json")["states"].items():
                summary = read_json(source_state_dir(source, key) / "summary.json")
                for dataset, detectors in DETECTORS.items():
                    gen = run_dir / "states" / state_id / "generations" / f"{dataset}.json"
                    for detector in detectors:
                        verdicts = load_verdicts(gen.parent.parent / "verdicts" / f"{dataset}__{detector}.json", gen)
                        metric = "frr" if dataset == "alpaca" else "asr"
                        self.assertEqual(sum(v["flagged"] for v in verdicts) / len(verdicts), summary[dataset][metric][detector][metric])

    def test_complete_analysis_on_tiny_synthetic_records(self):
        raw = effective_config(ROOT / "pairwise_experiment/config.yaml", "rtx5000")
        raw["pairwise"]["bootstrap_samples"] = 100
        names = [a["name"] for a in raw["model"]["adapters"]]
        states = build_states(names)
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            prompts = {d: [{"id": str(i)} for i in range(6)] for d in DETECTORS}
            write_json(run_dir / "prompts.json", prompts)
            write_json(run_dir / "imports.json", {"states": {k: "source" for k, s in states.items() if source_key(k, s)}})
            for index, state_id in enumerate(states):
                for dataset, detectors in DETECTORS.items():
                    gen = run_dir / "states" / state_id / "generations" / f"{dataset}.json"
                    write_json(gen, [{"prompt": p, "response": "synthetic"} for p in prompts[dataset]])
                    for detector in detectors:
                        save_verdicts(gen.parent.parent / "verdicts" / f"{dataset}__{detector}.json", gen,
                                      [{"flagged": i < index % 7} for i in range(6)])
            write_csv(run_dir / "geometry.csv", [{"adapter_i": a, "adapter_j": b, "level": "global",
                "cosine": i / 6, "norm_balance": (i + 1) / 6, "sign_conflict": i / 10, "top_jaccard": i / 8}
                for i, (a, b) in enumerate(combinations(names, 2))])
            with patch("pairwise_experiment.analyze.plot"):
                analyze(run_dir, raw, states)
            def rows(name):
                with open(run_dir / name) as f:
                    return list(csv.DictReader(f))
            self.assertEqual(len(rows("rates.csv")), 35 * 6)
            self.assertEqual(len(rows("composition_effects.csv")), 30 * 6)
            self.assertEqual(len(rows("correlations.csv")), 5 * 2 * 2 * 4)
            self.assertTrue(all(r["n_pairs"] == "6" for r in rows("correlations.csv")))

    def test_judge_primitive_retry_and_resume(self):
        from bench.detectors.base import Verdict
        from pairwise_experiment.run import judge
        from types import SimpleNamespace

        class TinyJudge:
            name, metric = "harm_cls", "asr"
            config = SimpleNamespace(batch_size=4)
            calls = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def _classify_batch(self, items):
                self.calls.append(len(items))
                if len(items) > 1:
                    raise torch.cuda.OutOfMemoryError("synthetic OOM")
                return [Verdict(flagged=g.prompt.id == "1", raw_output="synthetic") for g in items]

            def judge(self, items):
                return self._classify_batch(items)

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            canonical = {}
            for dataset in DETECTORS:
                canonical[dataset] = [{"id": str(i), "dataset": dataset, "text": "x", "behavior": "x"} for i in range(3)]
                write_json(run_dir / "states/new/generations" / f"{dataset}.json",
                           [{"prompt": p, "response": "x"} for p in canonical[dataset]])
            write_json(run_dir / "prompts.json", canonical)
            detector = TinyJudge()
            config = SimpleNamespace(detectors=None)
            with patch("bench.detectors.build_detectors", return_value=[detector]):
                judge(run_dir, {}, {"new": {}}, config)
                count = len(detector.calls)
                judge(run_dir, {}, {"new": {}}, config)
                self.assertEqual(len(detector.calls), count)
            gen = run_dir / "states/new/generations/harmbench.json"
            records = load_verdicts(run_dir / "states/new/verdicts/harmbench__harm_cls.json", gen)
            self.assertEqual([v["flagged"] for v in records], [False, True, False])


if __name__ == "__main__":
    unittest.main()
