"""Tiny CPU tensors and saved JSON only; never download models or adapters."""
from __future__ import annotations

import csv
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

torch.set_num_threads(1)

from ordering_experiment.compose import evaluate, key
from pairwise_experiment.common import read_json, write_json
from mechanistic_analysis.behavior import parse_expression, transition_plan, paired_change, prepare
from mechanistic_analysis.geometry import MatrixStates, endpoint_statistics, pair_statistics, ratios, square_sum
from mechanistic_analysis.analyze import add_stats, analyze, read_csv

ROOT = Path(__file__).resolve().parents[1]


class TensorChecks(unittest.TestCase):
    def test_large_updates_exclude_small_conflicts(self):
        x = torch.tensor([100., 90, 80, 70, 60, 5, 4, 3, 2, 1])
        y = torch.tensor([100., 90, 80, 70, 60, -5, -4, -3, -2, -1])
        stats = pair_statistics(x, y, 100., [.1, .2, .5])
        values = ratios(stats, [.1, .2, .5])
        self.assertEqual(values["raw_sign_conflict"], .5)
        for tag in ("top10", "top20", "top50"):
            self.assertEqual(values[tag + "_sign_conflict"], 0)
            self.assertEqual(values[tag + "_energy_weighted_conflict"], 0)
            self.assertEqual(values[tag + "_jaccard"], 1)
        self.assertAlmostEqual(values["incoming_relative_to_base"], y.norm().item() / 10, places=5)

    def test_zero_and_disjoint_support(self):
        x, y = torch.arange(1., 11.), torch.arange(10., 0., -1)
        values = ratios(pair_statistics(x, y, 1., [.1, .2, .5]), [.1, .2, .5])
        self.assertIsNone(values["top10_sign_conflict"])
        self.assertEqual(values["top10_jaccard"], 0)
        values = ratios(pair_statistics(torch.zeros_like(x), y, 1., [.1, .2, .5]), [.1, .2, .5])
        self.assertIsNone(values["cosine"])
        self.assertIsNone(values["incoming_over_composite_norm"])
        self.assertIsNone(values["top50_sign_conflict"])

    def test_aggregation_uses_counts_and_norms(self):
        x, y = torch.tensor([1., 2., 3., 4.]), torch.tensor([-1., 2., -3., 4.])
        stats = pair_statistics(x, y, 2., [.1, .2, .5])
        accum = {}
        add_stats(accum, "block", stats)
        add_stats(accum, "block", stats)
        values = ratios(accum["block"], [.1, .2, .5])
        self.assertAlmostEqual(values["composite_norm"], math.sqrt(2 * square_sum(x)))
        self.assertAlmostEqual(values["cosine"], torch.nn.functional.cosine_similarity(x, y, dim=0).item(), places=6)
        self.assertEqual(values["raw_sign_conflict"], .5)

    def test_endpoint_difference_and_bf16_rounding(self):
        low, high = torch.tensor([.001, .002]), torch.tensor([.0011, .0021])
        raw = endpoint_statistics(low, high, 2., [.1, .2, .5])
        self.assertGreater(raw["difference_sq"], 0)
        base = torch.ones(2, dtype=torch.bfloat16)
        effective_low = (base.float() + low).bfloat16().float() - base.float()
        effective_high = (base.float() + high).bfloat16().float() - base.float()
        rounded = endpoint_statistics(effective_low, effective_high, 2., [.1, .2, .5])
        self.assertNotEqual(raw["difference_sq"], rounded["difference_sq"])

    def test_bootstrap_integer_change(self):
        row = paired_change([False, True], [True, False], 1000, 42)
        self.assertEqual(row["change_pp"], 0)
        self.assertEqual((row["ci95_low_pp"], row["ci95_high_pp"]), (-100., 100.))

    def test_expression_parser(self):
        for text in ("a1", "sum(a1,a2,a3)", "ties(a1,ties(a2,a3))"):
            self.assertEqual(key(parse_expression(text)), text)
        with self.assertRaises(ValueError):
            parse_expression("ties(a1)")


class SourceAndPipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ROOT / "mechanistic_analysis/config.yaml").read_text())
        if not (ROOT / cls.cfg["source_run"]).exists():
            raise unittest.SkipTest("Completed ordering results are unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temporary.name)
        cls.cfg["bootstrap_samples"] = 200
        prepare(cls.run_dir, cls.cfg)
        cls.plan = read_json(cls.run_dir / "source_plan.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_transition_deduplication_and_endpoint_gap(self):
        transitions, occurrences = transition_plan(self.plan)
        self.assertEqual(sum(t["method"] == "linear" for t in transitions.values()), 32)
        self.assertEqual(sum(t["method"] == "ties" for t in transitions.values()), 40)
        self.assertEqual(len(occurrences), 192)
        rows = read_csv(self.run_dir / "ties_endpoint_asr.csv")
        self.assertEqual([float(r["change_pp"]) for r in rows if r["dataset"] == "harmbench"], [5., 5.])

    def test_prepare_resume_and_mutation_detection(self):
        before = (self.run_dir / "prepared.json").stat().st_mtime_ns
        prepare(self.run_dir, self.cfg)
        self.assertEqual(before, (self.run_dir / "prepared.json").stat().st_mtime_ns)
        # Isolate failure detection from real source files.
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "changed.json"
            source.write_text("changed")
            write_json(directory / "prepared.json", {"input_sha256": {str(source): "wrong hash"}})
            with self.assertRaises(ValueError):
                prepare(directory, self.cfg)

    def test_all_expressions_match_original_operator(self):
        deltas = {"a1": torch.tensor([[4., -3], [2, 1]]), "a2": torch.tensor([[-1., 5], [3, -4]]),
                  "a3": torch.tensor([[2., 1], [-5, 3]]), "a4": torch.tensor([[-3., -2], [1, 6]])}
        class Bank:
            def delta(self, name, module):
                return deltas[name]
        store = MatrixStates(deltas, .5)
        for text in self.plan["models"]:
            if text != "base":
                self.assertTrue(torch.equal(store.get(text), evaluate(parse_expression(text), "tiny", Bank(), .5)))
        linear_ends = [store.get(path[-1]) for path in self.plan["paths"]["linear"].values()]
        self.assertTrue(all(torch.equal(x, linear_ends[0]) for x in linear_ends))
        low = self.plan["paths"]["ties"][self.cfg["lower_asr_order"]][-1]
        high = self.plan["paths"]["ties"][self.cfg["higher_asr_order"]][-1]
        self.assertFalse(torch.equal(store.get(low), store.get(high)))

    def test_complete_pipeline_on_two_tiny_modules(self):
        self.fill_tiny_cache()
        with patch("mechanistic_analysis.analyze.plots"):
            analyze(self.run_dir, self.cfg)
        self.assertEqual(len(read_csv(self.run_dir / "transition_effects_global.csv")), 72 * 6)
        self.assertEqual(len(read_csv(self.run_dir / "selected_path_interactions.csv")), 16 * 6)
        control = read_json(self.run_dir / "linear_control.json")
        self.assertTrue(control["canonical_fp32_endpoint_equal"])
        self.assertEqual(control["max_absolute_difference"], 0)
        concentration = read_csv(self.run_dir / "ties_difference_concentration.csv")
        for representation in ("merge_update_fp32", "checkpoint_effective_bfloat16"):
            final = next(r for r in concentration if r["representation"] == representation and r["level"] == "module" and r["top_n"] == "2")
            self.assertAlmostEqual(float(final["difference_energy_fraction"]), 1)
        correlations = read_csv(self.run_dir / "exploratory_correlations.csv")
        counts = {r["method"]: int(r["n_transitions"]) for r in correlations
                  if r["level"] == "global" and r["step"] == "all_nonbase" and r["feature"] == "incoming_relative_to_base"}
        self.assertEqual(counts, {"linear": 28, "ties": 36})

    def test_geometry_worker_and_module_resume_with_mock_model(self):
        from types import SimpleNamespace
        from mechanistic_analysis.geometry import measure
        prepared = read_json(self.run_dir / "prepared.json")
        modules = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
        deltas = {"a1": torch.tensor([[4., -3], [2, 1]]), "a2": torch.tensor([[-1., 5], [3, -4]]),
                  "a3": torch.tensor([[2., 1], [-5, 3]]), "a4": torch.tensor([[-3., -2], [1, 6]])}
        class TinyBank:
            def __init__(self, adapters):
                self.shapes = {module: (2, 2) for module in modules}
                self.factors = dict.fromkeys(deltas)
                self.metadata = prepared["expected_adapters"]

            def delta(self, name, module):
                return deltas[name] * (modules.index(module) + 1)

        model = SimpleNamespace(config=SimpleNamespace(_commit_hash="tiny-test-revision"),
            named_parameters=lambda: [(module + ".weight", torch.ones(2, 2, dtype=torch.bfloat16)) for module in modules])
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            for filename in ("prepared.json", "transitions.json", "source_plan.json"):
                shutil.copyfile(self.run_dir / filename, run_dir / filename)
            with patch("mechanistic_analysis.geometry.AdapterBank", TinyBank), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load:
                measure(run_dir, self.cfg)
                measure(run_dir, self.cfg)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(read_json(run_dir / "geometry_complete.json")["modules"], 2)
            self.assertEqual(read_json(run_dir / "base_metadata.json")["revision"], "tiny-test-revision")
            for module in modules:
                cache = read_json(run_dir / "module_cache" / f"{module}.json")
                self.assertEqual(len(cache["transitions"]), 72)
                self.assertTrue(cache["linear_control"]["equal"])

    @classmethod
    def fill_tiny_cache(cls):
        transitions = read_json(cls.run_dir / "transitions.json")
        prepared = read_json(cls.run_dir / "prepared.json")
        modules = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
        write_json(cls.run_dir / "modules.json", modules)
        for index, module in enumerate(modules):
            scale = index + 1
            deltas = {"a1": torch.tensor([[4., -3], [2, 1]]) * scale, "a2": torch.tensor([[-1., 5], [3, -4]]) * scale,
                      "a3": torch.tensor([[2., 1], [-5, 3]]) * scale, "a4": torch.tensor([[-3., -2], [1, 6]]) * scale}
            store = MatrixStates(deltas, .5)
            rows = []
            for identifier, t in transitions.items():
                x, y, after = store.get(t["before"]), store.get(t["incoming"]), store.get(t["after"])
                stats = pair_statistics(x, y, 4., cls.cfg["top_fractions"])
                stats.update({"after_sq": square_sum(after), "integration_step_sq": square_sum(after - x),
                              "deviation_from_sum_sq": square_sum(after - x - y)})
                rows.append({"transition_id": identifier, "stats": stats})
            low, high = store.get(prepared["lower_endpoint"]), store.get(prepared["higher_endpoint"])
            endpoints = [{"representation": representation, "stats": endpoint_statistics(low, high, 4., cls.cfg["top_fractions"])}
                         for representation in ("merge_update_fp32", "checkpoint_effective_bfloat16")]
            write_json(cls.run_dir / "module_cache" / f"{module}.json", {"module": module, "transitions": rows,
                       "endpoints": endpoints, "linear_control": {"n_paths": 24, "equal": True, "max_absolute_difference": 0.}})


if __name__ == "__main__":
    unittest.main()
