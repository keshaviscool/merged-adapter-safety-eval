"""Run from repo root: python pairwise_experiment/run.py --profile rtx5000."""
from __future__ import annotations

import argparse
import copy
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pairwise_experiment.common import (DETECTORS, build_states, digest, load_verdicts,
    multiplier, read_json, retry_batches, save_verdicts, validate_generations,
    validate_verdicts, write_json)

LOG = logging.getLogger("pairwise")


def effective_config(path, profile):
    raw = yaml.safe_load(Path(path).read_text())
    profile = profile or raw.get("selected_profile", "rtx5000")
    if profile not in raw["profiles"]:
        raise ValueError(f"Unknown GPU profile: {profile}")
    overrides = raw["profiles"][profile]
    raw["generation"].update(overrides["generation"])
    for name, values in overrides["detectors"].items():
        raw["detectors"][name].update(values)
    raw["selected_profile"] = profile
    if raw["generation"].get("do_sample") is not False:
        raise ValueError("This experiment requires do_sample: false")
    for field in ("ties_density", "overlap_density"):
        if not 0 < raw["pairwise"][field] <= 1:
            raise ValueError(f"Invalid {field}")
    if raw["pairwise"]["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be positive")
    for section in [raw["generation"], *raw["detectors"].values()]:
        if "batch_size" in section and section["batch_size"] < 1:
            raise ValueError("batch_size must be positive")
    build_states([a["name"] for a in raw["model"]["adapters"]])
    if any(a.get("weight", 1.0) != 1.0 for a in raw["model"]["adapters"]):
        raise ValueError("Per-adapter weights must be 1; weights are defined by each merge variant")
    return raw


def science_config(raw):
    """Permit batch/profile changes on resume; reject changes to the experiment."""
    raw = copy.deepcopy(raw)
    for key in ("profiles", "selected_profile", "output"):
        raw.pop(key, None)
    raw["generation"].pop("batch_size", None)
    for detector in raw["detectors"].values():
        detector.pop("batch_size", None)
    return raw


def check_source_config(raw, source):
    for field in ("base_model", "dtype", "trust_remote_code"):
        if raw["model"].get(field) != source["model"].get(field):
            raise ValueError(f"Source model {field} differs")
    current = [(a["name"], a["path"]) for a in raw["model"]["adapters"]]
    previous = [(a["name"], a["path"]) for a in source["model"]["adapters"]]
    if current != previous:
        raise ValueError("Source adapter names/paths/order differ")
    # Cache paths and GPU batches may differ; generation semantics must match.
    for field in ("max_new_tokens", "do_sample", "repetition_penalty", "seed", "system_prompt"):
        if raw["generation"].get(field) != source["generation"].get(field):
            raise ValueError(f"Source generation {field} differs")
    for dataset in DETECTORS:
        if raw["datasets"][dataset] != source["datasets"][dataset]:
            raise ValueError(f"Source dataset configuration differs: {dataset}")
        if not raw["datasets"][dataset]["enabled"]:
            raise ValueError("All three datasets must be enabled")
    for name, cfg in raw["detectors"].items():
        relevant = lambda d: {k: v for k, v in d.items() if k not in ("batch_size", "device_map")}
        if relevant(cfg) != relevant(source["detectors"][name]) or not cfg["enabled"]:
            raise ValueError(f"Source classifier configuration differs: {name}")
    if raw["pairwise"]["ties_density"] != source["ordering"]["ties_density"]:
        raise ValueError("Source TIES density differs")


def source_key(state_id, state):
    method, names = state["method"], state["adapters"]
    if method in ("base", "single"):
        return state_id
    if method == "linear_sum":
        return "sum(" + ",".join(names) + ")"
    if method == "ties":
        return "ties(" + ",".join(sorted(names)) + ")"
    return None


def source_state_dir(source, key):
    return source / "states" / key.replace("(", "_").replace(")", "").replace(",", "-")


def prepare(run_dir, raw, states):
    """Copy validated source records; never change the completed ordering run."""
    if (run_dir / "imports.json").exists():
        validate_imports(run_dir)
        return
    source = Path(raw["pairwise"]["source_run"]).resolve()
    previous = yaml.safe_load((source / "config_used.yaml").read_text())
    check_source_config(raw, previous)
    plan = read_json(source / "plan.json")
    canonical = {}
    for dataset in DETECTORS:
        records = read_json(source / "states" / "base" / "generations" / f"{dataset}.json")
        canonical[dataset] = validate_generations(records)
    imports = {"source_run": str(source), "source_config": previous,
               "original_adapter_revisions": "Not recorded in source; repository IDs alone cannot prove identical revisions.",
               "files": {}, "states": {}}
    for state_id, state in states.items():
        key = source_key(state_id, state)
        if key is None:
            continue
        if key not in plan["models"]:
            raise ValueError(f"Source merge expression missing: {key}")
        src, dst = source_state_dir(source, key), run_dir / "states" / state_id
        imports["states"][state_id] = key
        for dataset, detectors in DETECTORS.items():
            generation_file = src / "generations" / f"{dataset}.json"
            records = read_json(generation_file)
            validate_generations(records, canonical[dataset])
            target = dst / "generations" / f"{dataset}.json"
            write_json(target, records)
            imports["files"][str(target.relative_to(run_dir))] = {
                "source": str(generation_file), "source_sha256": digest(generation_file), "sha256": digest(target)}
            for detector in detectors:
                verdict_file = src / "verdicts" / f"{dataset}__{detector}.json"
                verdicts = read_json(verdict_file)
                validate_verdicts(verdicts, records)
                output = dst / "verdicts" / verdict_file.name
                save_verdicts(output, target, verdicts)
                imports["files"][str(output.relative_to(run_dir))] = {
                    "source": str(verdict_file), "source_sha256": digest(verdict_file), "sha256": digest(output)}
        LOG.info("Imported %s", state_id)
    for filename in ("config_used.yaml", "plan.json"):
        imports.setdefault("source_metadata_hashes", {})[filename] = digest(source / filename)
    write_json(run_dir / "prompts.json", canonical)
    write_json(run_dir / "imports.json", imports)


def validate_imports(run_dir):
    manifest = read_json(run_dir / "imports.json")
    for relative, metadata in manifest["files"].items():
        if digest(run_dir / relative) != metadata["sha256"]:
            raise ValueError(f"Imported file changed: {relative}")
    prompts = read_json(run_dir / "prompts.json")
    for dataset in DETECTORS:
        validate_generations(read_json(run_dir / "states" / "base" / "generations" / f"{dataset}.json"), prompts[dataset])


def record_environment(run_dir, phase, raw):
    import importlib.metadata
    data = {"phase": phase, "time_utc": datetime.now(timezone.utc).isoformat(),
            "profile": raw["selected_profile"], "python": sys.version,
            "platform": platform.platform(), "packages": {},
            "generation": raw["generation"], "detectors": raw["detectors"]}
    for package in ("torch", "transformers", "peft", "accelerate", "numpy"):
        try:
            data["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if phase in ("geometry", "generate", "judge"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Run GPU phases on your NVIDIA server; no CPU inference fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        data["cuda"] = torch.version.cuda
        data["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    write_json(run_dir / "environment" / f"{stamp}_{phase}.json", data)


def generate(run_dir, raw, states, config):
    import torch
    from bench.datasets import Prompt
    from bench.generation import ResponseGenerator
    from bench.model_builder import build_model
    from pairwise_experiment.geometry import AdapterBank

    norms = read_json(run_dir / "adapter_norms.json")
    canonical = read_json(run_dir / "prompts.json")
    todo = []
    for state_id, state in states.items():
        missing = []
        for dataset, prompts in canonical.items():
            target = run_dir / "states" / state_id / "generations" / f"{dataset}.json"
            if target.exists():
                validate_generations(read_json(target), prompts)
            else:
                missing.append(dataset)
        if missing:
            if source_key(state_id, state) is not None:
                raise ValueError(f"Missing imported generations for {state_id}")
            todo.append((state_id, state, missing))
    if not todo:
        LOG.info("All generations already complete")
        return
    bank = AdapterBank(raw["model"]["adapters"])
    for name in bank.metadata:
        for field in ("config_sha256", "weights_sha256"):
            if bank.metadata[name][field] != norms[name][field]:
                raise ValueError("Adapter files changed since geometry measurement")
    model, tokenizer = build_model(replace(config.model, adapters=[]), config.merge)
    params = dict(model.named_parameters())
    original = {module: params[module + ".weight"].detach().cpu().clone() for module in bank.shapes}
    generator = ResponseGenerator(model, tokenizer, config.generation)
    # Wrap the primitive, rather than nesting the older generator's OOM handler.
    def generate_batch(formatted):
        return retry_batches(formatted, len(formatted), generator._generate_batch_once,
                             torch.cuda.OutOfMemoryError, torch.cuda.empty_cache, LOG.info)
    generator._generate_batch = generate_batch
    for state_id, state, missing in todo:
        a, b = state["adapters"]
        c = multiplier(state["method"], [norms[n]["scale"] for n in (a, b)],
                       [norms[n]["frobenius_norm"] for n in (a, b)])
        LOG.info("Generating %s (weights %.8g, %.8g)", state_id, c, c)
        with torch.no_grad():
            for module, base in original.items():
                # FP32 update then one final BF16 cast; always restore from W0.
                update = (bank.delta(a, module) + bank.delta(b, module)) * c
                weight = base.to("cuda").float() + update
                params[module + ".weight"].copy_(weight.to(params[module + ".weight"].dtype))
                del update, weight
        for dataset in missing:
            prompts = [Prompt(**p) for p in canonical[dataset]]
            records = [g.to_dict() for g in generator.generate(prompts)]
            validate_generations(records, canonical[dataset])
            write_json(run_dir / "states" / state_id / "generations" / f"{dataset}.json", records)


def judge(run_dir, raw, states, config):
    import torch
    from bench.detectors import build_detectors
    from bench.generation import Generation

    canonical = read_json(run_dir / "prompts.json")
    for detector in build_detectors(config.detectors):
        datasets = ("alpaca",) if detector.metric == "frr" else ("harmbench", "advbench")
        todo = []
        for state_id in states:
            for dataset in datasets:
                base = run_dir / "states" / state_id
                gen_path = base / "generations" / f"{dataset}.json"
                validate_generations(read_json(gen_path), canonical[dataset])
                path = base / "verdicts" / f"{dataset}__{detector.name}.json"
                if path.exists():
                    load_verdicts(path, gen_path)
                else:
                    todo.append((state_id, gen_path, path))
        if not todo:
            continue
        with detector:
            batch_size = getattr(detector.config, "batch_size", 32)
            # Wrap the large judges' primitive so their existing sorting, empty
            # response handling and progress bar are preserved.
            primitive_name = next((n for n in ("_judge_batch", "_classify_batch") if hasattr(detector, n)), None)
            if primitive_name:
                primitive = getattr(detector, primitive_name)

                def guarded_batch(chunk):
                    return retry_batches(chunk, len(chunk), primitive,
                                         torch.cuda.OutOfMemoryError, torch.cuda.empty_cache, LOG.info)

                setattr(detector, primitive_name, guarded_batch)
            for state_id, gen_path, path in todo:
                LOG.info("Judging %s / %s / %s", detector.name, state_id, gen_path.stem)
                generations = [Generation.from_dict(g) for g in read_json(gen_path)]
                if primitive_name:
                    verdicts = detector.judge(generations)
                else:
                    verdicts = retry_batches(generations, batch_size, detector.judge,
                                             torch.cuda.OutOfMemoryError, torch.cuda.empty_cache, LOG.info)
                save_verdicts(path, gen_path, [asdict(v) for v in verdicts])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=("rtx5000", "h100"))
    parser.add_argument("--run-dir", type=Path, help="Resume this run (uses its saved config by default)")
    parser.add_argument("--phase", choices=("all", "import", "geometry", "generate", "judge", "analyze"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Validate source config and print work count; write nothing")
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    saved_config = args.run_dir / "config_used.yaml" if args.run_dir else None
    config_path = args.config or (saved_config if saved_config and saved_config.exists() else Path(__file__).with_name("config.yaml"))
    raw = effective_config(config_path, args.profile)
    states = build_states([a["name"] for a in raw["model"]["adapters"]])
    if args.dry_run:
        previous = yaml.safe_load((Path(raw["pairwise"]["source_run"]) / "config_used.yaml").read_text())
        check_source_config(raw, previous)
        print("6 pairs x 5 methods = 30 pair results; import base + 4 singles + 12 pairs; generate 18 pairs.")
        print(yaml.safe_dump(raw["profiles"][raw["selected_profile"]], sort_keys=False))
        return
    run_dir = (args.run_dir or Path(raw["output"]["root"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(run_dir / "run.log"))
    snapshot = run_dir / "config_used.yaml"
    if snapshot.exists():
        if science_config(yaml.safe_load(snapshot.read_text())) != science_config(raw):
            raise ValueError("Experiment config changed; start a new run folder")
    else:
        snapshot.write_text(yaml.safe_dump(raw, sort_keys=False))
    write_json(run_dir / "plan.json", states)
    print(f"Run folder: {run_dir}", flush=True)
    prepare(run_dir, raw, states)
    if args.phase == "import":
        return
    if args.phase == "all":
        # Independent processes ensure no model survives into the next phase.
        for phase in ("geometry", "generate", "judge", "analyze"):
            subprocess.run([sys.executable, __file__, "--config", str(config_path.resolve()),
                            "--profile", raw["selected_profile"], "--run-dir", str(run_dir), "--phase", phase], check=True)
        return
    runtime = run_dir / "runtime_config.yaml"
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False))
    record_environment(run_dir, args.phase, raw)
    if args.phase == "geometry":
        if not (run_dir / "adapter_norms.json").exists():
            from pairwise_experiment.geometry import measure
            measure(run_dir, raw)
        else:
            LOG.info("Geometry already complete")
    elif args.phase in ("generate", "judge"):
        from bench.config import load_config
        config = load_config(runtime)
        if args.phase == "generate":
            generate(run_dir, raw, states, config)
        else:
            judge(run_dir, raw, states, config)
    elif args.phase == "analyze":
        from pairwise_experiment.analyze import analyze
        analyze(run_dir, raw, states)


if __name__ == "__main__":
    main()
