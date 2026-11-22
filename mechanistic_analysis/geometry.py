"""Server-only tensor calculations, bounded to one adapted matrix at a time."""
from __future__ import annotations

import math
import re
from pathlib import Path

import torch

from ordering_experiment.compose import ties, key as expression_key
from pairwise_experiment.common import read_json, write_json
from pairwise_experiment.geometry import AdapterBank, top_mask
from mechanistic_analysis.behavior import parse_expression


def square_sum(x):
    return x.double().square().sum().item()


def pair_statistics(x, y, base_sq, fractions, masks=None):
    """All returned values are additive sufficient statistics, except max_abs_difference."""
    joint = (x != 0) & (y != 0)
    opposite = joint & (torch.sign(x) != torch.sign(y))
    product = x.double() * y.double()
    stats = {"x_sq": square_sum(x), "y_sq": square_sum(y), "dot": product.sum().item(),
             "base_sq": base_sq, "n_coordinates": x.numel(),
             "joint_nonzero": joint.sum().item(), "raw_conflicts": opposite.sum().item()}
    del product
    energy = x.double().abs() * y.double().abs()
    for fraction in fractions:
        tag = f"top{round(100 * fraction)}"
        mx, my = masks(fraction) if masks else (top_mask(x, fraction), top_mask(y, fraction))
        intersection, union = mx & my, mx | my
        union_joint = union & joint
        stats.update({
            f"{tag}_x_count": mx.sum().item(), f"{tag}_y_count": my.sum().item(),
            f"{tag}_intersection": intersection.sum().item(), f"{tag}_union": union.sum().item(),
            f"{tag}_conflicts": (intersection & opposite).sum().item(),
            f"{tag}_union_joint": union_joint.sum().item(),
            f"{tag}_union_conflicts": (union_joint & opposite).sum().item(),
            f"{tag}_joint_energy": energy[intersection].sum().item(),
            f"{tag}_conflict_energy": energy[intersection & opposite].sum().item(),
            f"{tag}_x_energy": square_sum(x[mx]), f"{tag}_y_energy": square_sum(y[my]),
        })
    return stats


def ratios(stats, fractions):
    nx, ny, nw = (math.sqrt(max(0, stats[k])) for k in ("x_sq", "y_sq", "base_sq"))
    div = lambda a, b: a / b if b else None
    result = {"composite_norm": nx, "incoming_norm": ny,
        "cosine": max(-1., min(1., stats["dot"] / (nx * ny))) if nx and ny else None,
        "composite_relative_to_base": div(nx, nw), "incoming_relative_to_base": div(ny, nw),
        "incoming_over_composite_norm": div(ny, nx),
        "raw_sign_conflict": div(stats["raw_conflicts"], stats["joint_nonzero"])}
    if "after_sq" in stats:
        result.update({"after_norm": math.sqrt(stats["after_sq"]),
                       "integration_step_norm": math.sqrt(stats["integration_step_sq"]),
                       "deviation_from_sum_norm": math.sqrt(stats["deviation_from_sum_sq"])})
    if "difference_sq" in stats:
        result.update({"difference_norm": math.sqrt(stats["difference_sq"]),
                       "difference_relative_to_base": div(math.sqrt(stats["difference_sq"]), nw),
                       "difference_relative_to_lower_endpoint": div(math.sqrt(stats["difference_sq"]), nx),
                       "changed_fraction": div(stats["changed_coordinates"], stats["n_coordinates"])})
    for fraction in fractions:
        tag = f"top{round(100 * fraction)}"
        result.update({
            f"{tag}_sign_conflict": div(stats[f"{tag}_conflicts"], stats[f"{tag}_intersection"]),
            f"{tag}_energy_weighted_conflict": div(stats[f"{tag}_conflict_energy"], stats[f"{tag}_joint_energy"]),
            f"{tag}_union_sign_conflict": div(stats[f"{tag}_union_conflicts"], stats[f"{tag}_union_joint"]),
            f"{tag}_jaccard": div(stats[f"{tag}_intersection"], stats[f"{tag}_union"]),
            f"{tag}_retained_fraction_x": div(stats[f"{tag}_x_count"], stats["n_coordinates"]),
            f"{tag}_retained_fraction_y": div(stats[f"{tag}_y_count"], stats["n_coordinates"]),
            f"{tag}_retained_energy_x": div(stats[f"{tag}_x_energy"], stats["x_sq"]),
            f"{tag}_retained_energy_y": div(stats[f"{tag}_y_energy"], stats["y_sq"]),
        })
    return result


def locations(module):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", module)
    return [("module", module), ("block", f"layer_{match.group(1)}" if match else "other"),
            ("module_type", module.rsplit(".", 1)[-1]), ("global", "all")]


class MatrixStates:
    """Cache only this module's expression tensors and top-magnitude masks."""

    def __init__(self, deltas, density):
        self.deltas, self.density = deltas, density
        self.cache = {"base": torch.zeros_like(next(iter(deltas.values()))), **deltas}
        self.mask_cache = {}

    def get(self, key):
        if key not in self.cache:
            self.cache[key] = self.evaluate(parse_expression(key))
        return self.cache[key]

    def evaluate(self, expr):
        if expr[0] == "leaf":
            return self.deltas[expr[1]]
        if expr[0] == "sum":
            total = self.deltas[expr[1][0]]
            for name in expr[1][1:]:
                total = total + self.deltas[name]
            return total
        return ties(self.get(expression_key(expr[1])), self.get(expression_key(expr[2])), self.density)

    def mask(self, key, fraction):
        if (key, fraction) not in self.mask_cache:
            self.mask_cache[key, fraction] = (torch.zeros_like(self.get(key), dtype=torch.bool)
                                              if key == "base" else top_mask(self.get(key), fraction))
        return self.mask_cache[key, fraction]


def endpoint_statistics(low, high, base_sq, fractions):
    stats = pair_statistics(low, high, base_sq, fractions)
    difference = high - low
    stats.update({"difference_sq": square_sum(difference),
                  "changed_coordinates": (difference != 0).sum().item(),
                  "max_abs_difference": difference.abs().max().item()})
    return stats


@torch.inference_mode()
def measure(run_dir, cfg):
    from transformers import AutoModelForCausalLM
    prepared = read_json(run_dir / "prepared.json")
    transitions = read_json(run_dir / "transitions.json")
    plan = read_json(run_dir / "source_plan.json")
    bank = AdapterBank(prepared["source_model"]["adapters"])
    for name, actual in bank.metadata.items():
        for field in ("config_sha256", "weights_sha256"):
            if actual[field] != prepared["expected_adapters"][name][field]:
                raise ValueError(f"Adapter {name} differs from the earlier weight-space analysis ({field})")
    write_json(run_dir / "adapter_metadata.json", bank.metadata)
    modules = sorted(bank.shapes)
    write_json(run_dir / "modules.json", modules)
    pending = [m for m in modules if not (run_dir / "module_cache" / f"{m}.json").exists()]
    if not pending:
        print("All module calculations are already complete.", flush=True)
        write_json(run_dir / "geometry_complete.json", {"modules": len(modules), "transitions": len(transitions)})
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    base_metadata_path = run_dir / "base_metadata.json"
    revision = read_json(base_metadata_path).get("revision") if base_metadata_path.exists() else None
    model = AutoModelForCausalLM.from_pretrained(prepared["source_model"]["base_model"],
        torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=prepared["source_model"].get("trust_remote_code", False), revision=revision)
    write_json(base_metadata_path, {"model": prepared["source_model"]["base_model"],
        "revision": getattr(model.config, "_commit_hash", None), "dtype": "bfloat16",
        "original_ordering_revision": "not recorded; reconstruction identity cannot be certified"})
    params = dict(model.named_parameters())
    for index, module in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {module}", flush=True)
        base = params[module + ".weight"].detach()
        if tuple(base.shape) != bank.shapes[module]:
            raise ValueError(f"Base shape mismatch: {module}")
        base_sq = square_sum(base)
        store = MatrixStates({name: bank.delta(name, module) for name in bank.factors}, prepared["ties_density"])
        transition_rows = []
        for identifier, transition in transitions.items():
            current, incoming = transition["before"], transition["incoming"]
            x, y, after = store.get(current), store.get(incoming), store.get(transition["after"])
            stats = pair_statistics(x, y, base_sq, cfg["top_fractions"],
                masks=lambda fraction: (store.mask(current, fraction), store.mask(incoming, fraction)))
            stats.update({"after_sq": square_sum(after), "integration_step_sq": square_sum(after - x),
                          "deviation_from_sum_sq": square_sum(after - (x + y))})
            transition_rows.append({"transition_id": identifier, "stats": stats})
        low, high = (store.get(prepared[k]) for k in ("lower_endpoint", "higher_endpoint"))
        endpoint_rows = [{"representation": "merge_update_fp32",
                          "stats": endpoint_statistics(low, high, base_sq, cfg["top_fractions"])}]
        # These are the effective updates of the once-cast checkpoints, not just
        # the ideal FP32 expressions. The base is exactly the BF16-loaded tensor.
        effective_low = (base.float() + low).to(torch.bfloat16).float() - base.float()
        effective_high = (base.float() + high).to(torch.bfloat16).float() - base.float()
        endpoint_rows.append({"representation": "checkpoint_effective_bfloat16",
            "stats": endpoint_statistics(effective_low, effective_high, base_sq, cfg["top_fractions"])})
        reference = store.get(prepared["linear_endpoint"])
        equal = all(torch.equal(store.get(path[-1]), reference) for path in plan["paths"]["linear"].values())
        if not equal:
            raise ValueError("Canonical LINEAR endpoint invariance failed")
        write_json(run_dir / "module_cache" / f"{module}.json", {
            "module": module, "transitions": transition_rows, "endpoints": endpoint_rows,
            "linear_control": {"n_paths": 24, "equal": equal, "max_absolute_difference": 0.0},
        })
        del store, low, high, effective_low, effective_high, reference, x, y, after
    write_json(run_dir / "geometry_complete.json", {"modules": len(modules), "transitions": len(transitions)})
