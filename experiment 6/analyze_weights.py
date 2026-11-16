#!/usr/bin/env python3
"""Describe effective LoRA updates on CPU; no model generation or merging."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

DEFAULT_ADAPTERS = {
    'a1': 'Akarshika/qwen25-3b-text2sql-lora',
    'a2': 'madokalif/qwen2.5-3b-translation-warmup-adapter',
    'a3': 'balaguhanesh/tool-call-ft',
    'a4': 'Pilestin/qwen2.5-3b-debug-v1-lora',
}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter(source, revision=None):
    root = Path(source).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(
            source, revision=revision,
            allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin'],
        ))
    config_path = root / 'adapter_config.json'
    config = json.loads(config_path.read_text())
    if config.get('peft_type', 'LORA') != 'LORA':
        raise ValueError(f'{source}: only ordinary LoRA and rsLoRA are supported')
    for field in ('use_dora', 'use_bdlora', 'alora_invocation_tokens', 'layer_replication'):
        if config.get(field):
            raise ValueError(f'{source}: {field} requires a different effective-update calculation')
    weights_path = root / 'adapter_model.safetensors'
    if weights_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(weights_path), device='cpu')
    else:
        weights_path = root / 'adapter_model.bin'
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
    modules = {}
    for name, tensor in state.items():
        match = re.fullmatch(r'(.+)\.lora_([AB])(?:\.[^.]+)?\.weight', name)
        if not match:
            raise ValueError(f'{source}: unsupported tensor {name}; refusing to silently omit updates')
        module, factor = match.groups()
        factors = modules.setdefault(module, {})
        if factor in factors:
            raise ValueError(f'{source}: duplicate {factor} factor for {module}')
        if tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError(f'{source}: {name} must be a finite 2D matrix')
        factors[factor] = tensor.float()
    if not modules:
        raise ValueError(f'{source}: no LoRA matrices found')
    for module, factors in modules.items():
        if set(factors) != {'A', 'B'}:
            raise ValueError(f'{source}: incomplete A/B pair for {module}')
        a, b = factors['A'], factors['B']
        rank = pattern_value(config.get('rank_pattern', {}), module, config['r'])
        alpha = pattern_value(config.get('alpha_pattern', {}), module, config['lora_alpha'])
        if rank <= 0 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f'{source}: configured rank does not match factors for {module}')
        factors['scale'] = alpha / (math.sqrt(rank) if config.get('use_rslora') else rank)
        factors['shape'] = (b.shape[0], a.shape[1])
    metadata = {
        'source': source, 'requested_revision': revision,
        'resolved_snapshot': root.name if root.parent.name == 'snapshots' else None,
        'config': config, 'config_sha256': checksum(config_path),
        'weights_sha256': checksum(weights_path), 'module_count': len(modules),
    }
    return modules, metadata


def pattern_value(patterns, module, default):
    # Match PEFT's ordered suffix-pattern lookup, including per-module overrides.
    for key, value in patterns.items():
        if re.match(rf'(.*\.)?{key}$', module):
            return value
    return default


def update_block(factors, start, stop):
    block = (factors['B'][start:stop] @ factors['A']) * factors['scale']
    if not torch.isfinite(block).all():
        raise ValueError('Reconstructed update contains non-finite values')
    return block


def module_metrics(factors_by_adapter, chunk_rows=256):
    """Exact entry comparisons in row chunks; FP32 products and FP64 reductions."""
    norms = {label: 0.0 for label in factors_by_adapter}
    zero_counts = {label: 0 for label in factors_by_adapter}
    pairs = {}
    for left, right in combinations(factors_by_adapter, 2):
        if factors_by_adapter[left]['shape'] == factors_by_adapter[right]['shape']:
            pairs[left, right] = dict(dot=0.0, conflicts=0, both_nonzero=0, opposing=0)
    max_rows = max(f['shape'][0] for f in factors_by_adapter.values())
    for start in range(0, max_rows, chunk_rows):
        blocks = {}
        for label, factors in factors_by_adapter.items():
            if start >= factors['shape'][0]:
                continue
            block = update_block(factors, start, start + chunk_rows)
            blocks[label] = block
            norms[label] += block.double().square().sum().item()
            zero_counts[label] += (block == 0).sum().item()
        for (left, right), stats in pairs.items():
            if left not in blocks:
                continue
            a, b = blocks[left], blocks[right]
            stats['dot'] += (a.double() * b.double()).sum().item()
            stats['conflicts'] += (a.sign() != b.sign()).sum().item()
            both = (a != 0) & (b != 0)
            stats['both_nonzero'] += both.sum().item()
            stats['opposing'] += ((a.sign() != b.sign()) & both).sum().item()
    magnitudes = {}
    for label, squared in norms.items():
        size = math.prod(factors_by_adapter[label]['shape'])
        magnitudes[label] = {
            'frobenius_norm': math.sqrt(squared), 'entries': size,
            'zero_entries': zero_counts[label], 'zero_fraction': zero_counts[label] / size,
            'scale': factors_by_adapter[label]['scale'],
        }
    for (left, right), stats in pairs.items():
        size = magnitudes[left]['entries']
        denominator = magnitudes[left]['frobenius_norm'] * magnitudes[right]['frobenius_norm']
        stats['cosine_similarity'] = max(-1.0, min(1.0, stats['dot'] / denominator)) if denominator else None
        stats['sign_conflict_rate'] = stats['conflicts'] / size
        stats['nonzero_sign_conflict_rate'] = stats['opposing'] / stats['both_nonzero'] if stats['both_nonzero'] else None
    return magnitudes, pairs


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap(path, modules, columns, values, title, low, high):
    """Standalone PNG heatmap with labeled rows, numeric cells, and scale endpoints."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=13)
    label_width = max(300, int(max(map(len, modules)) * 8))
    cell_width, row_height = 100, 22
    canvas = Image.new('RGB', (label_width + cell_width * len(columns) + 30, 110 + row_height * len(modules)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill='black', font=font)
    draw.text((10, 30), f'Blue = {low:.3g}; red = {high:.3g}; gray = missing/undefined', fill='black', font=font)
    for j, column in enumerate(columns):
        draw.text((label_width + j * cell_width + 4, 62), column, fill='black', font=font)
    for i, module in enumerate(modules):
        y = 88 + i * row_height
        draw.text((8, y + 3), module, fill='black', font=font)
        for j, column in enumerate(columns):
            value = values.get((module, column))
            if value is None:
                color, caption = (225, 225, 225), 'NA'
            else:
                t = min(1, max(0, (value - low) / (high - low))) if high > low else 0.5
                color = (int(255 * min(1, 2 * t)), int(255 * (1 - abs(2 * t - 1))), int(255 * min(1, 2 * (1 - t))))
                caption = f'{value:.3f}'
            x = label_width + j * cell_width
            draw.rectangle((x, y, x + cell_width - 2, y + row_height - 1), fill=color)
            draw.text((x + 4, y + 3), caption, fill='black', font=font)
    canvas.save(path)


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', value)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', action='append', metavar='LABEL=PATH_OR_HUB_ID', help='Repeat four times to replace defaults')
    parser.add_argument('--revision', action='append', default=[], metavar='LABEL=REVISION')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--chunk-rows', type=int, default=256)
    parser.add_argument('--merge-weight', type=float, default=0.25, help='Additional magnitude column only; raw diagnostics are unchanged')
    args = parser.parse_args()
    if args.chunk_rows < 1 or not math.isfinite(args.merge_weight):
        parser.error('chunk-rows must be positive and merge-weight finite')
    def assignments(items):
        result = {}
        for item in items:
            key, separator, value = item.partition('=')
            if not separator or not key or not value or key in result:
                parser.error(f'Expected unique LABEL=VALUE, got {item!r}')
            result[key] = value
        return result
    sources = assignments(args.adapter) if args.adapter else DEFAULT_ADAPTERS
    revisions = assignments(args.revision)
    if len(sources) != 4 or set(revisions) - set(sources):
        parser.error('Provide exactly four adapters; revision labels must refer to those adapters')
    adapters, metadata = {}, {}
    for label, source in sources.items():
        print(f'Loading {label}: {source}', flush=True)
        adapters[label], metadata[label] = load_adapter(source, revisions.get(label))
    orientations = {bool(m['config'].get('fan_in_fan_out', False)) for m in metadata.values()}
    if len(orientations) != 1:
        raise ValueError('Mixed fan_in_fan_out orientations require explicit alignment before comparison')
    modules = sorted(set().union(*(set(a) for a in adapters.values())), key=natural_key)
    labels = list(sources)
    pair_labels = [(a, b, f'{a}-{b}') for a, b in combinations(labels, 2)]
    magnitude_rows, pair_rows = [], []
    for module in modules:
        available = {label: adapters[label][module] for label in labels if module in adapters[label]}
        magnitudes, pairs = module_metrics(available, args.chunk_rows)
        for label in labels:
            stats = magnitudes.get(label)
            magnitude_rows.append(dict(
                module=module, adapter=label, status='ok' if stats else 'missing_module',
                shape='x'.join(map(str, available[label]['shape'])) if stats else '',
                frobenius_norm=stats['frobenius_norm'] if stats else None,
                weighted_frobenius_norm=abs(args.merge_weight) * stats['frobenius_norm'] if stats else None,
                scale=stats['scale'] if stats else None, entries=stats['entries'] if stats else None,
                zero_entries=stats['zero_entries'] if stats else None,
                zero_fraction=stats['zero_fraction'] if stats else None,
            ))
        for left, right, pair in pair_labels:
            stats = pairs.get((left, right))
            status = 'missing_module' if left not in available or right not in available else 'shape_mismatch' if stats is None else 'zero_norm' if stats['cosine_similarity'] is None else 'ok'
            pair_rows.append(dict(
                module=module, pair=pair, status=status,
                cosine_similarity=stats['cosine_similarity'] if stats else None,
                sign_conflict_rate=stats['sign_conflict_rate'] if stats else None,
                nonzero_sign_conflict_rate=stats['nonzero_sign_conflict_rate'] if stats else None,
                both_nonzero_entries=stats['both_nonzero'] if stats else None,
            ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_'), dir=args.output_root))
    write_csv(output / 'update_magnitudes.csv', magnitude_rows)
    write_csv(output / 'pairwise_metrics.csv', pair_rows)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), adapters=metadata,
                      merge_weight=args.merge_weight, chunk_rows=args.chunk_rows,
                      torch_version=torch.__version__, numpy_version=np.__version__,
                      arithmetic='FP32 reconstruction; FP64 reductions; exact zero comparisons',
                      fan_in_fan_out='Metrics use BA orientation; a common transpose preserves these metrics')
    (output / 'metadata.json').write_text(json.dumps(provenance, indent=2))
    magnitude_values = {(r['module'], r['adapter']): math.log10(r['frobenius_norm']) if r['frobenius_norm'] else None for r in magnitude_rows}
    limits = [v for v in magnitude_values.values() if v is not None]
    heatmap(output / 'magnitude_heatmap.png', modules, labels, magnitude_values,
            'Effective update magnitude: log10(Frobenius norm); zero norms shown as NA', min(limits, default=0), max(limits, default=1))
    for metric, filename, low, high in [('cosine_similarity', 'cosine_heatmap.png', -1, 1), ('sign_conflict_rate', 'sign_conflict_heatmap.png', 0, 1)]:
        heatmap(output / filename, modules, [p[2] for p in pair_labels],
                {(r['module'], r['pair']): r[metric] for r in pair_rows}, metric, low, high)
    lines = ['# Experiment 6: weight-space diagnostics', '',
             f'Analyzed {len(modules)} distinct modules across four adapters and six pairs.', '',
             'Updates are reconstructed as sBA, using configured alpha/r or alpha/sqrt(r), including per-module overrides.',
             f'Raw diagnostics exclude merge coefficients. A separate magnitude column uses coefficient {args.merge_weight}.', '',
             '## Pair summaries', '',
             '| Pair | Comparable modules | Mean cosine | Mean conflict rate |',
             '| --- | ---: | ---: | ---: |']
    for _, _, pair in pair_labels:
        rows = [r for r in pair_rows if r['pair'] == pair and r['sign_conflict_rate'] is not None]
        cosines = [r['cosine_similarity'] for r in rows if r['cosine_similarity'] is not None]
        cosine = f'{np.mean(cosines):.4f}' if cosines else 'NA'
        conflict = f'{np.mean([r["sign_conflict_rate"] for r in rows]):.2%}' if rows else 'NA'
        lines.append(f'| {pair} | {len(rows)} | {cosine} | {conflict} |')
    bases = {m['config'].get('base_model_name_or_path') for m in metadata.values()}
    if len(bases) > 1:
        lines += ['', '**Warning:** adapters declare different base model paths; verify common model ancestry before interpreting alignment.']
    lines += ['', '## Interpretation and limitations', '',
              '- Pair summary means weight modules equally; inspect the layer-by-layer CSVs and heatmaps.',
              '- Larger absolute magnitudes across differently sized modules can reflect matrix size.',
              '- Missing modules and shape mismatches are explicit; they are not replaced by zero updates.',
              '- Sign conflict uses sign(0)=0, so zero versus nonzero counts as a disagreement. The additional nonzero-only rate excludes these cases.',
              '- Cosine is undefined for zero-norm updates. No epsilon threshold or magnitude pruning is applied.',
              '- PEFT non-SVD TIES operates on LoRA factors; these effective-update metrics do not directly measure its internal factor conflicts.',
              '- These are descriptive diagnostics, not evidence of safety, causation, or predictive performance.',
              '- No behavioral measurements are inferred here. Compare with matching merge experiments separately.',
              '- Unsupported tensors or adapter variants cause an error instead of being silently omitted.', '',
              'See metadata.json for checkpoint hashes, configuration, and arithmetic details.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Outputs: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
