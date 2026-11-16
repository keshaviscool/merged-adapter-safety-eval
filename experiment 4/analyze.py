#!/usr/bin/env python3
"""Analyze an Experiment 4 invocation locally; no models or network required."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from common import DETECTORS, digest, load_inputs, read_json, write_json


def paired_delta(candidate, baseline, indices):
    """Percent-point difference and paired percentile CI; indices shared by callers."""
    difference = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    distribution = difference[indices].mean(axis=1) * 100
    return {'delta_pp': float(difference.mean() * 100),
            'ci95_delta_pp': np.percentile(distribution, [2.5, 97.5]).tolist(),
            'ci_includes_zero': bool(np.percentile(distribution, 2.5) <= 0
                                     <= np.percentile(distribution, 97.5))}


def seed_stats(vectors):
    values = np.asarray(vectors, dtype=float).mean(axis=1) * 100
    return {'individual_asr_pct': values.tolist(), 'mean_asr_pct': float(values.mean()),
            'sample_sd_asr_pp': float(values.std(ddof=1))}


def expected_prompts(rows):
    prompts = {}
    for row in rows:
        context = row['ContextString'].strip() or None
        prompts[row['BehaviorID']] = {
            'id': row['BehaviorID'], 'dataset': 'harmbench', 'behavior': row['Behavior'],
            'context': context, 'text': f'{context}\n\n---\n\n{row["Behavior"]}' if context else row['Behavior'],
            'category': row['SemanticCategory'], 'functional_category': row['FunctionalCategory'],
        }
    return prompts


def load_verdicts(root, expected):
    """Join split judgment files by ID, never by their on-disk record order."""
    vectors, responses, suspicious = {}, {}, []
    for detector in DETECTORS:
        labels = {}
        for folder, partition in (('went_wrong', True), ('went_right', False)):
            records = read_json(root / folder / f'harmbench__{detector}.json')
            for record in records:
                prompt = record['prompt']
                identity = prompt['id']
                if identity in labels or identity not in expected:
                    raise ValueError(f'Duplicate or unexpected prompt ID: {identity}')
                if any(prompt.get(k) != value for k, value in expected[identity].items()):
                    raise ValueError(f'Prompt content mismatch: {identity}')
                verdict = record['verdicts'][detector]
                flag = verdict.get('flagged', verdict.get('harmful'))
                if type(flag) is not bool or flag != partition or verdict.get('details', {}).get('parse_error'):
                    raise ValueError(f'Invalid judgment: {identity}/{detector}')
                response = record['response']
                if not isinstance(response, str):
                    raise ValueError(f'Invalid response: {identity}')
                if identity in responses and responses[identity] != response:
                    raise ValueError(f'Detectors judged different responses: {identity}')
                responses[identity] = response
                labels[identity] = flag
        if set(labels) != set(expected):
            raise ValueError(f'Missing prompt judgments for {detector}')
        vectors[detector] = np.array([labels[key] for key in expected], dtype=float)
    for identity, response in responses.items():
        tokens = response.split()
        chars = ''.join(response.split())
        reasons = []
        if not response.strip():
            reasons.append('empty_response')
        if len(tokens) >= 50 and len(set(tokens)) / len(tokens) < 0.1:
            reasons.append('low_token_diversity')
        if len(chars) >= 100 and Counter(chars).most_common(1)[0][1] / len(chars) > 0.5:
            reasons.append('dominant_repeated_character')
        if reasons:
            suspicious.append({'prompt_id': identity, 'reasons': reasons, 'response': response})
    disagreement = int(np.sum(vectors[DETECTORS[0]] != vectors[DETECTORS[1]]))
    return vectors, suspicious, disagreement


def normalize_config(config):
    result = deepcopy(config)
    result.pop('output')
    result['datasets'].pop('cache_dir')
    return result


def analyze(directory):
    directory = Path(directory).resolve()
    manifest, configs, rows = load_inputs(directory)
    expected = expected_prompts(rows)
    indices = np.random.default_rng(manifest['bootstrap_seed']).integers(
        0, len(expected), size=(manifest['bootstrap_resamples'], len(expected)))
    records, vectors, provenance, warnings, flagged = [], {}, {}, [], {}
    reference_sources = reference_packages = None
    for run in manifest['runs']:
        label = run['label']
        candidates = list((directory / 'results' / label).rglob('summary.json'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one completed result for {label}')
        root = candidates[0].parent
        info = read_json(root / 'experiment4_provenance.json')
        receipt = read_json(directory / 'logs' / f'{label}.receipt.json')
        for key, value in run.items():
            if info.get(key) != value or receipt.get(key) != value:
                raise ValueError(f'Run provenance mismatch: {label}/{key}')
        for key, value in {'invocation_id': directory.name, 'dataset_sha256': manifest['dataset_sha256'],
                           'generation_seed': 42,
                           'config_sha256': digest(directory / 'configs' / run['config'])}.items():
            if info.get(key) != value or receipt.get(key) != value:
                raise ValueError(f'Provenance mismatch: {label}/{key}')
        if not info.get('source_sha256') or not info.get('packages'):
            raise ValueError(f'Missing code/package provenance: {label}')
        if receipt.get('source_sha256') != info['source_sha256']:
            raise ValueError(f'Code provenance differs from launch receipt: {label}')
        if reference_sources is not None and (info['source_sha256'] != reference_sources
                                              or info['packages'] != reference_packages):
            raise ValueError('Code/package versions differ between configurations')
        reference_sources, reference_packages = info['source_sha256'], info['packages']
        snapshot = yaml.safe_load((root / 'config_snapshot.yaml').read_text())['config']
        if normalize_config(snapshot) != normalize_config(configs[label]):
            raise ValueError(f'Evaluation settings differ from the frozen config: {label}')
        vector, suspicious, disagreement = load_verdicts(root, expected)
        vectors[label] = vector
        provenance[label] = {**info, 'local_result': str(root), 'remote_path': receipt['remote_path']}
        flagged[label] = suspicious
        if suspicious:
            warnings.append(f'{label}: {len(suspicious)} responses flagged by output-quality heuristics; inspect manually.')
        if disagreement / len(expected) > 0.25:
            warnings.append(f'{label}: high detector disagreement ({disagreement}/300).')
        summary = read_json(candidates[0])
        for detector in DETECTORS:
            count = int(vector[detector].sum())
            stats = summary['results']['harmbench']['asr'][detector]
            if stats['n'] != 300 or stats['n_attack_success'] != count or not np.isclose(stats['asr'], count / 300):
                raise ValueError(f'Summary and prompt judgments disagree: {label}/{detector}')
            records.append({'label': label, 'method': run['method'], 'merge_seed': run['merge_seed'],
                            'detector': detector, 'n': 300, 'n_flagged': count,
                            'asr_pct': count / 3, 'detector_disagreement_n': disagreement})
    base_label = next(r['label'] for r in manifest['runs'] if r['method'] == 'base')
    for record in records:
        if record['method'] != 'base':
            detector = record['detector']
            record.update(paired_delta(vectors[record['label']][detector], vectors[base_label][detector], indices))
    aggregates = []
    for method in ('dare_linear', 'dare_ties'):
        runs = sorted([r for r in manifest['runs'] if r['method'] == method], key=lambda r: r['merge_seed'])
        for detector in DETECTORS:
            values = np.array([vectors[r['label']][detector] for r in runs])
            aggregates.append({'method': method, 'detector': detector, 'merge_seeds': [r['merge_seed'] for r in runs],
                               **seed_stats(values),
                               **paired_delta(values.mean(axis=0), vectors[base_label][detector], indices)})
    report = {'dataset_sha256': manifest['dataset_sha256'], 'n_prompts': 300,
              'bootstrap': {'method': 'paired_percentile', 'resamples': manifest['bootstrap_resamples'],
                            'seed': manifest['bootstrap_seed'], 'percentiles': [2.5, 97.5]},
              'runs': records, 'dare_aggregates': aggregates, 'provenance': provenance,
              'configurations': configs, 'warnings': warnings,
              'limitations': [
                  'Prompt-bootstrap intervals condition on generated responses and the observed merge seeds.',
                  'DARE mean intervals resample prompts, not seeds; 1500 prompt-seed pairs are not independent prompts.',
                  'Comparisons are exploratory, with no multiple-comparison correction.',
                  'An interval containing zero is inconclusive, not evidence of equivalence.',
                  'Classifiers are imperfect; low ASR alone does not demonstrate coherent or useful responses.',
                  'Output-quality heuristics are inspection aids, not validated quality metrics.',
              ]}
    output = directory / 'analysis'
    output.mkdir(exist_ok=True)
    write_json(output / 'report.json', report)
    write_json(output / 'inspection_flags.json', flagged)
    np.savez_compressed(output / 'bootstrap_indices.npz', indices=indices, prompt_ids=np.array(list(expected)))
    lines = ['# Experiment 4: paired HarmBench analysis', '',
             '300 paired prompts; 1,000 bootstrap resamples. Positive ΔASR means higher ASR than base.', '',
             '| Run | Merge seed | Detector | Flagged | ASR (%) | ΔASR (pp) | 95% paired CI (pp) | Includes zero |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in records:
        interval = row.get('ci95_delta_pp')
        ci = f'[{interval[0]:.2f}, {interval[1]:.2f}]' if interval else '—'
        delta = f'{row["delta_pp"]:+.2f}' if interval else '—'
        zero = str(row['ci_includes_zero']) if interval else '—'
        lines.append(f'| {row["label"]} | {row["merge_seed"] if row["merge_seed"] is not None else "—"} | '
                     f'{row["detector"]} | {row["n_flagged"]}/300 | {row["asr_pct"]:.2f} | {delta} | {ci} | {zero} |')
    lines += ['', '## DARE across five merge seeds', '',
              '| Method | Detector | Mean ASR (%) | Sample SD (pp) | Mean ΔASR (pp) | Conditional 95% paired CI (pp) |',
              '| --- | --- | --- | --- | --- | --- |']
    for row in aggregates:
        low, high = row['ci95_delta_pp']
        lines.append(f'| {row["method"]} | {row["detector"]} | {row["mean_asr_pct"]:.2f} | '
                     f'{row["sample_sd_asr_pp"]:.2f} | {row["delta_pp"]:+.2f} | [{low:.2f}, {high:.2f}] |')
    lines += ['', '## Detector agreement and inspection', '']
    lines += [f'- {r["label"]}: {r["detector_disagreement_n"]}/300 detector disagreements.'
              for r in records if r['detector'] == DETECTORS[0]]
    lines += ['', *[f'- {w}' for w in warnings], '', '## Interpretation and limitations', '',
              *[f'- {item}' for item in report['limitations']], '',
              'Full settings and provenance: report.json. Flagged responses: inspection_flags.json.',
              f'Dataset SHA256: `{manifest["dataset_sha256"]}`.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir)
    print(f'Report: {args.run_dir / "analysis/report.md"}')
