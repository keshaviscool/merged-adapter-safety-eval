"""All permutations/checkpoints and every named detector, from verified result CSVs."""
import csv
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
with (HERE/'data/all_path_regressions.csv').open() as f:
    rows = list(csv.DictReader(f))
DETECTORS = {
    'harmbench': ('llama_guard_3', 'harm_cls'),
    'advbench': ('llama_guard_3', 'harm_cls'),
    'alpaca': ('rejection_classifier', 'keyword_refusal'),
}
selected = [r for r in rows if r['detector'] in DETECTORS[r['dataset']]]
lookup = {(r['method'], r['order'], r['dataset'], r['detector'], int(r['checkpoint'])):
          float(r['rate_pct']) for r in selected}
assert len(selected) == len(lookup) == 2*24*5*6
orders = ['>'.join(p) for p in permutations(['a1', 'a2', 'a3', 'a4'])]
wide = []
columns = [(ds, det) for ds, detectors in DETECTORS.items() for det in detectors]
for method in ['linear', 'ties']:
    for order in orders:
        for k in range(5):
            row = dict(method=method, order=order, checkpoint=k)
            row.update({f'{ds}__{det}__pct': lookup[method, order, ds, det, k] for ds, det in columns})
            wide.append(row)
with (HERE/'data/complete_results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(wide[0]))
    writer.writeheader()
    writer.writerows(wide)
assert len(wide) == 240
out = HERE/'tables'
out.mkdir(exist_ok=True)

for method in ['linear', 'ties']:
    panels = []
    for ds, label in [('harmbench', 'HarmBench ASR'), ('advbench', 'AdvBench ASR'), ('alpaca', 'Alpaca FRR')]:
        detectors = DETECTORS[ds]
        def cell(order, checkpoint):
            return '/'.join(f'{lookup[method, order, ds, d, checkpoint]:.2f}' for d in detectors)
        for d in detectors:
            assert len({lookup[method, o, ds, d, 0] for o in orders}) == 1
        heading = 'LG / HC' if ds != 'alpaca' else 'Classifier / keyword'
        width = '0.33' if ds == 'harmbench' else '0.325'
        lines = [r'\begin{minipage}[t]{'+width+r'\linewidth}',
                 r'\centering',
                 r'{\bfseries '+label+r' (\%)}\par',
                 r'{\small '+heading+r'; base: \textbf{'+cell(orders[0], 0)+r'}}\par\vspace{5pt}',
                 r'\small\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.19}',
                 r'\begin{tabular}{@{}lrrrr@{}}',
                 r'\toprule Order & Step 1 & Step 2 & Step 3 & Step 4 \\',
                 r'\midrule']
        for i, order in enumerate(orders):
            if i and i % 6 == 0:
                lines.append(r'\addlinespace[3pt]')
            line = [order.replace('a', '').replace('>', '-')]
            line += [cell(order, k) for k in range(1, 5)]
            lines.append(' & '.join(line)+r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{minipage}']
        panels.append('\n'.join(lines))
    (out/f'raw_{method}.tex').write_text('\n\\hfill\n'.join(panels)+'\n')
print('Verified all 1,440 named-detector path/checkpoint rates; wrote two complete appendix pages.')
