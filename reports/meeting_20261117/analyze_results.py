"""Reproduce the meeting analysis from saved generations and per-prompt verdicts.

Run from the repository root: env/bin/python reports/meeting_20261117/analyze_results.py
No models are loaded, and original experiment outputs are never changed.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import csv
import hashlib
import json
import os

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEQ = ROOT / 'sequential_merging/results/20261117_001902'
ORD = ROOT / 'ordering_experiment/results/20261117_045604'
DATA = HERE / 'data'
FIG = HERE / 'figures'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = 20000
SEED = 20261117
DETS = {'harmbench': ['llama_guard_3', 'harm_cls'],
        'advbench': ['llama_guard_3', 'harm_cls'],
        'alpaca': ['rejection_classifier', 'keyword_refusal']}
SHORT = {'llama_guard_3': 'Llama Guard', 'harm_cls': 'HarmCLS',
         'rejection_classifier': 'Refusal classifier', 'keyword_refusal': 'Keyword refusal'}
COLORS = ['#21618C', '#CB6336']
INPUTS = set()

def read(p):
    INPUTS.add(p)
    return json.loads(p.read_text())

def csv_out(name, rows):
    with (DATA / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_json(name, data):
    (DATA / name).write_text(json.dumps(data, indent=2) + '\n')

def dirname(key):
    return key.replace('(', '_').replace(')', '').replace(',', '-')

def signature(g):
    return json.dumps(g['prompt'], sort_keys=True)

def load_state(folder, seq=False):
    result = {}
    summary = read(folder / 'summary.json')
    if seq:
        summary = summary['results']
    for ds, dets in DETS.items():
        gen = read(folder / 'generations' / f'{ds}.json')
        ids = [g['prompt']['id'] for g in gen]
        assert len(set(ids)) == len(ids), (folder, ds, 'duplicate IDs')
        verdicts = {}
        for det in dets:
            if seq:
                records = sum([read(folder / tag / f'{ds}__{det}.json')
                               for tag in ['went_wrong', 'went_right']], [])
                byid = {r['prompt']['id']: r for r in records}
                assert len(byid) == len(records) == len(gen)
                assert set(byid) == set(ids)
                for g in gen:
                    r = byid[g['prompt']['id']]
                    assert r['prompt'] == g['prompt'] and r['response'] == g['response']
                values = [byid[i]['verdicts'][det] for i in ids]
            else:
                values = read(folder / 'verdicts' / f'{ds}__{det}.json')
            assert len(values) == len(gen)
            assert all(type(v['flagged']) is bool for v in values)
            flags = np.array([v['flagged'] for v in values], dtype=np.int8)
            metric = 'frr' if ds == 'alpaca' else 'asr'
            assert abs(flags.mean() - summary[ds][metric][det][metric]) < 1e-12
            verdicts[det] = {'flags': flags, 'parse_errors': sum(bool(v.get('details', {}).get('parse_error')) for v in values),
                             'empty_skips': sum(v.get('details', {}).get('skipped') == 'empty response' for v in values)}
        result[ds] = {'gen': gen, 'verdicts': verdicts}
    return result

def flags(state, ds, det):
    if det == 'any':
        return np.maximum.reduce([v['flags'] for v in state[ds]['verdicts'].values()])
    return state[ds]['verdicts'][det]['flags']

def paired(a, b, rng):
    """Percentile paired bootstrap for rate(a)-rate(b), ordinary prompt resampling.

    Resampling the empirical {-1,0,+1} difference distribution by multinomial
    is exactly equivalent to resampling paired prompt indices for this statistic.
    """
    d = a.astype(int) - b.astype(int)
    counts = np.array([(d == -1).sum(), (d == 0).sum(), (d == 1).sum()])
    boot = rng.multinomial(len(d), counts / len(d), size=B)
    lo, hi = np.quantile((boot[:, 2] - boot[:, 0]) / len(d), [.025, .975]) * 100
    return dict(delta_pp=float(d.mean() * 100), ci_low_pp=float(lo), ci_high_pp=float(hi),
                newly_flagged=int(counts[2]), newly_unflagged=int(counts[0]), n=len(d))

def savefig(name):
    plt.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.savefig(FIG / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close()

def main():
    DATA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    plan = read(ORD / 'plan.json')
    saved_configs = {}
    for label, folder in [('sequential', SEQ), ('ordering', ORD)]:
        config_path = folder/'config_used.yaml'
        INPUTS.add(config_path)
        cfg = yaml.safe_load(config_path.read_text())
        assert cfg['model']['base_model'] == 'Qwen/Qwen2.5-3B-Instruct'
        assert cfg['generation']['do_sample'] is False
        saved_configs[label] = cfg
    report_metadata = read(SEQ/'experiment_report.json')
    paths = plan['paths']
    seq_folders = sorted(SEQ.glob('stage[0-4]_*'))
    assert len(seq_folders) == 5
    sequential = [load_state(p, seq=True) for p in seq_folders]
    states = {k: load_state(ORD / 'states' / dirname(k)) for k in plan['models']}
    assert len(states) == 46
    for method in ['linear', 'ties']:
        assert len(paths[method]) == 24
    for ds in DETS:
        reference = [signature(g) for g in states['base'][ds]['gen']]
        for state in [*states.values(), *sequential]:
            assert [signature(g) for g in state[ds]['gen']] == reference, (ds, 'prompt mismatch')

    rng = np.random.default_rng(SEED)
    bootstrap = []
    rates = []
    for family, items in [('sequential', {str(i): s for i, s in enumerate(sequential)}), ('ordering', states)]:
        for key, state in items.items():
            for ds, dets in DETS.items():
                for det in dets:
                    v = state[ds]['verdicts'][det]
                    rates.append(dict(experiment=family, state=key, dataset=ds, detector=det,
                                      n=len(v['flags']), flagged=int(v['flags'].sum()), rate_pct=float(v['flags'].mean()*100),
                                      parse_errors=v['parse_errors'], empty_skips=v['empty_skips']))
    csv_out('state_rates.csv', rates)

    def comparison(label, source, target, ds, det):
        r = dict(comparison=label, dataset=ds, detector=det,
                 **paired(flags(target, ds, det), flags(source, ds, det), rng))
        bootstrap.append(r)
        return r

    for i in range(1, 5):
        for det in DETS['harmbench']:
            comparison(f'sequential: step {i} vs step {i-1}', sequential[i-1], sequential[i], 'harmbench', det)
            comparison(f'sequential: step {i} vs base', sequential[0], sequential[i], 'harmbench', det)
    for det in DETS['harmbench']:
        comparison('sequential: final vs step 2 (apparent recovery)', sequential[2], sequential[4], 'harmbench', det)
        for method in ['linear', 'ties']:
            final = states[paths[method]['a1>a2>a3>a4'][-1]]
            comparison(f'{method}: fixed-order endpoint vs A1', states['a1'], final, 'harmbench', det)
        comparison('ordering: LINEAR endpoint vs fixed-order TIES endpoint',
                   states[paths['ties']['a1>a2>a3>a4'][-1]],
                   states[paths['linear']['a1>a2>a3>a4'][-1]], 'harmbench', det)

    path_rows, gate_rows, trajectory = [], [], []
    seq_regressions = []
    for ds, dets in DETS.items():
        for det in dets:
            rr = [float(flags(s, ds, det).mean() * 100) for s in sequential]
            for i in range(5):
                seq_regressions.append(dict(dataset=ds, detector=det, checkpoint=i, rate_pct=rr[i],
                                            local_pp=None if i == 0 else rr[i]-rr[i-1], cumulative_pp=rr[i]-rr[0]))
    csv_out('sequential_regressions.csv', seq_regressions)

    for method, orders in paths.items():
        for ds, dets in DETS.items():
            for det in dets + ['any']:
                curves = {o: np.array([flags(states[k], ds, det).mean()*100 for k in pp]) for o, pp in orders.items()}
                for order, rr in curves.items():
                    for i, key in enumerate(orders[order]):
                        path_rows.append(dict(method=method, order=order, checkpoint=i, state=key, dataset=ds, detector=det,
                                              rate_pct=float(rr[i]), local_pp=None if not i else float(rr[i]-rr[i-1]),
                                              cumulative_pp=float(rr[i]-rr[0])))
                diffs = np.array([np.diff(r) for r in curves.values()])
                trajectory.append(dict(method=method, dataset=ds, detector=det,
                                       nondecreasing_paths=int(np.all(diffs >= -1e-10, axis=1).sum()),
                                       nonincreasing_paths=int(np.all(diffs <= 1e-10, axis=1).sum()),
                                       fluctuating_paths=int((np.any(diffs < -1e-10, axis=1) & np.any(diffs > 1e-10, axis=1)).sum())))
                if ds == 'alpaca':
                    continue
                for eps in [1., 2., 5.]:
                    for tau in [2., 5., 10.]:
                        whole_local, whole_cum, hidden, reachable_hidden = [], [], [], []
                        for o, r in curves.items():
                            local_pass = np.diff(r) <= eps + 1e-10
                            cum_pass = r[1:] - r[0] <= tau + 1e-10
                            if local_pass.all(): whole_local.append(o)
                            if cum_pass.all(): whole_cum.append(o)
                            if local_pass.all() and not cum_pass.all(): hidden.append(o)
                            first_bad_cum = next((k for k, ok in enumerate(cum_pass) if not ok), None)
                            if first_bad_cum is not None and local_pass[:first_bad_cum+1].all(): reachable_hidden.append(o)
                        gate_rows.append(dict(method=method, dataset=ds, detector=det, eps_pp=eps, tau_pp=tau,
                                              local_all_pass=len(whole_local), cumulative_all_pass=len(whole_cum),
                                              hidden_full_path=len(hidden), hidden_at_reachable_step=len(reachable_hidden),
                                              hidden_orders=';'.join(hidden)))

    csv_out('all_path_regressions.csv', path_rows)
    csv_out('gates.csv', gate_rows)
    csv_out('trajectory_shapes.csv', trajectory)

    ordering_stats = []
    endpoint_stats = []
    for method, orders in paths.items():
        for ds, dets in DETS.items():
            for det in dets:
                for k in range(1, 5):
                    keys = sorted({p[k] for p in orders.values()})
                    rr = [flags(states[key], ds, det).mean()*100 for key in keys]
                    ordering_stats.append(dict(method=method, dataset=ds, detector=det, checkpoint=k,
                                               n_unique=len(keys), min_pct=float(min(rr)), max_pct=float(max(rr)),
                                               range_pp=float(max(rr)-min(rr))))
                ends = sorted({p[-1] for p in orders.values()})
                best = min(ends, key=lambda key: flags(states[key], ds, det).mean())
                worst = max(ends, key=lambda key: flags(states[key], ds, det).mean())
                if ds == 'harmbench':
                    for key in ends:
                        comparison(f'{method}: endpoint {key} vs base', states['base'], states[key], ds, det)
                    comparison(f'{method}: selected max endpoint vs min endpoint', states[best], states[worst], ds, det)
                disagree = max((float(np.mean(flags(states[a], ds, det) != flags(states[b], ds, det))*100)
                                for a, b in combinations(ends, 2)), default=0.)
                endpoint_stats.append(dict(method=method, dataset=ds, detector=det,
                                           base_pct=float(flags(states['base'], ds, det).mean()*100),
                                           min_pct=float(flags(states[best], ds, det).mean()*100),
                                           max_pct=float(flags(states[worst], ds, det).mean()*100),
                                           min_state=best, max_state=worst, max_disagreement_pct=disagree))
    csv_out('ordering_spreads.csv', ordering_stats)
    csv_out('endpoints.csv', endpoint_stats)

    singles = []
    for name in ['a1', 'a2', 'a3', 'a4']:
        for det in DETS['harmbench']:
            r = comparison(f'single {name} vs base (ordering run)', states['base'], states[name], 'harmbench', det)
            singles.append(dict(adapter=name, rate_pct=float(flags(states[name], 'harmbench', det).mean()*100), **r))
    csv_out('single_adapters.csv', singles)

    # Endpoint association with the last adapter: a descriptive grouping, not causal attribution.
    last_rows = []
    for det in DETS['harmbench']:
        for last in ['a1', 'a2', 'a3', 'a4']:
            keys = sorted({p[-1] for o,p in paths['ties'].items() if o.split('>')[-1] == last})
            rr = [flags(states[k], 'harmbench', det).mean()*100 for k in keys]
            last_rows.append(dict(detector=det, last=last, unique_endpoints=len(keys),
                                  mean_pct=float(np.mean(rr)), min_pct=float(min(rr)), max_pct=float(max(rr))))
    csv_out('ties_last_adapter.csv', last_rows)

    # Concrete equal-threshold gate example; retain all qualifying paths in gates.csv.
    witnesses = [g for g in gate_rows if g['dataset']=='harmbench' and g['detector']=='llama_guard_3'
                 and g['eps_pp']==5 and g['tau_pp']==5 and g['method']=='ties']
    witness_order = witnesses[0]['hidden_orders'].split(';')[0] if witnesses[0]['hidden_orders'] else None
    witness = []
    if witness_order:
        for det in DETS['harmbench']:
            pp = paths['ties'][witness_order]
            rr = [flags(states[key], 'harmbench', det).mean()*100 for key in pp]
            for i in range(1, 5):
                r = comparison(f'gate example {witness_order}: step {i} vs base', states[pp[0]], states[pp[i]], 'harmbench', det)
                local = comparison(f'gate example {witness_order}: step {i} vs previous', states[pp[i-1]], states[pp[i]], 'harmbench', det)
                witness.append(dict(order=witness_order, detector=det, step=i, asr_pct=float(rr[i]),
                                    local_pp=float(rr[i]-rr[i-1]), cumulative_pp=float(rr[i]-rr[0]),
                                    cumulative_ci=[r['ci_low_pp'], r['ci_high_pp']], local_ci=[local['ci_low_pp'],local['ci_high_pp']]))
    csv_out('paired_bootstrap.csv', bootstrap)

    # Between-run response differences are not an estimate of the adapter effect.
    cross = []
    for i, ss in enumerate(sequential):
        oo = states[paths['linear']['a1>a2>a3>a4'][i]]
        for ds, dets in DETS.items():
            n_diff = sum(a['response'] != b['response'] for a,b in zip(ss[ds]['gen'],oo[ds]['gen']))
            for det in dets:
                cross.append(dict(checkpoint=i, dataset=ds, detector=det, n=len(ss[ds]['gen']),
                                  responses_different=n_diff,
                                  verdicts_different=int(np.sum(flags(ss,ds,det)!=flags(oo,ds,det))),
                                  sequential_rate=float(flags(ss,ds,det).mean()*100),
                                  ordering_rate=float(flags(oo,ds,det).mean()*100)))
    csv_out('between_run_differences.csv', cross)

    categories = []
    for det in DETS['harmbench']:
        for field in ['category', 'functional_category']:
            vals = [g['prompt'][field] for g in states['base']['harmbench']['gen']]
            for category in sorted(set(vals)):
                ix = np.array([v == category for v in vals])
                for method in ['linear', 'ties']:
                    # Use fixed a1>a2>a3>a4 for category comparison, avoiding category-wise cherry-picking.
                    final = states[paths[method]['a1>a2>a3>a4'][-1]]
                    base = flags(states['base'],'harmbench',det)[ix]
                    end = flags(final,'harmbench',det)[ix]
                    categories.append(dict(method=method, detector=det, field=field, category=category,n=int(ix.sum()),
                                           base_pct=float(base.mean()*100),end_pct=float(end.mean()*100),delta_pp=float((end-base).mean()*100)))
    csv_out('category_breakdown.csv', categories)

    disagreement = []
    for family, entries in [('sequential', {str(i):s for i,s in enumerate(sequential)}), ('ordering',states)]:
        for key,state in entries.items():
            for ds in ['harmbench','advbench']:
                lg=flags(state,ds,'llama_guard_3'); hc=flags(state,ds,'harm_cls')
                disagreement.append(dict(experiment=family,state=key,dataset=ds,n=len(lg),
                                         both=int(((lg==1)&(hc==1)).sum()),lg_only=int(((lg==1)&(hc==0)).sum()),
                                         hc_only=int(((lg==0)&(hc==1)).sum()),neither=int(((lg==0)&(hc==0)).sum())))
    csv_out('detector_agreement.csv',disagreement)

    empties=[]
    for family, entries in [('sequential', {str(i):s for i,s in enumerate(sequential)}), ('ordering',states)]:
        for key,state in entries.items():
            for ds in DETS:
                empties.append(dict(experiment=family,state=key,dataset=ds,
                                    empty_responses=sum(not g['response'].strip() for g in state[ds]['gen'])))
    csv_out('empty_responses.csv',empties)

    # Validate supplied aggregate artifacts against independently reconstructed raw values.
    INPUTS.add(ORD/'order_results.csv')
    raw_lookup = {(r['method'],r['order'],r['checkpoint'],r['dataset'],r['detector']):r['rate_pct']/100 for r in path_rows}
    supplied = list(csv.DictReader((ORD/'order_results.csv').open()))
    assert len(supplied) == len(path_rows) == 2160
    for r in supplied:
        assert abs(float(r['rate'])-raw_lookup[(r['method'],r['order'],int(r['checkpoint']),r['dataset'],r['detector'])]) < 1e-12

    audit = dict(bootstrap_resamples=B,bootstrap_seed=SEED,distinct_ordering_models=len(states),sequential_checkpoints=5,
                 generation_batch_sizes={k:v['generation']['batch_size'] for k,v in saved_configs.items()},
                 sequential_final_report_environment=report_metadata['environment'],
                 prompt_counts={ds:len(states['base'][ds]['gen']) for ds in DETS},
                 harmbench_functional_counts=dict(Counter(g['prompt']['functional_category'] for g in states['base']['harmbench']['gen'])),
                 raw_summary_checks='All raw flags match stored summary rates; all prompt records aligned, including across runs.',
                 total_parse_errors=sum(r['parse_errors'] for r in rates),
                 total_empty_skips=sum(r['empty_skips'] for r in rates),
                 model_dataset_evaluations=51*3, generated_responses=51*sum(len(states['base'][ds]['gen']) for ds in DETS),
                 verdict_count=sum(r['n'] for r in rates), witness_order=witness_order, witness=witness,
                 sequential_harmbench={det:[float(flags(s,'harmbench',det).mean()*100) for s in sequential] for det in DETS['harmbench']},
                 ordering_endpoints=endpoint_stats, between_run_base=[r for r in cross if r['checkpoint']==0],
                 equal_5pp_gates=[r for r in gate_rows if r['eps_pp']==5 and r['tau_pp']==5],
                 trajectory_shapes=trajectory)
    write_json('audit.json',audit)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
                         'grid.alpha':.18,'pdf.fonttype':42,'savefig.facecolor':'white'})
    # Fixed sequence, including local and cumulative changes.
    fig, axs = plt.subplots(1,3,figsize=(10.4,2.8),layout='constrained')
    for j,ds in enumerate(DETS):
        for det,color in zip(DETS[ds],COLORS):
            yy = [flags(s,ds,det).mean()*100 for s in sequential]
            axs[j].plot(range(5),yy,'o-',color=color,label=SHORT[det],lw=1.8,ms=4)
        display = {'harmbench':'HarmBench','advbench':'AdvBench','alpaca':'Alpaca'}[ds]
        axs[j].set(title=f'{display} (n={len(sequential[0][ds]["gen"])})',xlabel='Adapters integrated',xticks=range(5))
        axs[j].set_ylabel('FRR (%)' if ds=='alpaca' else 'ASR (%)')
        axs[j].legend(frameon=False,fontsize=7,loc='best')
    savefig('fixed_chain')

    fig,axs=plt.subplots(1,2,figsize=(10.4,3.1),layout='constrained')
    for ax,det in zip(axs,DETS['harmbench']):
        rows=[r for r in bootstrap if r['dataset']=='harmbench' and r['detector']==det and r['comparison'].startswith('sequential: step')]
        for offset,word,color in [(-.1,'vs step',COLORS[0]),(.1,'vs base',COLORS[1])]:
            rr=[r for r in rows if word in r['comparison']]
            y=np.array([r['delta_pp'] for r in rr]);lo=np.array([r['ci_low_pp'] for r in rr]);hi=np.array([r['ci_high_pp'] for r in rr])
            ax.errorbar(np.arange(1,5)+offset,y,yerr=[y-lo,hi-y],fmt='o',capsize=3,color=color,label='Local' if word=='vs step' else 'Cumulative')
        ax.axhline(0,color='#333333',lw=.7)
        ax.set(title=SHORT[det],xticks=range(1,5),xlabel='Integration step',ylabel='HarmBench ASR change (pp)')
        ax.legend(frameon=False,fontsize=8)
    savefig('fixed_regression_ci')

    fig,axs=plt.subplots(2,2,figsize=(10.4,5.5),layout='constrained',sharex=True)
    for row,method in enumerate(['linear','ties']):
        for col,det in enumerate(DETS['harmbench']):
            ax=axs[row,col]
            for pp in paths[method].values():
                ax.plot(range(5),[flags(states[k],'harmbench',det).mean()*100 for k in pp],color=COLORS[col],alpha=.22,lw=.9)
            mins=[]; maxs=[]
            for i in range(5):
                ys=[flags(states[pp[i]],'harmbench',det).mean()*100 for pp in paths[method].values()]
                mins.append(min(ys));maxs.append(max(ys))
            ax.plot(range(5),mins,'o--',color=COLORS[col],lw=1,ms=3)
            ax.plot(range(5),maxs,'o--',color=COLORS[col],lw=1,ms=3)
            ax.set(title=f'{method.upper()} / {SHORT[det]}',ylabel='HarmBench ASR (%)',xticks=range(5),ylim=(14,46))
            if row==1:ax.set_xlabel('Adapters integrated')
    savefig('all_trajectories')

    # A compact matrix gives every permutation equal visibility, even when states are shared.
    fig,axs=plt.subplots(1,2,figsize=(10.4,6.0),layout='constrained',sharey=True)
    order_list=list(paths['ties'])
    for ax,det in zip(axs,DETS['harmbench']):
        vals=np.array([[flags(states[key],'harmbench',det).mean()*100 for key in paths['ties'][o]] for o in order_list])
        im=ax.imshow(vals,aspect='auto',vmin=15,vmax=33,cmap='YlOrRd')
        ax.set(xticks=range(5),xticklabels=['Base','1','2','3','4'],yticks=range(24),yticklabels=[o.replace('a','').replace('>','-') for o in order_list],title=SHORT[det],xlabel='TIES checkpoint')
        ax.tick_params(axis='y',labelsize=7)
        for i in range(24):
            for j in range(5):ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=6.8,color='white' if vals[i,j]>26 else '#222222')
    fig.colorbar(im,ax=axs,label='HarmBench ASR (%)',shrink=.8)
    savefig('ties_heatmap')

    fig,axs=plt.subplots(1,2,figsize=(10.4,3.5),layout='constrained')
    for ax,det in zip(axs,DETS['harmbench']):
        rows=[r for r in bootstrap if r['dataset']=='harmbench' and r['detector']==det and r['comparison'].startswith('ties: endpoint')]
        rows.sort(key=lambda r:r['delta_pp'])
        y=np.array([r['delta_pp'] for r in rows]);lo=np.array([r['ci_low_pp'] for r in rows]);hi=np.array([r['ci_high_pp'] for r in rows])
        ax.errorbar(y,np.arange(len(y)),xerr=[y-lo,hi-y],fmt='o',ms=3,color=COLORS[DETS['harmbench'].index(det)],capsize=2)
        ax.axvline(0,color='#333333',lw=.8)
        ax.set(yticks=range(12),yticklabels=[str(i+1) for i in range(12)],xlabel='Endpoint ASR minus base (pp)',ylabel='Endpoint rank within detector',title=SHORT[det])
    savefig('ties_endpoint_ci')

    fig,axs=plt.subplots(1,2,figsize=(10.4,2.9),layout='constrained')
    for ax,det in zip(axs,DETS['harmbench']):
        r=[r for r in last_rows if r['detector']==det]
        yy=np.array([x['mean_pct'] for x in r]);lo=np.array([x['min_pct'] for x in r]);hi=np.array([x['max_pct'] for x in r])
        ax.errorbar(range(1,5),yy,yerr=[yy-lo,hi-yy],fmt='o',capsize=5,color=COLORS[DETS['harmbench'].index(det)])
        ax.axhline(flags(states['base'],'harmbench',det).mean()*100,ls='--',color='#666666',lw=1,label='Base')
        ax.set(xticks=range(1,5),xticklabels=['A1','A2','A3','A4'],xlabel='Adapter merged last',ylabel='Endpoint ASR (%)',title=SHORT[det])
        ax.legend(frameon=False,fontsize=8)
    savefig('ties_last_adapter')

    fig,axs=plt.subplots(1,2,figsize=(10.4,2.8),layout='constrained')
    for ax,det in zip(axs,DETS['harmbench']):
        for offset,method,color in [(-.18,'linear',COLORS[0]),(.18,'ties',COLORS[1])]:
            rr=[r for r in categories if r['detector']==det and r['field']=='functional_category' and r['method']==method]
            ax.bar(np.arange(2)+offset,[r['delta_pp'] for r in rr],width=.34,label=method.upper(),color=color)
        ax.axhline(0,color='#333333',lw=.7)
        ax.set(xticks=range(2),xticklabels=['Contextual (100)','Standard (200)'],ylabel='ASR change from base (pp)',title=SHORT[det])
        ax.legend(frameon=False,fontsize=8)
    savefig('functional_categories')

    if witness_order:
        fig,axs=plt.subplots(1,2,figsize=(10.4,2.9),layout='constrained')
        pp=paths['ties'][witness_order]
        for ax,det in zip(axs,DETS['harmbench']):
            rr=np.array([flags(states[key],'harmbench',det).mean()*100 for key in pp])
            ax.bar(np.arange(1,5)-.17,np.diff(rr),width=.33,label='Local',color=COLORS[0])
            ax.bar(np.arange(1,5)+.17,rr[1:]-rr[0],width=.33,label='Cumulative',color=COLORS[1])
            ax.axhline(5,ls='--',color='#333333',label='5 pp gate')
            ax.axhline(0,color='#444444',lw=.7)
            ax.set(title=SHORT[det],xlabel='Integration step',xticks=range(1,5),ylabel='ASR increase (pp)')
            ax.legend(frameon=False,fontsize=7)
        savefig('gate_example')

    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)}
    write_json('input_sha256.json',manifest)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    main()
