"""Generate LaTeX tables directly from the independently checked analysis CSVs."""
from pathlib import Path
import csv
import json

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'
OUT = HERE/'tables'
OUT.mkdir(exist_ok=True)

def rows(name):
    return list(csv.DictReader((DATA/name).open()))

def esc(s):
    return str(s).replace('_', r'\_').replace('>', r'$\to$')

def f(x, signed=False):
    return f'{float(x):+.1f}' if signed else f'{float(x):.1f}'

def interval(r):
    return f"{f(r['delta_pp'], True)} [{f(r['ci_low_pp'], True)}, {f(r['ci_high_pp'], True)}]"

def table(name, spec, headers, body):
    txt = '\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    txt += '\n'.join(' & '.join(map(str,r))+r' \\' for r in body)
    txt += '\n\\bottomrule\n\\end{tabular}\n'
    (OUT/(name+'.tex')).write_text(txt)

DETS = ['llama_guard_3','harm_cls']
SHORT={'llama_guard_3':'Llama Guard','harm_cls':'HarmCLS','rejection_classifier':'Refusal clf.','keyword_refusal':'Keyword'}
DS={'harmbench':'HarmBench','advbench':'AdvBench','alpaca':'Alpaca'}
state_rates=rows('state_rates.csv')
boot=rows('paired_bootstrap.csv')
reg=rows('sequential_regressions.csv')

fixed=[]
for ds in DS:
    dets=DETS if ds!='alpaca' else ['rejection_classifier','keyword_refusal']
    for det in dets:
        rr=[r for r in state_rates if r['experiment']=='sequential' and r['dataset']==ds and r['detector']==det]
        rr.sort(key=lambda r:int(r['state']))
        fixed.append([DS[ds]+' / '+SHORT[det]]+[f(r['rate_pct']) for r in rr])
table('fixed_rates','lrrrrr',['Dataset / detector',r'$M_0$','+A1','+A2','+A3','+A4'],fixed)

changes=[]
for det in DETS:
    for i in range(1,5):
        local=next(r for r in boot if r['detector']==det and r['comparison']==f'sequential: step {i} vs step {i-1}')
        cum=next(r for r in boot if r['detector']==det and r['comparison']==f'sequential: step {i} vs base')
        changes.append([SHORT[det] if i==1 else '',str(i),interval(local),interval(cum)])
table('fixed_changes','llrr',['Detector','Step','Local change [95\\% CI]','Cumulative change [95\\% CI]'],changes)

singles=rows('single_adapters.csv')
body=[]
for a in ['a1','a2','a3','a4']:
    rr=[next(r for r in singles if r['adapter']==a and r['detector']==d) for d in DETS]
    body.append([a.upper()]+[f(r['rate_pct']) for r in rr]+[interval(r) for r in rr])
table('singles','lrrrr',['Adapter','LG ASR','HC ASR','LG change [95\\% CI]','HC change [95\\% CI]'],body)

spread=rows('ordering_spreads.csv')
body=[]
for method in ['linear','ties']:
    for det in DETS:
        rr=[next(r for r in spread if r['method']==method and r['dataset']=='harmbench' and r['detector']==det and int(r['checkpoint'])==i) for i in range(1,5)]
        body.append([method.upper(),SHORT[det]]+[f(r['range_pp']) for r in rr])
table('ordering_spreads','llrrrr',['Method','Detector','Step 1','Step 2','Step 3','Step 4'],body)

ends=rows('endpoints.csv')
body=[]
for ds in DS:
    dets=DETS if ds!='alpaca' else ['rejection_classifier','keyword_refusal']
    for det in dets:
        lin=next(r for r in ends if r['method']=='linear' and r['dataset']==ds and r['detector']==det)
        ties=next(r for r in ends if r['method']=='ties' and r['dataset']==ds and r['detector']==det)
        body.append([DS[ds]+' / '+SHORT[det],f(lin['base_pct']),f(lin['min_pct']),f(ties['min_pct'])+'--'+f(ties['max_pct'])])
table('endpoint_rates','lrrr',['Dataset / detector','Base','LINEAR','TIES range'],body)

body=[]
for det in DETS:
    rr=[next(r for r in boot if r['detector']==det and r['comparison']==label) for label in
        ['linear: endpoint sum(a1,a2,a3,a4) vs base','ties: selected max endpoint vs min endpoint']]
    body.append([SHORT[det]]+[interval(r) for r in rr])
table('endpoint_contrasts','lrr',['Detector','LINEAR final minus base','TIES max minus min*'],body)

gates=rows('gates.csv')
body=[]
for method in ['linear','ties']:
    for det in DETS:
        r=next(r for r in gates if r['method']==method and r['dataset']=='harmbench' and r['detector']==det and float(r['eps_pp'])==5 and float(r['tau_pp'])==5)
        body.append([method.upper(),SHORT[det],r['local_all_pass'],r['cumulative_all_pass'],r['hidden_full_path']])
table('gates_equal','llrrr',['Method','Detector','Local admits','Cumul. admits','Local only*'],body)

audit=json.loads((DATA/'audit.json').read_text())
wr=[r for r in audit['witness'] if r['detector']=='llama_guard_3']
body=[['Base','17.7','--','0.0']]+[[str(r['step']),f(r['asr_pct']),f(r['local_pp'],True),f(r['cumulative_pp'],True)] for r in wr]
table('gate_witness','lrrr',['Checkpoint','ASR (\\%)','Local (pp)','Cumulative (pp)'],body)

cross=rows('between_run_differences.csv')
body=[]
for ds in DS:
    rr=[r for r in cross if r['checkpoint']=='0' and r['dataset']==ds]
    first=rr[0]
    body.append([DS[ds],f"{first['responses_different']}/{first['n']}",
                 f"{rr[0]['verdicts_different']}/{first['n']}",f"{rr[1]['verdicts_different']}/{first['n']}"])
table('cross_run_base','lrrr',['Dataset','Different texts','Detector 1 flips','Detector 2 flips'],body)

allpaths=rows('all_path_regressions.csv')
for method in ['linear','ties']:
    body=[]
    orders=list(dict.fromkeys(r['order'] for r in allpaths if r['method']==method))
    for order in orders:
        line=[order.replace('a','').replace('>', '-')]
        for k in range(1,5):
            rr=[next(r for r in allpaths if r['method']==method and r['order']==order and int(r['checkpoint'])==k and r['dataset']=='harmbench' and r['detector']==d) for d in DETS]
            line.append('/'.join(f(r['rate_pct']) for r in rr))
        body.append(line)
    table('paths_'+method,'lrrrr',['Order','After 1','After 2','After 3','After 4'],body)

plan=json.loads((HERE.parents[1]/'ordering_experiment/results/20261117_045604/plan.json').read_text())
end_keys=sorted({p[-1] for p in plan['paths']['ties'].values()})
body=[]
for key in end_keys:
    order=next(o for o,p in plan['paths']['ties'].items() if p[-1]==key)
    line=[order.replace('a','').replace('>','-')]
    for det in DETS:
        r=next(r for r in boot if r['detector']==det and r['comparison']==f'ties: endpoint {key} vs base')
        sr=next(r for r in state_rates if r['experiment']=='ordering' and r['state']==key and r['dataset']=='harmbench' and r['detector']==det)
        line.extend([f(sr['rate_pct']),interval(r)])
    body.append(line)
table('all_endpoint_ci','lrrrr',['Representative order','LG ASR','LG change [95\\% CI]','HC ASR','HC change [95\\% CI]'],body)

body=[]
for method in ['linear','ties']:
    for det in DETS:
        for eps in [1,2,5]:
            rr=[next(r for r in gates if r['method']==method and r['dataset']=='harmbench' and r['detector']==det and float(r['eps_pp'])==eps and float(r['tau_pp'])==tau) for tau in [2,5,10]]
            body.append([method.upper() if det==DETS[0] and eps==1 else '',SHORT[det] if eps==1 else '',str(eps)]+[r['hidden_full_path'] for r in rr])
table('gate_sweep','llrrrr',['Method','Detector',r'$\epsilon$ (pp)',r'$\tau=2$',r'$\tau=5$',r'$\tau=10$'],body)
print('Wrote',len(list(OUT.glob('*.tex'))),'LaTeX tables')
