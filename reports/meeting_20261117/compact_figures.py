"""Paper-sized figures drawn from the checked result CSVs; no model loading."""
from pathlib import Path
import os
import csv
import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parents[1]/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rows(name):
    return list(csv.DictReader((HERE/'data'/name).open()))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,
                     'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})
C=['#21618C','#CB6336']
D=['llama_guard_3','harm_cls']
LABEL=['Llama Guard','HarmCLS']
r=rows('sequential_regressions.csv')
fig,ax=plt.subplots(figsize=(3.32,1.95),layout='constrained')
for d,c,label in zip(D,C,LABEL):
    yy=[float(x['rate_pct']) for x in r if x['dataset']=='harmbench' and x['detector']==d]
    ax.plot(range(5),yy,'o-',color=c,label=label,lw=1.6,ms=3)
ax.set(xlabel='Adapters integrated (fixed order 1-2-3-4)',ylabel='HarmBench ASR (%)',xticks=range(5))
ax.legend(frameon=False,fontsize=7,loc='upper left')
fig.savefig(HERE/'figures/compact_fixed.pdf',bbox_inches='tight');plt.close(fig)

r=rows('all_path_regressions.csv')
fig,axs=plt.subplots(2,2,figsize=(7.05,2.95),layout='constrained',sharex=True,sharey=True)
for i,m in enumerate(['linear','ties']):
    for j,d in enumerate(D):
        ax=axs[i,j]
        rr=[x for x in r if x['method']==m and x['dataset']=='harmbench' and x['detector']==d]
        orders=list(dict.fromkeys(x['order'] for x in rr))
        curves=np.array([[float(x['rate_pct']) for x in rr if x['order']==o] for o in orders])
        for curve in curves:ax.plot(range(5),curve,color=C[j],alpha=.22,lw=.7)
        ax.plot(range(5),curves.min(axis=0),'o--',color=C[j],lw=.8,ms=2)
        ax.plot(range(5),curves.max(axis=0),'o--',color=C[j],lw=.8,ms=2)
        ax.set(title=f'{m.upper()} / {LABEL[j]}',xticks=range(5),ylim=(14,46))
        if j==0:ax.set_ylabel('ASR (%)')
        if i==1:ax.set_xlabel('Adapters integrated')
fig.savefig(HERE/'figures/compact_ordering.pdf',bbox_inches='tight');plt.close(fig)

rr=[x for x in r if x['method']=='ties' and x['order']=='a3>a4>a1>a2' and x['dataset']=='harmbench' and x['detector']==D[0]]
fig,ax=plt.subplots(figsize=(3.32,1.85),layout='constrained')
rr=rr[1:]
ax.bar(np.arange(1,5)-.17,[float(x['local_pp']) for x in rr],width=.33,color=C[0],label='Local')
ax.bar(np.arange(1,5)+.17,[float(x['cumulative_pp']) for x in rr],width=.33,color=C[1],label='Cumulative')
ax.axhline(5,color='#333333',ls='--',lw=1,label='5 pp threshold')
ax.axhline(0,color='#555555',lw=.5)
ax.set(xlabel='Integration step: TIES order 3-4-1-2',ylabel='ASR increase (pp)',xticks=range(1,5),ylim=(-2,7))
ax.legend(frameon=False,fontsize=7,loc='upper left')
fig.savefig(HERE/'figures/compact_gate.pdf',bbox_inches='tight');plt.close(fig)
print('Wrote three paper-sized vector figures.')
