import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
D='/tmp/claude-0/-home-claude/3912e017-035b-5620-b088-26c17838d5e9/scratchpad/data/'
O='/tmp/claude-0/-home-claude/3912e017-035b-5620-b088-26c17838d5e9/scratchpad/paper/figures/'
rcParams.update({'font.family':'serif','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'legend.fontsize':7.5,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c3c2b7',
 'axes.linewidth':0.6,'grid.color':'#e1e0d9','grid.linewidth':0.5,'pdf.fonttype':42})
BLUE,ORANGE,AQUA,YELLOW,MAG,GREEN='#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300'
GRAY='#6b6b66'
m=pd.read_csv(D+'mega_all.csv'); ev=pd.read_csv(D+'eval.csv'); pt=pd.read_csv(D+'pubtrain.csv')
thr,cap=0.0323580503463745,0.3264590799808502

# ---------- Fig 1: NLL distributions
fig,ax=plt.subplots(figsize=(3.4,2.3))
bins=np.logspace(-4,1,70)
for df,c,lab in [(m,BLUE,'ARC-Mega (n=114,067)'),(pt,AQUA,'ARC-AGI-2 training (n=200)'),(ev,ORANGE,'ARC-AGI-2 evaluation (n=172)')]:
    x=np.clip(df.nll_mean,1e-4,10)
    ax.hist(x,bins=bins,density=True,histtype='step',lw=1.4,color=c,label=lab)
ax.axvspan(thr,cap,color=ORANGE,alpha=0.10,lw=0)
ax.axvline(thr,color=ORANGE,lw=0.8,ls='--'); ax.axvline(cap,color=ORANGE,lw=0.8,ls='--')
ax.text(np.sqrt(thr*cap),ax.get_ylim()[1]*0.55,'selection\nband',ha='center',va='top',fontsize=7,color='#9a3f17')
ax.set_xscale('log'); ax.set_xlabel('zero-shot per-token NLL of the test output'); ax.set_ylabel('density')
ax.legend(frameon=False,loc='upper right'); ax.grid(axis='y')
fig.tight_layout(); fig.savefig(O+'nll_dist.pdf'); plt.close(fig)

# ---------- Fig 2: partition of the Mega pool + greedy-exact by NLL bin
cat=np.select([m.greedy_exact==True, m.nll_mean<thr, m.nll_mean>cap],['solved','easy','extreme'],'band')
counts=pd.Series(cat).value_counts()
fig,axs=plt.subplots(1,2,figsize=(7.0,2.2),gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]
order=[('solved','solved zero-shot\n→ replay',AQUA),('easy','below band\n(too easy)','#b7d3f6'),('band','in band\n→ train',ORANGE),('extreme','above band\n(unlearnable/short)','#c3c2b7')]
vals=[counts[k] for k,_,_ in order]; labs=[l for _,l,_ in order]; cols=[c for _,_,c in order]
bars=ax.barh(range(4)[::-1],vals,color=cols,height=0.62)
for b,v in zip(bars,vals): ax.text(v+800,b.get_y()+b.get_height()/2,f'{v:,} ({100*v/len(m):.0f}%)',va='center',fontsize=7.5)
ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(labs); ax.set_xlim(0,66000); ax.set_xlabel('tasks (of 114,067 scored)'); ax.grid(axis='x')
ax.set_title('(a) Partition of the scored pool',loc='left')
ax=axs[1]
edges=[0,0.005,0.01,0.02,0.0324,0.05,0.1,0.2,0.3265,0.5,1,10]
g=m.groupby(pd.cut(m.nll_mean,edges),observed=True).agg(n=('key','size'),solved=('greedy_exact','mean'))
x=np.sqrt(np.array(edges[:-1])*np.array(edges[1:]))
ax.plot(x,100*g.solved.values,'-o',color=BLUE,ms=4,lw=1.5)
ax.axvspan(thr,cap,color=ORANGE,alpha=0.10,lw=0)
ax.set_xscale('log'); ax.set_xlabel('per-token NLL (bin, log scale)'); ax.set_ylabel('greedy-exact rate (%)')
ax.set_title('(b) Zero-shot solve rate vs. NLL (Mega)',loc='left'); ax.grid(axis='y')
fig.tight_layout(); fig.savefig(O+'partition.pdf'); plt.close(fig)

# ---------- Fig 3: hidden-test draws
runs=[('Stock NVARC\n(baseline)',[28.89,28.47,28.19],GRAY),('r1\n(+6k band)',[32.22,30.14],ORANGE),('r2\n(+47k band)',[29.58],BLUE)]
fig,ax=plt.subplots(figsize=(3.4,2.1))
for i,(lab,s,c) in enumerate(runs):
    ax.scatter([i]*len(s),s,color=c,s=28,zorder=3)
    mu=np.mean(s); ax.hlines(mu,i-0.22,i+0.22,color=c,lw=2)
    ax.text(i+0.26,mu,f'mean {mu:.1f}',va='center',fontsize=7.5,color='#333')
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in runs]); ax.set_ylabel('hidden-test score (%)'); ax.set_ylim(27,33.5); ax.set_xlim(-0.5,3.0); ax.grid(axis='y')
fig.tight_layout(); fig.savefig(O+'lb_draws.pdf'); plt.close(fig)

# ---------- Fig 4: decoupling
fig,axs=plt.subplots(1,2,figsize=(7.0,2.2))
ax=axs[0]
pts=[('base',0.0734,28.52,GRAY),('r1',0.0603,31.18,ORANGE),('r2',0.0450,29.58,BLUE)]
for lab,l,s,c in pts: ax.scatter(l,s,color=c,s=40,zorder=3); ax.annotate(lab,(l,s),xytext=(5,4),textcoords='offset points',fontsize=8)
ax.plot([p[1] for p in pts],[p[2] for p in pts],color='#c3c2b7',lw=0.8,zorder=1)
ax.invert_xaxis(); ax.set_xlabel('held-out Mega loss (band tasks, 300)  ← better'); ax.set_ylabel('hidden-test score (%)'); ax.set_ylim(27.5,32.5); ax.grid(axis='y')
ax.set_title('(a) In-distribution loss vs. hidden score',loc='left')
ax=axs[1]
w=0.35; x=np.arange(2)
ax.bar(x-w/2,[0.0734,0.0724],w,color='#c3c2b7',label='start (step 0)')
ax.bar(x+w/2,[0.0603,0.0692],w,color=ORANGE,label='r1 end')
ax.bar(x+w/2+w,[0.0450,0.0716],w,color=BLUE,label='r2 end')
for xi,v in zip([x[0]-w/2,x[0]+w/2,x[0]+1.5*w,x[1]-w/2,x[1]+w/2,x[1]+1.5*w],[0.0734,0.0603,0.0450,0.0724,0.0692,0.0716]):
    ax.text(xi,v+0.001,f'{v:.4f}',ha='center',fontsize=6.5)
ax.set_xticks([x[0]+w/2,x[1]+w/2]); ax.set_xticklabels(['held-out Mega loss','ARC-AGI-2 eval loss\n(forgetting monitor)']); ax.set_ylim(0,0.10); ax.set_ylabel('loss (nats / output token)')
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.0),columnspacing=0.8,handlelength=1.0); ax.grid(axis='y')
ax.set_title('(b) Both monitored losses, start → end',loc='left')
fig.tight_layout(); fig.savefig(O+'decoupling.pdf'); plt.close(fig)

# ---------- Fig 5: what the band selects (sample 1, with grid features)
s0=pd.read_csv(D+'s0.csv'); f=pd.read_csv(D+'mega_s0_features.csv'); f['key']=f.key.str[:-2]
mm=s0[s0.source=='mega'].merge(f,on='key')
mm['cat']=np.select([mm.greedy_exact==True, mm.nll_mean<thr, mm.nll_mean>cap],['solved','easy','extreme'],'band')
ef=pd.read_csv(D+'eval_features.csv'); ef['tid']=ef.key.str[:-2]
ee=ev.assign(tid=ev.key.str.rsplit('_',n=1).str[0]).merge(ef.drop(columns='key'),on='tid')
groups=[('below band',mm[mm.cat=='easy'],'#b7d3f6'),('in band',mm[mm.cat=='band'],ORANGE),('above band',mm[mm.cat=='extreme'],'#c3c2b7'),('ARC-AGI-2 eval',ee,GREEN)]
fig,axs=plt.subplots(1,3,figsize=(7.0,2.0))
for ax,(col,lab,fn) in zip(axs,[('n_ans_tokens','median answer tokens',np.median),('same_shape','output same shape as input (%)',lambda s:100*np.mean(s)),('frac_tokens_correct','mean token accuracy (%)',lambda s:100*np.mean(s))]):
    vals=[fn(g[col]) for _,g,_ in groups]
    ax.bar(range(4),vals,color=[c for _,_,c in groups],width=0.62)
    for i,v in enumerate(vals): ax.text(i,v*1.01,f'{v:.0f}',ha='center',fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups],rotation=20,ha='right'); ax.set_title(lab,loc='left'); ax.grid(axis='y')
    ax.set_ylim(0,max(vals)*1.18)
fig.tight_layout(); fig.savefig(O+'band_profile.pdf'); plt.close(fig)
print('ok', counts.to_dict())
