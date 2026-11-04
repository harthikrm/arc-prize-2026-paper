import pandas as pd, numpy as np, json, sys
D='/tmp/claude-0/-home-claude/3912e017-035b-5620-b088-26c17838d5e9/scratchpad/data/'
s0=pd.read_csv(D+'s0.csv'); s1=pd.read_csv(D+'s1.csv')
for name,df in [('s0',s0),('s1',s1)]:
    print(name, df.shape, df.source.value_counts().to_dict())
    print(df.groupby('source').greedy_exact.mean())
# combine mega
m=pd.concat([s0[s0.source=='mega'].assign(k='s0_'+s0.key), s1[s1.source=='mega'].assign(k='s1_'+s1.key)])
ev=s0[s0.source=='eval']; pt=s0[s0.source=='pubtrain']
print('mega n',len(m),'eval n',len(ev),'pubtrain n',len(pt))
def q(x,p): 
    xs=sorted(x); return xs[min(len(xs)-1,int(len(xs)*p))]
for nm,x in [('mega',m.nll_mean),('eval',ev.nll_mean),('pubtrain',pt.nll_mean)]:
    print(nm, 'quantiles 5/25/50/75/90/95:', [round(q(x,p),4) for p in (.05,.25,.5,.75,.9,.95)], 'mean',round(x.mean(),4))
thr=q(ev.nll_mean,0.25); cap=q(ev.nll_mean,0.95)
print('thr',thr,'cap',cap)
band=m[(m.greedy_exact==False)&(m.nll_mean>=thr)&(m.nll_mean<=cap)]
print('band',len(band),'solved',int(m.greedy_exact.sum()),'below thr unsolved',int(((m.greedy_exact==False)&(m.nll_mean<thr)).sum()),'above cap',int((m.nll_mean>cap).sum()))
print('greedy exact: mega',m.greedy_exact.mean(),'eval',ev.greedy_exact.mean(),'pubtrain',pt.greedy_exact.mean())
print('frac_tokens_correct mean: mega',m.frac_tokens_correct.mean(),'eval',ev.frac_tokens_correct.mean())
print('n_ans_tokens median mega',m.n_ans_tokens.median(),'eval',ev.n_ans_tokens.median())
print('eval n_train_used', ev.n_train_used.describe())
# eval ids: 120 tasks 172 outputs?
print('eval keys',ev.key.nunique(), ev.key.str.split('_').str[0].nunique())
print('pubtrain keys',pt.key.nunique(), pt.key.str.split('_').str[0].nunique())
# compare s1 eval rows identical to s0?
e1=s1[s1.source=='eval']; print('s1 eval n',len(e1), 'max diff', np.abs(e1.set_index('key').nll_mean - ev.set_index('key').nll_mean).max() if len(e1) else None)
m.to_csv(D+'mega_all.csv',index=False); ev.to_csv(D+'eval.csv',index=False); pt.to_csv(D+'pubtrain.csv',index=False)
