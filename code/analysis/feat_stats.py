import pandas as pd, numpy as np
D='/tmp/claude-0/-home-claude/3912e017-035b-5620-b088-26c17838d5e9/scratchpad/data/'
s0=pd.read_csv(D+'s0.csv'); f=pd.read_csv(D+'mega_s0_features.csv'); f['key']=f.key.str[:-2]
m=s0[s0.source=='mega'].merge(f,on='key')
ev=s0[s0.source=='eval']; ef=pd.read_csv(D+'eval_features.csv'); 
# eval keys like id_0 / id_1 -> features keyed id_0 ; map by task id
ef['tid']=ef.key.str[:-2]; ev=ev.assign(tid=ev.key.str.rsplit('_',n=1).str[0]).merge(ef.drop(columns='key'),on='tid')
pt=s0[s0.source=='pubtrain']; pf=pd.read_csv(D+'pubtrain_features.csv'); pf['tid']=pf.key.str[:-2]
pt=pt.assign(tid=pt.key.str.rsplit('_',n=1).str[0]).merge(pf.drop(columns='key'),on='tid')
thr,cap=0.0323580503463745,0.3264590799808502
m['cat']=np.select([m.greedy_exact==True, m.nll_mean<thr, m.nll_mean>cap],['solved','easy','extreme'],'band')
print(m.cat.value_counts())
cols=['n_demos','in_cells','in_colours','out_cells','same_shape','n_ans_tokens','frac_tokens_correct']
print(m.groupby('cat')[cols].agg(['mean','median']).round(2).T)
print('eval'); print(ev[cols].agg(['mean','median']).round(2).T)
print('pubtrain'); print(pt[cols].agg(['mean','median']).round(2).T)
# greedy-exact rate vs nll bins on mega
bins=[0,0.005,0.01,0.02,0.0324,0.05,0.1,0.2,0.3265,0.5,1,10]
m['bin']=pd.cut(m.nll_mean,bins); print(m.groupby('bin',observed=True).agg(n=('key','size'),solved=('greedy_exact','mean'),ftc=('frac_tokens_correct','mean')).round(3))
ev['bin']=pd.cut(ev.nll_mean,bins); print(ev.groupby('bin',observed=True).agg(n=('key','size'),solved=('greedy_exact','mean'),ftc=('frac_tokens_correct','mean')).round(3))
# per-token nll vs colours on eval
print(ev.groupby(pd.cut(ev.in_colours,[0,4,6,8,10]),observed=True).nll_mean.agg(['size','mean','median']).round(3))
print(m.groupby(pd.cut(m.in_colours,[0,4,6,8,10]),observed=True).nll_mean.agg(['size','mean','median']).round(3))
