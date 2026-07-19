from __future__ import annotations
import numpy as np
import pandas as pd

def c(g,n): return g[n].to_numpy(float) if n in g else np.zeros(len(g))
def a2spr(g): return .40*c(g,'user_aspect_overlap')+.30*c(g,'item_aspect_salience')+.20*c(g,'metadata_aspect_overlap')+.10*c(g,'user_sem_sim')
def erra_r(g): return .30*c(g,'user_sem_sim')+.25*c(g,'metadata_sem_sim')+.20*c(g,'item_sem_sim')+.15*c(g,'user_aspect_overlap')+.10*c(g,'item_aspect_salience')
def prag(g):
    z=np.zeros(len(g))
    for n in ['user_sem_sim','item_sem_sim','metadata_sem_sim','user_aspect_overlap','item_aspect_salience']:
        v=c(g,n); o=np.argsort(-v); r=np.empty(len(g),int); r[o]=np.arange(1,len(g)+1); z+=1/(60+r)
    return z
def narre(g):
    b=.45*c(g,'user_sem_sim')+.25*c(g,'item_sem_sim')+.15*c(g,'helpfulness_norm')+.15*c(g,'recency_norm')
    t=pd.DataFrame({'review_id':g['review_id'] if 'review_id' in g else g.sentence_id,'b':b}); rv=t.groupby('review_id').b.transform('mean').to_numpy(); return .8*rv+.2*b
def hrdr(g):
    s=.35*c(g,'user_sem_sim')+.20*c(g,'metadata_sem_sim')+.20*c(g,'user_aspect_overlap')+.15*c(g,'item_aspect_salience')+.10*c(g,'sentiment_match')
    t=pd.DataFrame({'review_id':g['review_id'] if 'review_id' in g else g.sentence_id,'s':s}); rv=t.groupby('review_id').s.transform('max').to_numpy(); return .55*s+.45*rv
SCORERS={'narre':narre,'hrdr':hrdr,'a2spr':a2spr,'erra_r':erra_r,'prag':prag}
