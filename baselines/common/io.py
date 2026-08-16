from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from peer.utils import write_jsonl
from peer.selectors import resolve_k

def to_list(x):
    if x is None: return []
    if isinstance(x,(list,tuple,set,np.ndarray)): return list(x)
    try:
        if pd.isna(x): return []
    except Exception: pass
    return [x]

def emit_predictions(df, method, scorer, k_values, output):
    rows=[]
    for (case_id,dataset),g in df.groupby(['case_id','dataset'],sort=False):
        g=g.copy(); g['_score']=np.asarray(scorer(g),float)
        for kval in k_values:
            k=resolve_k(kval,int(g.iloc[0].get('user_avg_k',3)),len(g))
            chosen=g.sort_values('_score',ascending=False).head(k)
            f=g.iloc[0]
            rows.append({'case_id':str(case_id),'dataset':str(dataset),'method':method,'k':str(kval),
              'selected_sentence_ids':chosen.sentence_id.astype(str).tolist(),
              'selected_texts':chosen.text.astype(str).tolist(),
              'selected_aspects':[a for v in chosen.aspects for a in to_list(v)],
              'selected_evidence':[{'sentence_id':str(r.sentence_id),'text':str(r.text),'score':float(r._score),'rank':j+1,'source_review_id':str(r.get('review_id','')),'aspects':to_list(r.get('aspects'))} for j,(_,r) in enumerate(chosen.iterrows())],
              'ground_truth_text':str(f.ground_truth_text),'gt_aspects':to_list(f.get('gt_aspects')),
              'user_aspects':to_list(f.get('user_aspects')),'user_avg_k':int(f.get('user_avg_k',3))})
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); write_jsonl(rows,output); return rows
