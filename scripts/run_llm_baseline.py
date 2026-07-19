#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,pandas as pd
from peer.utils import write_jsonl
from peer.selectors import resolve_k
from baselines.common.io import to_list
from baselines.llm.openai_compatible import select_with_llm

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pairs-dir',default='data/processed/pairs'); p.add_argument('--split',default='test'); p.add_argument('--model',required=True); p.add_argument('--base-url'); p.add_argument('--api-key'); p.add_argument('--prefilter',type=int,default=30); p.add_argument('--k',default='user_avg'); p.add_argument('--max-cases',type=int,default=0); p.add_argument('--output',default='outputs/predictions/llm_test.jsonl'); a=p.parse_args()
    df=pd.read_parquet(Path(a.pairs_dir)/f'{a.split}.parquet'); rows=[]
    for ci,((cid,ds),g) in enumerate(df.groupby(['case_id','dataset'],sort=False)):
        if a.max_cases and ci>=a.max_cases: break
        g=g.copy(); g['_pre']=.5*g.user_sem_sim+.5*g.metadata_sem_sim; cand=g.nlargest(a.prefilter,'_pre'); f=g.iloc[0]; kval='user_avg' if a.k in {'user','user_avg'} else int(a.k); k=resolve_k(kval,int(f.user_avg_k),len(cand)); ids=select_with_llm(f.to_dict(),cand,k,a.model,a.base_url,a.api_key); chosen=cand[cand.sentence_id.astype(str).isin(ids)].copy(); chosen['_o']=chosen.sentence_id.astype(str).map({v:i for i,v in enumerate(ids)}); chosen=chosen.sort_values('_o')
        rows.append({'case_id':str(cid),'dataset':str(ds),'method':'llm_zero_shot_select','k':str(a.k),'selected_sentence_ids':chosen.sentence_id.astype(str).tolist(),'selected_texts':chosen.text.astype(str).tolist(),'selected_aspects':[z for v in chosen.aspects for z in to_list(v)],'ground_truth_text':str(f.ground_truth_text),'gt_aspects':to_list(f.gt_aspects),'user_aspects':to_list(f.user_aspects),'user_avg_k':int(f.user_avg_k)})
    write_jsonl(rows,a.output); print(f'Wrote {len(rows)} -> {a.output}')
if __name__=='__main__': main()
