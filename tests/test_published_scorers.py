import pandas as pd
from baselines.published.scorers import SCORERS

def test_scorers():
    d={k:[.1,.8,.4] for k in ['user_aspect_overlap','item_aspect_salience','metadata_aspect_overlap','user_sem_sim','metadata_sem_sim','item_sem_sim','helpfulness_norm','recency_norm','sentiment_match']}; d['review_id']=['r1','r1','r2']; d['sentence_id']=['s1','s2','s3']; g=pd.DataFrame(d)
    for n,f in SCORERS.items(): assert len(f(g))==3,n
