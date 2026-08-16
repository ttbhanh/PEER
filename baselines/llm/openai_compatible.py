from __future__ import annotations
import json,os,re

def select_with_llm(case,candidates,k,model,base_url=None,api_key=None):
    try: from openai import OpenAI
    except ImportError as e: raise RuntimeError("Install: pip install -e '.[llm]'") from e
    client=OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY','EMPTY'),base_url=base_url or os.getenv('OPENAI_BASE_URL'))
    payload=[{'id':str(r.sentence_id),'text':str(r.text)} for _,r in candidates.iterrows()]
    prompt=(f"Select exactly {k} sentence IDs that best anticipate the user's interests for the target item. "
            f"User aspects: {case.get('user_aspects',[])}. Metadata aspects: {case.get('metadata_aspects',[])}. "
            "Avoid redundancy and irrelevant aspects. Return JSON only as {\"ids\":[...]}. Candidates: "+json.dumps(payload,ensure_ascii=False))
    out=client.chat.completions.create(model=model,messages=[{'role':'user','content':prompt}],temperature=0).choices[0].message.content or ''
    m=re.search(r'\{.*\}',out,re.S)
    if not m: raise ValueError('Non-JSON LLM output: '+out[:300])
    return [str(x) for x in json.loads(m.group(0)).get('ids',[])][:k]
