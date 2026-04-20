from src.utils.redis_db import r, norm, h, k, dump, load
import orjson

EMBED_TTL = 24 * 3600 
RETR_TTL = 3600 
LLM_TTL = 3 * 3600  
PROMPT_VERSION = "p1"

def get_embedding(query, embed_model):
    qn = norm(query)
    key = k("embed", h(qn))

    cached = r.get(key)
    if cached:
        return load(cached)

    vec = embed_model.get_query_embedding(qn)
    r.set(key, dump(vec), ex=EMBED_TTL)
    return vec

def get_retrieval(query, retriever):
    qn = norm(query)
    key = k("retr", h(qn))

    cached = r.get(key)
    if cached:
        return load(cached)

    nodes = retriever.retrieve(qn)

    data = [
        {
            "id": n.node.node_id,
            "text": n.node.text,
            "score": float(getattr(n, "score", 0.0)),
        }
        for n in nodes
    ]

    r.set(key, dump(data), ex=RETR_TTL)
    return data

async def get_llm_response_async(query, query_engine, filters=None):
    qn = norm(query)
    fhash = h(orjson.dumps(filters).decode() if filters else "")
    key = k("llm", PROMPT_VERSION, h(qn), fhash)

    cached = r.get(key)
    if cached:
        return load(cached)

    resp = await query_engine.aquery(qn) 
    out = getattr(resp, "response", None) or str(resp)

    r.set(key, dump(out), ex=LLM_TTL)
    return out