import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI

# ===== Config =====
COLLECTION = "resumos_liz"
EMBED_MODEL = "text-embedding-3-large"  # 3072 dims
TOP_K_DEFAULT = 10

# ===== Init =====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not (OPENAI_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Faltam variáveis no .env: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI(title="Liz • Busca de Artigos", version="1.0.0")


# ===== Models =====
class SearchRequest(BaseModel):
    q: str
    top_k: Optional[int] = TOP_K_DEFAULT


class SearchItem(BaseModel):
    id: str
    titulo: str
    score: float


class SearchResponse(BaseModel):
    total_encontrados: int
    resultados: List[SearchItem]
    pergunta: str


# ===== Utils =====
def embed_text(text: str) -> list:
    emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    return emb


def formatar_resposta_liz(pergunta: str, itens: List[SearchItem]) -> str:
    # System prompt simples para tom da Liz
    system = (
        "Você é a Liz, assistente precisa e educada da OCESP. "
        "Responda em português do Brasil. "
        "Liste os artigos encontrados com ID e Título, em ordem de relevância. "
        "Se nada for encontrado, sugira reformular a pergunta de forma clara."
    )

    if not itens:
        return "Não encontrei artigos relevantes para essa busca. Tente reformular com termos mais específicos."

    # Monta uma lista textual com ID + Título
    lista = "\n".join([f"- {it.id} — {it.titulo}" for it in itens])

    user = (
        f"Pergunta do usuário: {pergunta}\n"
        f"Artigos encontrados (ID — Título):\n{lista}\n\n"
        "Formate uma resposta breve no seu tom, sem inventar nada além do que está na lista."
    )

    msg = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return msg.choices[0].message.content.strip()


# ===== Endpoints =====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    q = req.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    # 1) Embedding da pergunta
    query_vec = embed_text(q)

    # 2) Busca no Qdrant
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=max(1, min(req.top_k or TOP_K_DEFAULT, 25)),
        with_payload=True,
        score_threshold=None,  # ajustar threshold depois se quiser
    )

    resultados: List[SearchItem] = []
    for h in hits:
        payload = h.payload or {}
        resultados.append(
            SearchItem(
                id=str(payload.get("id", "")),
                titulo=str(payload.get("titulo", "")),
                score=float(h.score if h.score is not None else 0.0),
            )
        )

    # 3) (Opcional) Filtrar vazios
    resultados = [r for r in resultados if (r.id or r.titulo)]

    # 4) Montar resposta final
    resposta_liz = formatar_resposta_liz(q, resultados)

    return SearchResponse(
        total_encontrados=len(resultados),
        resultados=resultados,
        pergunta=q,
    )

@app.post("/search_liz")
def search_liz(req: SearchRequest):
    # Variante que devolve a resposta já “escrita” pela Liz (texto pronto)
    q = req.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    query_vec = embed_text(q)
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=max(1, min(req.top_k or TOP_K_DEFAULT, 25)),
        with_payload=True,
    )

    resultados = []
    for h in hits:
        payload = h.payload or {}
        resultados.append(
            SearchItem(
                id=str(payload.get("id", "")),
                titulo=str(payload.get("titulo", "")),
                score=float(h.score if h.score is not None else 0.0),
            )
        )

    texto = formatar_resposta_liz(q, resultados)
    return {"resposta": texto, "itens": [r.model_dump() for r in resultados], "pergunta": q}
