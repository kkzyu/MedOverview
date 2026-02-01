import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

from deepseek_client import DeepSeekClient, DeepSeekConfig


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"


def load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_doc_text(p: dict[str, Any]) -> str:
    kw = " ".join(p.get("keywords") or [])
    tri = p.get("triple") or {}
    return (
        f"Title: {p.get('title','')}\n"
        f"Venue: {p.get('venue','')} {p.get('year','')}\n"
        f"Keywords: {kw}\n"
        f"Method: {tri.get('method','')}\n"
        f"Result: {tri.get('result','')}\n"
        f"Contribution: {tri.get('contribution','')}\n"
        f"Abstract: {p.get('abstract','')}\n"
    )


load_dotenv()
app = FastAPI(title="Medical Topic Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


# in-memory index
_papers: list[dict[str, Any]] = []
_paper_by_id: dict[str, dict[str, Any]] = {}
_doc_ids: list[str] = []
_doc_emb: np.ndarray | None = None
_st_model: SentenceTransformer | None = None
_deepseek: DeepSeekClient | None = None


@app.on_event("startup")
def _startup() -> None:
    global _papers, _paper_by_id, _doc_ids, _doc_emb, _st_model, _deepseek

    papers_path = DATA_DIR / "papers.json"
    if not papers_path.exists():
        return

    _papers = load_json(papers_path)
    _paper_by_id = {p["id"]: p for p in _papers}
    _doc_ids = [p["id"] for p in _papers]

    model_name = os.environ.get(
        "EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    _st_model = SentenceTransformer(model_name)

    docs = [build_doc_text(p) for p in _papers]
    emb = _st_model.encode(docs, batch_size=64, show_progress_bar=False)
    _doc_emb = l2_normalize(np.asarray(emb, dtype=np.float32))

    # DeepSeek optional (only needed for /api/ask)
    try:
        _deepseek = DeepSeekClient.from_env()
    except Exception:
        _deepseek = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "papers": len(_papers),
        "has_embeddings": _doc_emb is not None,
        "has_deepseek": _deepseek is not None,
    }


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str) -> dict[str, Any]:
    p = _paper_by_id.get(paper_id)
    if not p:
        raise HTTPException(404, "paper not found")
    return p


@app.get("/api/search")
def search(q: str = Query("", min_length=0), k: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    q = (q or "").strip()
    if not q:
        return {"results": _papers[:k]}

    if _st_model is None or _doc_emb is None:
        raise HTTPException(500, "embeddings not ready")

    q_emb = _st_model.encode([q], show_progress_bar=False)
    q_emb = l2_normalize(np.asarray(q_emb, dtype=np.float32))[0]

    sims = _doc_emb @ q_emb
    idx = np.argsort(-sims)[:k]

    results = []
    for i in idx:
        pid = _doc_ids[int(i)]
        p = _paper_by_id.get(pid)
        if not p:
            continue
        results.append({"paper": p, "score": float(sims[int(i)])})

    return {"query": q, "results": results}


@app.post("/api/ask")
def ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if _st_model is None or _doc_emb is None:
        raise HTTPException(500, "embeddings not ready")

    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "missing question")

    style = (payload.get("style") or "overview").strip().lower()
    if style not in {"overview", "cite"}:
        style = "overview"

    top_k = int(payload.get("top_k") or 8)
    top_k = max(3, min(12, top_k))

    # BYOK: use request api_key if provided (do NOT store on server)
    request_api_key = (payload.get("api_key") or "").strip()
    request_model = (payload.get("model") or "").strip()
    if request_api_key:
        deepseek = DeepSeekClient(DeepSeekConfig(api_key=request_api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), model=request_model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")))
    else:
        if _deepseek is None:
            raise HTTPException(400, "DEEPSEEK_API_KEY not configured on server, and request api_key is empty")
        deepseek = _deepseek

    restrict_ids = payload.get("paper_ids")
    if isinstance(restrict_ids, list) and restrict_ids:
        restrict_ids = [x for x in restrict_ids if isinstance(x, str) and x in _paper_by_id]
    else:
        restrict_ids = None

    # retrieve
    q_emb = _st_model.encode([question], show_progress_bar=False)
    q_emb = l2_normalize(np.asarray(q_emb, dtype=np.float32))[0]

    if restrict_ids is None:
        sims = _doc_emb @ q_emb
        idx = np.argsort(-sims)[:top_k]
        picked = [(_doc_ids[int(i)], float(sims[int(i)])) for i in idx]
    else:
        id_to_pos = {pid: i for i, pid in enumerate(_doc_ids)}
        sims = []
        for pid in restrict_ids:
            pos = id_to_pos.get(pid)
            if pos is None:
                continue
            sims.append((pid, float(_doc_emb[pos] @ q_emb)))
        sims.sort(key=lambda x: -x[1])
        picked = sims[:top_k]

    contexts = []
    for pid, score in picked:
        p = _paper_by_id[pid]
        tri = p.get("triple") or {}
        contexts.append(
            {
                "id": pid,
                "score": score,
                "title": p.get("title"),
                "venue": p.get("venue"),
                "year": p.get("year"),
                "url": p.get("url"),
                "keywords": p.get("keywords"),
                "method": tri.get("method"),
                "result": tri.get("result"),
                "contribution": tri.get("contribution"),
                "abstract": p.get("abstract"),
            }
        )

    system = (
        "You are a precise research assistant. "
        "Answer based ONLY on the provided paper contexts. "
        "If evidence is insufficient, say so. "
        "Return JSON only."
    )

    user: dict[str, Any] = {
        "question": question,
        "style": style,
        "papers": contexts,
    }
    if style == "cite":
        user["output_schema"] = {
            "answer_cn": "string",
            "key_points": "list of short strings (optional)",
            "citations": "list of paper ids used",
        }
    else:
        user["output_schema"] = {
            "answer_cn": "string",
            "key_points": "list of short strings (optional)",
            "representative_papers": "list of paper ids (optional)",
        }

    import json

    resp = deepseek.chat_json(
        system=system,
        user=json.dumps(user, ensure_ascii=False),
        max_tokens=900,
        temperature=0.2,
    )

    citations: list[str] = []
    if style == "cite":
        raw = resp.get("citations")
        if isinstance(raw, list):
            citations = [c for c in raw if isinstance(c, str) and c in _paper_by_id]
    else:
        raw = resp.get("representative_papers")
        if isinstance(raw, list):
            citations = [c for c in raw if isinstance(c, str) and c in _paper_by_id]

    return {
        "question": question,
        "used_style": style,
        "answer": resp.get("answer_cn") or "",
        "key_points": resp.get("key_points") if isinstance(resp.get("key_points"), list) else [],
        "citations": citations,
        "retrieved": contexts,
    }
