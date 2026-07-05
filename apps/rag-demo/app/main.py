"""
RAG service — lightweight.

Pipeline: upload a document -> extract text -> chunk (sentence-aware) -> hash
each chunk -> INCREMENTALLY index (embed only new/changed chunks, reuse
unchanged, drop removed) into Postgres/pgvector. Query does HYBRID retrieval —
semantic (pgvector cosine) candidates unioned with lexical (Postgres full-text)
candidates — then RE-RANKS with BM25 fused with vector similarity via
Reciprocal Rank Fusion.

Embeddings are local and free (fastembed / ONNX) — no key, small footprint.
The optional answer step is provider-agnostic via LiteLLM (bring your own key):
set ANSWER_MODEL to any provider-prefixed model id; without it the service
returns the ranked passages (still a full retrieval demo). This keeps the image
small and cheap to run for everyone.
"""
import hashlib
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import numpy as np
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from pypdf import PdfReader
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rag")

APP_VERSION = os.environ.get("APP_VERSION", "dev")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@db:5432/rag")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.environ.get("FASTEMBED_CACHE", "/app/models")
EMBED_DIM = 384        # BAAI/bge-small-en-v1.5
CANDIDATES = 30        # per-retriever candidate pool before re-ranking
RRF_K = 60             # Reciprocal Rank Fusion constant
# Smaller chunks = one topic per chunk, so retrieval pinpoints the right passage.
CHUNK_TARGET = int(os.environ.get("CHUNK_TARGET", "400"))

# Optional answer synthesis via ANY provider (bring your own key). Set ANSWER_MODEL
# to a provider-prefixed model id and supply that provider's key via env:
#   anthropic/claude-opus-4-8            -> ANTHROPIC_API_KEY
#   openai/gpt-4o                        -> OPENAI_API_KEY
#   gemini/gemini-1.5-pro                -> GEMINI_API_KEY
#   bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 -> AWS creds/role + AWS_REGION_NAME
# Leave ANSWER_MODEL empty (default) for retrieval-only — no key, no cost.
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "").strip()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
state: dict = {}


def _apply_schema(conn) -> None:
    """Create the schema, and self-heal older ones. Using ADD COLUMN IF NOT
    EXISTS (not just CREATE TABLE IF NOT EXISTS) means a database created by an
    earlier version — missing e.g. the `ts` full-text column — is migrated in
    place instead of crashing the app on a stale volume."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        " id SERIAL PRIMARY KEY,"
        " filename TEXT UNIQUE NOT NULL,"
        " version INT NOT NULL DEFAULT 1,"
        " chunk_count INT NOT NULL DEFAULT 0,"
        " updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        " id SERIAL PRIMARY KEY,"
        " document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
        " chunk_index INT NOT NULL,"
        " content TEXT NOT NULL,"
        " content_hash TEXT NOT NULL DEFAULT '',"
        f" embedding vector({EMBED_DIM}) NOT NULL,"
        " version INT NOT NULL DEFAULT 1,"
        " ts tsvector,"
        " UNIQUE (document_id, content_hash))"
    )
    # Self-heal columns that older schemas may lack.
    conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1")
    conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INT NOT NULL DEFAULT 0")
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1")
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS ts tsvector")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                 "ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_ts_idx ON chunks USING gin (ts)")


def init_db(retries: int = 30) -> None:
    """Wait for Postgres to accept connections, then apply/heal the schema."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
                _apply_schema(conn)
            log.info("database ready and schema applied")
            return
        except Exception as e:
            last_err = e
            if attempt == 1 or attempt % 5 == 0:
                log.warning("database not ready (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(1.5)
    raise RuntimeError(f"database not ready after {retries} tries: {last_err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting rag-demo version=%s", APP_VERSION)
    log.info("config: embed_model=%s embed_dim=%d chunk_target=%d answer=%s",
             EMBED_MODEL, EMBED_DIM, CHUNK_TARGET, answer_backend_label())
    init_db()
    log.info("loading embedding model %s (cache=%s) ...", EMBED_MODEL, EMBED_CACHE)
    state["embedder"] = TextEmbedding(model_name=EMBED_MODEL, cache_dir=EMBED_CACHE)
    pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5,
                          configure=register_vector, open=False)
    pool.open()
    state["pool"] = pool
    log.info("startup complete — ready to serve on :8000")
    yield
    log.info("shutting down")
    pool.close()


app = FastAPI(title="RAG demo (hybrid + incremental)", lifespan=lifespan)


# ---- helpers ---------------------------------------------------------------

def embed(texts: list[str]):
    # numpy arrays — pgvector's psycopg adapter dumps these as `vector`.
    return list(state["embedder"].embed(texts))


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, target: int = CHUNK_TARGET, overlap_sentences: int = 1) -> list[str]:
    """Sentence-aware chunking. Group whole sentences up to ~`target` chars so a
    chunk never starts or ends mid-sentence, with a 1-sentence overlap to keep
    context continuous. Much tighter and more topically-coherent than a raw
    fixed-size character window, so retrieved passages are smaller and sharper."""
    text = " ".join(text.split())
    raw = re.split(r"(?<=[.!?])\s+", text)

    # Hard-split any monster "sentence" (e.g. PDFs with no punctuation) so a
    # single blob can't dominate a chunk.
    sentences: list[str] = []
    for s in raw:
        if len(s) <= target * 2:
            sentences.append(s)
        else:
            sentences.extend(s[i:i + target] for i in range(0, len(s), target))

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + len(s) + 1 > target:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


SYSTEM_PROMPT = ("Answer the question using ONLY the numbered context passages. "
                 "Cite passages like [1], [2]. If the context lacks the answer, say so.")


def synthesize_answer(question: str, matches: list[dict]) -> str | None:
    """Optional cited answer from ANY provider via LiteLLM (bring your own key).
    Returns None (retrieval-only) unless ANSWER_MODEL is set — no key, no cost,
    no heavy compute. The provider is inferred from the ANSWER_MODEL prefix and
    its key is read from the standard provider env var."""
    if not matches or not ANSWER_MODEL:
        return None
    try:
        from litellm import completion
        context = "\n\n".join(f"[{i + 1}] {m['content']}" for i, m in enumerate(matches))
        resp = completion(
            model=ANSWER_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Context passages:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        # e.g. missing/invalid key, unknown model, provider error — degrade to
        # retrieval-only with a visible note rather than failing the request.
        return f"(answer synthesis unavailable: {e})"


# ---- models ----------------------------------------------------------------

class Query(BaseModel):
    question: str
    k: int = 3


# ---- endpoints -------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness: process is up. Cheap and dependency-free so a transient DB blip
    never triggers a pod restart (use /ready for dependency checks)."""
    return {"status": "ok", "version": APP_VERSION}


def answer_backend_label() -> str:
    if ANSWER_MODEL:
        return f"llm: {ANSWER_MODEL}"
    return "retrieval-only (set ANSWER_MODEL + provider key for answers)"


@app.get("/ready")
def ready():
    """Readiness: dependencies usable (embedder loaded + DB reachable). Point
    startup/readiness probes here; use /health (below) for liveness."""
    if "pool" not in state or "embedder" not in state:
        raise HTTPException(503, "still starting up")
    try:
        with state["pool"].connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(503, f"database not reachable: {e}")
    return {"status": "ready", "version": APP_VERSION}


@app.get("/stats")
def stats():
    with state["pool"].connection() as conn:
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"documents": docs, "chunks": chunks, "version": APP_VERSION,
            "embed_model": EMBED_MODEL,
            "retrieval": "hybrid (vector + BM25) with RRF re-ranking",
            "answer_backend": answer_backend_label()}


@app.get("/documents")
def list_documents():
    with state["pool"].connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, version, chunk_count, updated_at "
            "FROM documents ORDER BY updated_at DESC"
        ).fetchall()
    return [{"id": r[0], "filename": r[1], "version": r[2],
             "chunks": r[3], "updated_at": r[4].isoformat()} for r in rows]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    with state["pool"].connection() as conn:
        n = conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,)).rowcount
    if not n:
        raise HTTPException(404, "document not found")
    return {"deleted": doc_id}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Incremental indexing: embed only chunks whose content hash is new,
    reuse unchanged chunks, and drop chunks that no longer appear."""
    data = await file.read()
    text = extract_text(file.filename, data)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No extractable text found in the file.")

    # De-dup identical chunks within one document (keep first occurrence).
    seen, ordered = set(), []
    for i, c in enumerate(chunks):
        h = sha(c)
        if h not in seen:
            seen.add(h)
            ordered.append((i, c, h))
    new_hashes = {h for _, _, h in ordered}

    with state["pool"].connection() as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT id, version FROM documents WHERE filename = %s", (file.filename,)
            ).fetchone()
            if row:
                doc_id, version = row[0], row[1] + 1
            else:
                doc_id = conn.execute(
                    "INSERT INTO documents (filename, version, chunk_count) "
                    "VALUES (%s, 1, 0) RETURNING id", (file.filename,)
                ).fetchone()[0]
                version = 1

            existing = dict(conn.execute(
                "SELECT content_hash, id FROM chunks WHERE document_id = %s", (doc_id,)
            ).fetchall())

            to_add = [(i, c, h) for i, c, h in ordered if h not in existing]
            to_remove = [cid for h, cid in existing.items() if h not in new_hashes]

            if to_remove:
                conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (to_remove,))

            if to_add:
                vectors = embed([c for _, c, _ in to_add])
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks "
                        "(document_id, chunk_index, content, content_hash, embedding, version, ts) "
                        "VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('english', %s)) "
                        "ON CONFLICT (document_id, content_hash) DO NOTHING",
                        [(doc_id, i, c, h, v, version, c)
                         for (i, c, h), v in zip(to_add, vectors)],
                    )

            conn.execute(
                "UPDATE documents SET version = %s, chunk_count = %s, updated_at = now() "
                "WHERE id = %s", (version, len(ordered), doc_id)
            )

    result = {
        "filename": file.filename,
        "version": version,
        "chunks_total": len(ordered),
        "chunks_added": len(to_add),         # newly embedded
        "chunks_reused": len(ordered) - len(to_add),  # unchanged, not re-embedded
        "chunks_removed": len(to_remove),    # deleted (content gone)
    }
    log.info("upload %r v%d: total=%d added=%d reused=%d removed=%d",
             file.filename, version, result["chunks_total"], result["chunks_added"],
             result["chunks_reused"], result["chunks_removed"])
    return result


@app.post("/query")
def query(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    qvec = embed([q.question])[0]
    k = max(1, min(q.k, 20))

    with state["pool"].connection() as conn:
        semantic = conn.execute(
            "SELECT c.id, c.content, d.filename, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "ORDER BY c.embedding <=> %s LIMIT %s",
            (qvec, CANDIDATES),
        ).fetchall()
        lexical = conn.execute(
            "SELECT c.id, c.content, d.filename, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.ts @@ websearch_to_tsquery('english', %s) "
            "ORDER BY ts_rank_cd(c.ts, websearch_to_tsquery('english', %s)) DESC LIMIT %s",
            (q.question, q.question, CANDIDATES),
        ).fetchall()

    # Union the two candidate pools by chunk id.
    pool: dict[int, dict] = {}
    for cid, content, filename, emb in semantic + lexical:
        if cid not in pool:
            pool[cid] = {"content": content, "source": filename, "emb": np.asarray(emb, dtype=float)}
    if not pool:
        log.info("query %r: 0 candidates (empty corpus or no match)", q.question[:80])
        return {"question": q.question, "matches": [], "answer": None,
                "retrieval": "hybrid vector+BM25 / RRF",
                "answer_backend": answer_backend_label(), "candidates_considered": 0}

    ids = list(pool)
    embs = np.stack([pool[i]["emb"] for i in ids])
    qn = qvec / (np.linalg.norm(qvec) + 1e-9)
    cos = embs @ qn / (np.linalg.norm(embs, axis=1) + 1e-9)   # cosine similarity per candidate

    bm25 = BM25Okapi([tokenize(pool[i]["content"]) for i in ids])
    bm = np.asarray(bm25.get_scores(tokenize(q.question)), dtype=float)

    # Reciprocal Rank Fusion of the two rankings.
    vec_rank = {ids[idx]: r for r, idx in enumerate(np.argsort(-cos))}
    bm_rank = {ids[idx]: r for r, idx in enumerate(np.argsort(-bm))}
    fused = {i: 1.0 / (RRF_K + vec_rank[i]) + 1.0 / (RRF_K + bm_rank[i]) for i in ids}

    top = sorted(ids, key=lambda i: fused[i], reverse=True)[:k]
    cos_by_id = dict(zip(ids, cos))
    bm_by_id = dict(zip(ids, bm))
    matches = [{
        "content": pool[i]["content"],
        "source": pool[i]["source"],
        "vector_score": round(float(cos_by_id[i]), 4),
        "bm25_score": round(float(bm_by_id[i]), 4),
        "fused_score": round(float(fused[i]), 5),
    } for i in top]

    answer = synthesize_answer(q.question, matches)
    log.info("query %r: %d candidates -> top-%d (best vec=%.3f) answer=%s",
             q.question[:80], len(pool), len(matches),
             matches[0]["vector_score"] if matches else 0.0,
             "yes" if answer else "no")
    return {"question": q.question, "matches": matches,
            "answer": answer,
            "answer_backend": answer_backend_label(),
            "retrieval": "hybrid vector+BM25 / RRF",
            "candidates_considered": len(pool)}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
