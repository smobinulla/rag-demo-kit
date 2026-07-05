# RAG demo — hybrid retrieval, incremental indexing, bring-your-own-LLM

A real Retrieval-Augmented Generation service that goes beyond a toy: upload a
document, it's chunked and embedded **locally** (no key), stored in **pgvector**,
and retrieved with **hybrid search (semantic vectors + BM25 keyword) re-ranked
via Reciprocal Rank Fusion**. Re-uploading a document **only re-embeds the
changed chunks**.

The written-answer step is **optional and provider-agnostic** — point it at any
model you have a key for (Claude/Anthropic, OpenAI, Google Gemini, AWS Bedrock).
With no key configured, the demo returns the ranked passages, which is a complete
retrieval demo at **zero cost and a small footprint**.

> This is the **free demo** from the EKS Production Kit. Deploying it to real
> **EKS** — hardened Helm chart, multi-env Terraform, RDS/Aurora + pgvector,
> IRSA, ALB, drift/import — is part of the **[full kit](../../README.md#-upgrade-to-the-full-kit)**.

## Features

- **Local, free embeddings** — `fastembed` running `BAAI/bge-small-en-v1.5`
  (384-dim, ONNX). No API key, no per-query network calls, ~130MB.
- **Hybrid retrieval + BM25 re-ranking** — candidates from *both* pgvector cosine
  similarity *and* Postgres full-text search, re-ranked with BM25 fused via
  **Reciprocal Rank Fusion**. Each result reports `vector_score`, `bm25_score`,
  `fused_score`.
- **Sentence-aware chunking** (~400 chars; tune with `CHUNK_TARGET`).
- **Incremental (delta) indexing** — content-hashed chunks; re-uploading embeds
  only *new* chunks, reuses unchanged, removes gone ones.
- **Versioned documents** + **document management** (list / delete).
- **Bring-your-own-LLM answers (optional)** — set `ANSWER_MODEL` to any
  [LiteLLM](https://docs.litellm.ai/docs/providers) model id + that provider's key.

## Run it (docker compose)

```bash
docker compose up --build        # small image; only the ~130MB embed model downloads
# open http://localhost:8000
docker compose down -v           # tear down incl. data
```

Enable answers (optional) by copying `.env.example` to `.env` and setting a model
+ key, then re-run `docker compose up -d`:

| Provider | `ANSWER_MODEL` | Key |
|---|---|---|
| Anthropic (Claude) | `anthropic/claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/gemini-1.5-pro` | `GEMINI_API_KEY` |
| AWS Bedrock | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | AWS creds / IAM role |

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/upload` | multipart `file` | `{filename, version, chunks_total, chunks_added, chunks_reused, chunks_removed}` |
| `POST` | `/query` | `{"question": "...", "k": 3}` | `{matches:[{content, source, vector_score, bm25_score, fused_score}], answer, answer_backend, ...}` |
| `GET` | `/documents` | — | `[{id, filename, version, chunks, updated_at}]` |
| `DELETE` | `/documents/{id}` | — | `{deleted}` |
| `GET` | `/stats` | — | `{documents, chunks, version, embed_model, retrieval, answer_backend}` |
| `GET` | `/health` | — | `{status, version}` — cheap **liveness** |
| `GET` | `/ready` | — | `{status, version}` or `503` — **readiness** (DB reachable) |

## Try the incremental indexing

```bash
curl -F "file=@sample.txt" http://localhost:8000/upload      # version 1
curl -F "file=@sample.txt" http://localhost:8000/upload      # identical -> 0 re-embedded
# edit sample.txt (append a section), re-upload -> only the new chunk is embedded
```

## Operations

- **Multi-stage image**, structured unbuffered logs, `HEALTHCHECK`, liveness
  (`/health`) vs readiness (`/ready`), self-healing schema, secrets via env only.
- If you ever hit a stale-volume error, reset with `docker compose down -v`.

## Notes / limits

- Embeddings are local; the answer step's quality/latency/cost are the chosen
  provider's. With a tiny corpus, retrieval scores sit close together — load
  several real documents to see the ranking separate.
