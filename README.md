# EKS Production Kit — Free RAG Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![GitHub stars](https://img.shields.io/github/stars/smobinulla/rag-demo-kit?style=social)
![Run: docker compose](https://img.shields.io/badge/run-docker%20compose-2496ED?logo=docker&logoColor=white)

**Upload a document, ask questions, get cited answers — a production-minded RAG
service you can run in 5 minutes with `docker compose`.** Hybrid retrieval
(pgvector + BM25, fused via Reciprocal Rank Fusion), incremental indexing, and
bring-your-own-LLM answers (Claude / OpenAI / Gemini / Bedrock). Embeddings run
locally, so it works with **no API key at all**.

> ⭐ **Star the repo** to follow the full production kit — Terraform for EKS,
> hardened Helm chart, multi-env, drift/import — **coming soon**.

<!-- Demo: record a ~15s screen capture (upload a doc → ask → cited answer),
     save it as docs/demo.gif, then uncomment the line below. -->
<!-- ![RAG demo](docs/demo.gif) -->

This is the **free demo** from the EKS Production Kit — genuinely useful on its
own, and a taste of the code quality in the full kit.

## ▶️ Quick start (no cloud, no API key)

```bash
cd apps/rag-demo
docker compose up --build      # ~130MB embed model downloads on first run
# open http://localhost:8000  — upload a document and ask questions
```

Everything runs locally: embeddings are computed on-device (free, no key), and
you get **hybrid retrieval (vector + BM25) with Reciprocal Rank Fusion**,
**incremental indexing**, and **document versioning**. Want written answers? Plug
in **any** LLM key you already have (Claude, OpenAI, Gemini, Bedrock) — see
[apps/rag-demo/README.md](apps/rag-demo/README.md).

## What's in the free demo

- ✅ The full RAG service (upload → embed → hybrid retrieve → optional cited answer)
- ✅ Runs locally via `docker compose`
- ✅ Bring-your-own-LLM (any provider) or retrieval-only with no key
- ✅ Production-minded engineering: multi-stage image, health/readiness probes,
  structured logs, self-healing schema, non-root/read-only container

## 🚀 Upgrade to the full kit

The full **EKS Production Kit** takes this from "runs on my laptop" to
"running securely on real EKS," and is what most teams actually need:

| | Free demo (this repo) | Full kit (paid) |
|---|:---:|:---:|
| RAG demo app | ✅ | ✅ |
| Run locally (docker compose) | ✅ | ✅ |
| **Hardened Helm chart** (zero-downtime, HPA, PDB, NetworkPolicy, PodSecurity, ALB) | | ✅ |
| **Terraform** for EKS: VPC, cluster, **dev / sit / prod** with isolated state | | ✅ |
| **IRSA**, ECR, RDS/Aurora + pgvector wiring | | ✅ |
| **Drift detection, import & refresh** workflows | | ✅ |
| Guided **deploy / destroy** wizard (bash + PowerShell) | | ✅ |
| Runbooks, testing guide, and updates | | ✅ |

**The full kit is in the works.** ⭐ **Star / watch this repo** to be notified
when it launches — or [open an issue](https://github.com/smobinulla/rag-demo-kit/issues)
to tell me what you'd want included.

## License

The demo in this repository is released under the [MIT License](LICENSE).
