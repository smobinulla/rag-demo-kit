# EKS Production Kit — Free Demo

A free, runnable taste of the **EKS Production Kit**: a production-grade toolkit
for running real workloads on Amazon EKS.

This public repo ships one thing you can run in five minutes on your laptop — a
genuinely useful **RAG (Retrieval-Augmented Generation) service** — so you can
judge the code quality before buying the full kit.

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

The paid **EKS Production Kit** takes this from "runs on my laptop" to
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

**→ Get the full kit: [add your Gumroad / product link here]**

## License

The demo in this repository is released under the [MIT License](LICENSE).
