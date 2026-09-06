---
title: Self-Healing RAG
emoji: 🩺
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Self-Healing RAG

A retrieval-augmented generation pipeline that doesn't just retrieve-and-generate — it
critiques its own output and retries. Built with [LangGraph](https://langchain-ai.github.io/langgraph/)
as a stateful, cyclic graph (not a linear chain).

## How it works

```
                 ┌─────────────┐
      ┌─────────▶│  retrieve   │
      │          └──────┬──────┘
      │                 ▼
      │          ┌─────────────┐
      │          │  generate   │
      │          └──────┬──────┘
      │                 ▼
      │          ┌─────────────┐
      │          │  critique   │
      │          └──────┬──────┘
      │                 │
      │      ┌──────────┼───────────┐
      │  grounded   not grounded, │  not grounded,
      │      │      attempts left │  attempts exhausted
      │      ▼          │         ▼
      │  ┌────────┐     │    ┌─────────┐
      │  │finalize│     │    │ refuse  │
      │  └────────┘     │    └─────────┘
      └──────(retry, reformulated query)
```

1. **retrieve** — embed the current query (BGE-M3) and pull the top-k chunks from Chroma.
2. **generate** — an LLM answers the question strictly from those chunks.
3. **critique** — a second LLM call acts as a skeptical fact-checker: is every claim in the
   answer actually traceable to the retrieved chunks (`grounded`), does it invent something
   (`hallucinated`), or does the context simply not cover the question
   (`insufficient_information`)?
4. **route**:
   - `grounded` → done, return the answer.
   - not grounded, and the critic proposed a reformulated query → loop back to `retrieve`
     with that new query.
   - not grounded, and attempts are exhausted → refuse gracefully instead of guessing.
   - not grounded, and the critic deliberately offered *no* reformulation (its signal that
     the knowledge base doesn't cover the topic at all) → refuse immediately, since
     re-running the same query would only retrieve the same chunks.

The retry loop is a real cycle in the graph (`critique → retrieve`), which is why this is
built on LangGraph rather than a linear chain — plain sequential chains can't express
"go back and try again."

## Stack

- **Orchestration**: LangGraph (`StateGraph`)
- **LLM**: provider-pluggable via `LLM_PROVIDER` in `.env` — **Groq**
  (`openai/gpt-oss-120b`, free tier) or **Anthropic** (`claude-opus-5`). The graph,
  prompts, and retrieval are provider-agnostic; only [rag/llm.py](rag/llm.py) knows which
  SDK is in use.
- **Embeddings**: `BAAI/bge-m3` via `sentence-transformers`, local and offline
- **Vector store**: Chroma, persisted to disk

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; use `venv\Scripts\activate` in cmd.exe
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY (free: https://console.groq.com/keys)
```

To run on Claude instead, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`.
No code changes needed.

First run downloads the BGE-M3 model (~2 GB) from HuggingFace.

## Usage

The corpus starts **empty** — everything the pipeline answers comes from documents you
supply. Accepted formats: `.pdf`, `.txt`, `.md`. PDFs must have an embedded text layer
(scanned images are rejected with a clear message; there is no OCR).

Add documents and ask questions:

```bash
python main.py add handbook.pdf notes.md    # embed documents
python main.py list                         # show what's in the corpus
python main.py ask "What is the stipend?" --verbose
python main.py clear                        # empty the corpus
```

`--verbose` prints every retrieve/generate/critique attempt, including the critic's
reasoning and any reformulated query, so you can watch the self-healing loop happen.
`--max-attempts N` caps how many cycles run before the pipeline refuses (default: 3).

`sample_handbook.md` in the project root is a small throwaway file for checking the
pipeline end-to-end. Delete it once you have your own documents.

## Web UI

The CLI hides the thing that makes this project interesting — the retry loop. The web UI
renders it live: the pipeline graph lights up node by node as each LangGraph step runs,
and every pass appears as a trace entry showing the query, retrieved chunks, answer, the
critic's verdict, and the reformulated query that feeds the next pass.

The backend ([server.py](server.py)) streams each node as a server-sent event via
`graph.stream(..., stream_mode="updates")`, so the frontend never waits on a full result.

**Terminal 1 — backend** (loads BGE-M3 once at startup, so give it a few seconds):

```bash
./venv/Scripts/python.exe -m uvicorn server:app --port 8000
```

**Terminal 2 — frontend:**

```bash
cd frontend
npm install     # first time only
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to the backend, so both must be running.
Drag documents onto the corpus panel to populate it. Set **max passes** to 1 to force the
refusal path quickly during a demo.

To capture screenshots for a report (both servers must be up):

```bash
cd frontend
node shot.mjs out.png                       # idle state
node shot.mjs out.png --ask "your question" # runs a query first
```

## Tests

```bash
pytest
```

Tests exercise the graph's retry/accept/refuse routing with fake LLMs and a fake vector
store (`tests/test_graph.py`), so they run without an API key or network access.

## Project layout

```
rag/
  config.py      env-driven settings (provider, models, chunk size, top-k, max attempts)
  llm.py         provider factory — the only module that imports a vendor SDK
  schemas.py     CriticVerdict — the critic's structured output
  state.py       RAGState — the LangGraph state schema
  prompts.py     system prompts + context formatting
  ingest.py      extract/chunk/embed uploaded documents into Chroma
  graph.py       the StateGraph: nodes, routing, retry loop
main.py          CLI: `add`, `list`, `clear`, `ask`
server.py        FastAPI backend streaming each node as an SSE event
frontend/        React + TypeScript + Vite UI
  src/api.ts     SSE client over fetch
  src/App.tsx    loop state machine
  src/components/  GraphDiagram (animated pipeline), CorpusPanel (PDF upload),
                   AttemptCard, ChunkList, FinalAnswer
tests/test_graph.py
```

## Deploy to Hugging Face Spaces

This repository is pre-configured with a multi-stage `Dockerfile` and HF Spaces metadata for 1-click or Git deployment:

### 1. Create a New Space
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Set Space name (e.g. `self-healing-rag`).
3. Select **Docker** as the Space SDK (Blank).
4. Choose **Public** or **Private**.

### 2. Configure Secrets
In your Space's **Settings > Variables and secrets > New secret**:
- Add `GROQ_API_KEY` = your Groq API key (free from [Groq Console](https://console.groq.com/keys))
- *(Optional)* Add `ANTHROPIC_API_KEY` if using Claude (`LLM_PROVIDER=anthropic`).

### 3. Push to Hugging Face Space
Clone or add your Space as a git remote from your local repository:

```bash
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>
git push space main
```

Alternatively, connect your GitHub repository directly in Space Settings to automatically build and deploy on every commit!
