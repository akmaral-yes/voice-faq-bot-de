# Project

Small, time-boxed portfolio prototype: a German customer-service voice FAQ bot for an AI Engineer (Chat/Voicebot) application.
Goals: retrieval-augmented FAQ answering, input guardrails, safe fallback/handoff, answer-quality checks, later ASR/TTS and evaluation.
Deliberately small and interview-explainable, not production-grade.

# Working rules

- Work in explicitly requested stages. Do only the requested stage, then STOP for review.
- Do not silently expand scope, even for possible improvements; mention them instead.
- Report unexpected results honestly; do not hide them or automatically add extra techniques.
- Prefer simple, explicit Python over abstractions. No unrelated refactors.
- No LangChain, web UI, API layer, Docker, or configuration framework unless explicitly requested.
- Use `uv` (`uv run`, `uv add`); Python 3.12.
- Never print, expose, or commit API keys.
- Mask PII before retrieval, LLM calls, or debugging/logging. Show unmasked input only as synthetic/fake data in explicit manual tests.
- Keep experiments (`experiments/`) separate from the production pipeline.
- When a stage is completed, update the Current architecture, Known limitations, and Stage status sections of CLAUDE.md to match the repo.

# Current architecture

- `prepare_data.py`: DFKI / Deutsche Telekom FAQ JSON (`data/raw/`, git-ignored) → `data/corpus.jsonl`
- `build_index.py`: embeds corpus, wipes and rebuilds persistent Chroma index `data/chroma/` (git-ignored)
- `retrieval.py`: `retrieve(query, top_k=3)`; opens existing index, embeds only the query
- `guards.py`: `mask_pii()`, `detect_injection()` (returns list of reasons)
- `llm.py`: `LLMClient` Protocol, `OpenAIClient` (`OPENAI_API_KEY`; model `gpt-4.1-mini`, override via `OPENAI_MODEL`)
- `pipeline.py`: `process_query(text)` → dict with `status`, `masked_query`, `pii_types_found`, `retrieved_faqs`, `answer`, `reasons`
- `experiments/`: run as modules from repo root, e.g. `uv run python -m experiments.compare_models`

Stage 3 pipeline: query → PII masking → injection detection (→ `BLOCKED_INJECTION`, no retrieval/LLM)
→ E5 retrieval top_k=3 → LLM generation from FAQ context → `OK`.
Statuses: `OK`, `BLOCKED_INJECTION`. `HANDOFF` is not implemented yet.

# Key decisions

- Corpus: DFKI / Deutsche Telekom FAQ dataset, CC BY-SA 4.0 (see `LICENSE-NOTICE.md`).
- 50 records after exact normalized question+answer dedup; `source_instance_ids` kept for provenance.
- One FAQ = one retrieval document (`question + "\n" + answer`); no chunking.
- Embeddings: `intfloat/multilingual-e5-base` with `query: ` / `passage: ` prefixes, L2-normalized.
- Chroma cosine distance: lower = closer; not a calibrated confidence.
- E5 replaced MiniLM after a controlled comparison: better Top-3 retrieval, almost no truncation (1/50 vs 17/50).
- Top-3 context: a tested query had the correct FAQ at rank 2, and the LLM selected the right evidence.
- Customer/contract numbers masked only after explicit labels (e.g. `Kundennummer`), to reduce false positives.
- Injection detection is deterministic pattern matching; bypassable by rephrasing (known limitation).
- LLM access goes through `LLMClient` so the provider can be swapped.
- LLM output is not fully deterministic even at temperature 0, so evaluation should score stable things (retrieved instance_ids, status, groundedness verdict), not exact answer text.

# Known limitations

- Small, narrow FAQ corpus; some source answers are long or bundle several sub-FAQs.
- Pattern-based injection detection and regex PII masking are intentionally limited.
- Retrieval distances not yet calibrated for handoff (E5 distances are compressed: ~0.14–0.24 observed).
- No context-relevance or groundedness checks, no ASR/TTS, no systematic evaluation set yet.

# Stage status

Completed:
1. Data preparation
2. Retrieval baseline and embedding-model experiments
3. PII/injection guards + LLM generation

Next:
4. Relevance gating, HANDOFF, and groundedness
5. ASR/TTS
6. Evaluation
7. README / final documentation
