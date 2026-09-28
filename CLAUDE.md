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
- `pipeline.py`: `process_query(text)` → dict with `status`, `masked_query`, `pii_types_found`, `reasons`, `retrieved_faqs`,
  `top_1_distance`, `answer`, plus internal/debug `relevance_check`, `rejected_answer`, `groundedness_check`
- `experiments/`: run as modules from repo root, e.g. `uv run python -m experiments.compare_models`

Pipeline: query → PII masking → injection detection (→ `BLOCKED_INJECTION`, no retrieval/LLM) → E5 retrieval top_k=3
→ distance gate (→ `HANDOFF`, 0 LLM calls) → context relevance (→ `HANDOFF`) → generation → fallback-sentence check
(→ `HANDOFF`) → groundedness (→ `HANDOFF`) → `OK`. Statuses: `OK`, `BLOCKED_INJECTION`, `HANDOFF` (fixed handoff message).

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
- Context relevance (before generation: can the FAQs answer the question?) and groundedness (after: is the answer
  supported by the FAQs?) are separate LLM-judge checks with strict JSON; unparseable output → `HANDOFF`.
- Accepted requests make 3 sequential LLM calls (relevance, generation, groundedness) for clarity; latency cost accepted for now.
- LLM access goes through `LLMClient` so the provider can be swapped.
- LLM output is not fully deterministic even at temperature 0, so evaluation should score stable things (retrieved instance_ids, status, groundedness verdict), not exact answer text.

# Known limitations

- Small, narrow FAQ corpus; some source answers are long or bundle several sub-FAQs.
- Pattern-based injection detection and regex PII masking are intentionally limited.
- `OUT_OF_DOMAIN_MAX_DISTANCE = 0.20` is provisional (from ~5 manual queries; E5 distances are compressed ~0.11–0.24);
  must be calibrated in Stage 6. It only catches clearly off-topic queries, not "right topic, wrong FAQ".
- Relevance/groundedness judges are LLMs themselves and may err; 3 sequential LLM calls add latency.
- No ASR/TTS, no systematic evaluation set yet.

# Stage status

Completed:
1. Data preparation
2. Retrieval baseline and embedding-model experiments
3. PII/injection guards + LLM generation
4. Relevance gating, HANDOFF, and groundedness

Next:
5. ASR/TTS
6. Evaluation
7. README / final documentation
