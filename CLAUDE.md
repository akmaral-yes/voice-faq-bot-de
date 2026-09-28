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
- `asr.py`: `transcribe(path)` with faster-whisper `small`, CPU, int8, language forced to `de` (no language detection);
  model loaded lazily and cached, load time reported separately from transcription time.
- `pipeline.py`: `process_audio(path)` = `transcribe()` → `process_query()`, adds `transcript` and `timings`
  (`asr_model_load`, `asr`, `total`); the raw transcript is diagnostic prototype output, never logged or persisted.
  `process_query(text)` → dict with `status`, `masked_query`, `pii_types_found`, `reasons`, `retrieved_faqs`,
  `top_1_distance`, `answer`, plus internal/debug `relevance_check`, `rejected_answer`, `groundedness_check`.
  `speak_result(result, path)` returns a copy with `tts`: speaks only `answer` (OK or fixed HANDOFF message),
  never debug fields; no TTS for `BLOCKED_INJECTION` (`tts: None`). Adds `timings["tts"]` and adds it to `total`.
- `tts.py`: `synthesize(text, path)` via OpenAI TTS (`gpt-4o-mini-tts-2025-12-15`, voice `marin`, WAV); model/voice
  duplicated from `generate_test_audio.py` (no import of a test-data script), but with a brisk customer-service
  instruction instead of the slower customer-voice one.
- `out/`: generated answer audio; git-ignored because it may contain user-derived content.
- `generate_test_audio.py`: OpenAI TTS → `audio/<stem>.wav` from `audio_samples.txt` (`--force` regenerates).
  `audio/` holds committed synthetic TTS test samples (no real recordings or personal data), so Stage 5 ASR
  is reproducible right after cloning.
- `eval.py` + `eval_cases.json`: frozen evaluation set (20 scored typed cases, 1 diagnostic grounding probe,
  4 typed-vs-audio pairs), each run once; scores status, reasons, retrieved IDs, PII masking, LLM call counts.
  Writes `eval_results.csv` as generated output (no full answers, no raw PII).
- `experiments/`: run as modules from repo root, e.g. `uv run python -m experiments.compare_models`

Pipeline: [audio → Whisper transcript →] query → PII masking → injection detection (→ `BLOCKED_INJECTION`, no retrieval/LLM) → E5 retrieval top_k=3
→ distance gate (→ `HANDOFF`, 0 LLM calls) → context relevance (→ `HANDOFF`) → generation → fallback-sentence check
(→ `HANDOFF`) → groundedness (→ `HANDOFF`) → `OK` [→ TTS of the answer]. Statuses: `OK`, `BLOCKED_INJECTION`, `HANDOFF` (fixed handoff message).

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
- Synthetic clean TTS speech tests the pipeline, not real-world ASR robustness; real speech, noise, accents,
  telephone codecs, and Swiss German are not covered.
- Swiss German is untested; forcing `de` disables language detection.
- Regex PII masking runs on ASR output: spoken digits came out as digit groups (masked in both tests), but with
  wrong digit counts, and other spoken-number renderings could bypass the regex. A production voicebot needs
  PII handling designed for ASR output.
- Observed warm end-to-end processing (ASR + retrieval + 3 sequential LLM calls, excluding cold start and TTS) was ~4.5 s for four clean synthetic clips on the current machine and network; a small prototype measurement, not a production benchmark.
- TTS is a cloud API call (OpenAI). Answers are synthesized as plain text: no number/IBAN pronunciation rules,
  no streaming, no barge-in. Audio input is still tested only with clean synthetic speech.
- With TTS, two single warm runs of 01_umzug took ~9.5 s and ~6.5 s end to end (ASR ~1.2 s + text pipeline
  ~3.4–4.1 s + TTS ~2.0–4.3 s); the spread comes mostly from API/network time. TTS returns the full file before
  playback. Clean synthetic input, warmed local models; small prototype observations, not a production benchmark.
- PII: numbers typed or transcribed as digit words ("null eins fünf …") are not masked at all and reached
  retrieval and all LLM calls in the evaluation.
- Stage 6 evaluation (small, manually authored): in-domain unanswerable top-1 distances (0.14–0.18) overlap
  answerable ones (0.07–0.16), so only the relevance check separates them. `handoff:not_grounded` never fired on
  natural cases, so groundedness recall is only shown by constructed examples.

# Stage status

Completed:
1. Data preparation
2. Retrieval baseline and embedding-model experiments
3. PII/injection guards + LLM generation
4. Relevance gating, HANDOFF, and groundedness
5A. ASR (faster-whisper) feeding the text pipeline
5B. TTS for the final answer
6. Evaluation

Next:
7. README / final documentation
