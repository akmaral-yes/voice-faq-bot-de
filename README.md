# German Voice FAQ Bot: Retrieval, Guardrails, Handoff and Evaluation

A small German customer-service voice FAQ prototype built on a public telecom FAQ dataset.
German text or speech comes in; speech is transcribed locally with Whisper. Sensitive-looking
user input is masked before retrieval or any LLM call, and obvious prompt-injection attempts are
blocked. The system retrieves FAQ entries, checks whether they actually contain enough information,
and lets an LLM answer only from that context. Insufficient or unsupported cases are handed off to a
human agent. The final answer can be synthesized back to speech. Everything was evaluated on a small,
frozen test set.

This is a **small, time-boxed prototype** intended to explore retrieval, guardrails, voice integration,
and evaluation trade-offs. It is not production software.

The architecture is reusable across customer-service domains, but adapting it requires replacing the
knowledge corpus and evaluation set and reviewing language- and domain-specific guardrail policies
(German prompts, German/English injection patterns, regex PII rules, telecom test cases).

---

## 1. Architecture

```text
Audio file                         Typed German text
  ↓                                        │
ASR (faster-whisper, local)                │
  ↓                                        │
Transcript ────────────────────────────────┤
                                           ↓
                                 PII masking (regex)
                                           ↓
                        Prompt-injection detection ──→ BLOCKED_INJECTION (no retrieval, no LLM)
                                           ↓
                    E5 embedding + Chroma retrieval (top 3)
                                           ↓
                     Coarse distance gate ──→ HANDOFF (0 LLM calls)
                                           ↓
                   LLM context-relevance check ──→ HANDOFF
                                           ↓
                        LLM answer generation
                                           ↓
               Fixed fallback-sentence check ──→ HANDOFF
                                           ↓
                     LLM groundedness check ──→ HANDOFF
                                           ↓
                                          OK
                                           ↓
                   TTS of the answer (OK or fixed handoff message)
```

An accepted request makes **three sequential LLM calls**: context relevance → generation →
groundedness. They are sequential by dependency: relevance decides whether generation should happen
at all, and groundedness needs the generated answer.

| File | Responsibility |
|---|---|
| `prepare_data.py` | Raw DFKI JSON → deduplicated `data/corpus.jsonl` |
| `build_index.py` | Embed the corpus and (re)build the persistent Chroma index in `data/chroma/` |
| `retrieval.py` | `retrieve(query, top_k=3)` against the existing index; returns raw cosine distances |
| `guards.py` | `mask_pii()` (regex) and `detect_injection()` (phrase patterns) |
| `llm.py` | `LLMClient` protocol and the OpenAI implementation |
| `pipeline.py` | `process_query()`, `process_audio()`, `speak_result()`; gating, judges, handoff |
| `asr.py` | `transcribe()` with faster-whisper |
| `tts.py` | `synthesize()` with OpenAI TTS |
| `eval.py` | Runs the frozen evaluation once and prints metrics |
| `eval_cases.json` | Frozen evaluation cases and expected outcomes |

Result statuses: `OK`, `HANDOFF` (fixed message: *"Dazu verbinde ich Sie mit einem Kundenberater."*),
`BLOCKED_INJECTION`. Every result carries a `reasons` list such as `handoff:insufficient_context`.

---

## 2. Key design decisions

**Embedding model.** The first baseline, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
has a 128-token limit and truncated 17 of 50 FAQ documents. A question-only ablation (no truncation)
did not fix the observed misses, so truncation alone was not the explanation. Switching to
`intfloat/multilingual-e5-base` (`query: ` / `passage: ` prefixes, normalized embeddings, cosine
distance, 512 tokens) reduced truncation to 1/50 and substantially improved the difficult manual
examples. E5 improved the difficult retrieval examples and reduced truncation, but did not eliminate
retrieval errors (see §3). Raw embedding distance is treated only as a heuristic similarity signal,
not calibrated confidence. The comparison scripts are in `experiments/`.

**One FAQ = one document.** The corpus is small (50 records) and each FAQ is already semantically
self-contained, so each is indexed as `question + "\n" + answer` with no chunking. This is a
deliberate simplicity choice for this corpus.

**Top-3 context.** Manual tests repeatedly showed a distractor at rank 1 with the correct FAQ at rank 2
(e.g. "switching provider" ranking above "moving" for keep-my-number questions); the LLM still answered
correctly from top-3 context. Top-3 does not solve retrieval failures: one Stage 6 case had the
correct FAQ outside the top 3.

**Relevance vs. groundedness.** Two separate LLM judges with strict JSON output:
- *Relevance* (before generation): does the retrieved context contain enough information to answer
  the user's actual question?
- *Groundedness* (after generation): is every factual claim in the generated answer supported by that
  context?

They catch different failures: an answer can be fully supported by an irrelevant FAQ while answering
the wrong question. Unparseable judge output leads to `HANDOFF`. In the frozen Stage 6 evaluation,
relevance handled every in-domain-unanswerable case before generation, so groundedness did not fire
naturally. Its demonstrated failure-catching behavior currently comes from a deliberately constructed
earlier test (an invented price and deadline were flagged; a supported answer passed).

**Deterministic guards.** PII masking uses regex rules (IBAN, German/Swiss phone numbers, and
customer/contract numbers only after explicit labels such as `Kundennummer`); injection detection uses
German/English phrase combinations. They are predictable, cheap, explainable and need no extra LLM
call. Pattern-based guards are incomplete and can be bypassed by unseen phrasing or formats.

**LLM abstraction.** All LLM calls go through the small `LLMClient` interface (`generate(prompt) -> str`).
The current provider is OpenAI with `gpt-4.1-mini` at temperature 0 (overridable via `OPENAI_MODEL`).
The provider-specific implementation is isolated behind `LLMClient`. A different provider adapter,
such as AWS Bedrock, could be implemented, but Bedrock has not been implemented or tested in this
project.

---

## 3. Evaluation results

**Setup.** The frozen evaluation (project Stage 6) in `eval_cases.json` contains 20 scored typed cases, 1 separate diagnostic grounding probe,
and 4 typed-vs-audio pairs. Expected outcomes were written and frozen before the run; expected FAQ IDs
were checked against `data/corpus.jsonl`. Each case ran once. Labels were authored manually by the
project author, not independently annotated. Audio is synthetic and clean. Only stable signals are
scored (status, reasons, retrieved IDs, masking), never exact answer text, because LLM output is not
fully deterministic even at temperature 0.

### Status accuracy: 19/20

| Category | Result |
|---|---:|
| Normal FAQ (close to FAQ wording) | 5/5 |
| Paraphrased FAQ (caller phrasing) | 4/5 |
| Off-topic | 2/2 |
| In-domain unanswerable | 4/4 |
| Prompt injection (1 German, 1 English) | 2/2 |
| PII (fake phone number, fake IBAN) | 2/2 |

**False accepts: 0. False handoffs: 1.** The single miss was a false handoff, not a wrong answer.

### Retrieval

| Group | Top-1 | Top-3 |
|---|---:|---:|
| Normal FAQ | 5/5 | 5/5 |
| Paraphrased FAQ | 3/5 | 4/5 |
| PII cases | 2/2 | 2/2 |
| All answerable | 10/12 | 11/12 |

Headline: **retrieval degraded substantially under natural paraphrasing** in this small set.

### Guards

- **Injection:** both frozen injection cases were blocked before retrieval. This is not evidence of a
  robust injection defense.
- **PII:** in the two scored cases (digit-form phone number, IBAN), the raw fake value was absent from
  `masked_query`. This does not mean PII handling is complete; see §5.

### Handoff reasons (20 scored cases)

| Reason | Count |
|---|---:|
| `handoff:insufficient_context` (relevance judge) | 5 |
| `handoff:out_of_domain` (distance gate) | 2 |
| `handoff:not_grounded` | 0 |
| Judge JSON parse failures | 0 |

The fixed generation fallback sentence fired zero times in the Stage 6 run because
insufficient-context cases were intercepted before generation by the relevance judge.

### The one failure: `para_05`

> "Ich wechsle die Wohnung. Muss ich meinen Internet- und Telefonvertrag vorher beenden?"

The corpus directly answers this in `faq_erste_schritte-4` (no cancellation is needed when moving); the
frozen evaluation also accepted `faq_erste_schritte-7` as supporting context. Retrieval
returned provider-switch FAQs (`-31`, `-39`, `-17`) at a plausible top-1 distance of 0.1401 and left
both acceptable FAQs out of the top 3. This was a genuine retrieval failure rather than a bad label. The
relevance judge recognized that the retrieved context did not answer the question and handed off
instead of producing an answer.

### Diagnostic grounding probe (not scored)

*"Wie viel kostet der Umzug meines Festnetz-Anschlusses genau in Euro?"* The relevant FAQ refers to a
price list without giving an amount, which could tempt the generator to invent a figure. The relevance
judge handed it off before generation, so the probe did not exercise the groundedness check.

---

## 4. Threshold findings

`OUT_OF_DOMAIN_MAX_DISTANCE = 0.20` was set provisionally from a handful of manual queries.
Top-1 cosine distances in Stage 6:

| Group | Top-1 distance range |
|---|---|
| Answerable, ended OK (n=11) | 0.0655 – 0.1569 |
| Answerable, false handoff (`para_05`) | 0.1401 |
| In-domain unanswerable (n=4) | 0.1426 – 0.1783 |
| Off-topic (n=2) | 0.2204 – 0.3059 |
| Audio / ASR runs (n=4) | 0.1017 – 0.1801 |

Embedding distance separated the two tested clearly off-topic questions from the FAQ-like questions in
this small sample (the closer one only by 0.02), but it did not separate answerable from
in-domain-unanswerable questions: those ranges overlap heavily. The 0.20 gate is therefore a coarse
out-of-domain heuristic, not an answerability or confidence threshold. All four in-domain-unanswerable
cases were handled by the relevance judge, none by the distance gate.

This is not a calibration study: the sample is tiny and manually authored. A separate
development/calibration set would be needed before threshold tuning.

---

## 5. Notable findings

**Paraphrasing is the main observed retrieval weakness.** Normal and paraphrased cases target the same
five FAQ intents, making the comparison more directly indicative of sensitivity to wording: top-1 fell
from 5/5 to 3/5 and top-3 from 5/5 to 4/5. Five pairs do not support significance claims.

**Safe handoff prevented a wrong answer.** In `para_05` the correct FAQ was absent from the top 3 while
the distance (0.1401) looked entirely plausible, and the in-domain-unanswerable distances overlapped the
answerable ones (§4). The relevance judge handled those cases. In this evaluation, similarity distance
alone did not distinguish answerable from in-domain-unanswerable queries.

**Relevance is load-bearing.** In this small evaluation, the relevance judge handled all
in-domain-unanswerable cases that the distance gate could not distinguish. It is itself a single LLM
judgment with no independent fallback if it makes the wrong relevance decision.

**Groundedness is weakly exercised.** `handoff:not_grounded` fired zero times in the natural Stage 6
evaluation; its detection has only been shown on a deliberately fabricated answer. Its incremental
real-world value is not established by the current evaluation.

**The generation fallback may be redundant.** It fired 0 times because relevance stopped
insufficient-context queries first. The evaluation suggests potential redundancy between the relevance
gate and the generation fallback, but the sample is too small to justify removing it.

**PII + speech is a real gap.**
- The typed source text of audio sample 03 writes a fake phone number as German digit words
  ("null eins fünf eins …"). It was **not masked** and reached retrieval and all three LLM calls.
- In the audio version, Whisper happened to transcribe the spoken digits as digit groups, which the
  regex then masked. Masking worked partly because of the ASR output format, not because the guard
  understands spoken numbers.
- Whisper also changed the number itself (13 and 10 digits instead of 12 in the two number samples).
  In sample 04, the typed and ASR versions collapsed to the same `[TELEFONNUMMER]` placeholder, so the
  ASR error was invisible downstream.

Voice PII protection must be designed against realistic ASR transcripts rather than only typed numeric
formats.

**Retrieved corpus content is outside query-side masking.** `mask_pii()` protects incoming user text
before retrieval and LLM calls, but it does not sanitize values already present in retrieved corpus
documents. For example, `faq_erste_schritte-20` contains a public service SMS number for the provider's
switching advisers. That is service contact data, not customer PII, but the point stands: corpus
content reaches the LLM unmasked, so knowledge-base ingestion needs its own policy.

**ASR changes retrieval behavior.** "SIM-Karte" was transcribed as "Sinnkarte". The correct FAQ stayed
at rank 1, but the top-1 distance moved from 0.1596 (typed source) to 0.1801, about 0.02 below the
distance gate. A slightly worse transcription could therefore cause a valid question to be rejected
before relevance checking. Across the four typed-vs-audio pairs: status changed 0/4, top-1 FAQ changed
1/4, expected FAQ dropped out of top-1 1/4 and out of top-3 0/4. Four clean synthetic clips are not a
basis for generalization.

---

## 6. Latency and voice UX

These are single prototype observations on one machine, dependent on network and API conditions, and
not benchmarks. Local models (Whisper, E5) were already loaded; cold-start loading is not included.

**Processing latency before playback.**
- Text response after audio input: warm ASR + retrieval + sequential LLM processing was roughly 4–6 s
  in the synthetic-audio tests (single runs per clip, across two test sessions).
- Full non-streaming voice response including TTS: two warm runs of the same clip took ~9.5 s and
  ~6.5 s before the complete answer audio was available (ASR ~1.2 s, text pipeline ~3.4–4.1 s, TTS
  ~2.0–4.3 s). The spread came mostly from API/network time.

**Playback duration.** The generated normal answers last about 10–13 s when played. This is
separate from pre-playback latency: processing delay determines how long the caller waits before speech
starts, while audio duration determines how long the caller then listens to the response. They are separate voice-UX costs.

Changing the TTS speaking-style instruction to a brisker customer-service tempo did not materially
shorten the audio, which suggests answer length and content matter
more than the tested style instruction. TTS currently waits for the complete file; there is no
streaming playback and no barge-in.

---

## 7. Limitations

- **Evaluation:** 20 scored typed cases and 4 audio pairs; manually authored labels, no independent
  annotator; one run per case; LLM output not fully deterministic; no real user traffic.
- **Audio:** synthetic, clean TTS speech only; no phone-channel noise, accents or real speakers; Swiss
  German untested (ASR language is forced to `de`); no microphone or telephony integration.
- **Voice UX:** no streaming ASR/TTS, no barge-in or interruption handling.
- **Latency:** three sequential LLM calls per accepted request; cloud LLM and TTS dependencies; timings
  depend on machine, network and API.
- **Guards:** regex PII masking has known coverage gaps (e.g. digit words); pattern-based injection
  detection is bypassable.
- **Retrieval:** raw vector distance is not calibrated confidence; the 0.20 gate was not calibrated on
  a development set; no reranker or hybrid retrieval; one-FAQ-per-document suits only this small
  50-record corpus.
- **Checks:** the groundedness checker was not naturally exercised in Stage 6.

---

## 8. Production path

What I would investigate for a real deployment. None of this is implemented.

- **Retrieval and evaluation:** separate development and evaluation sets built from real anonymized
  query distributions, with many more paraphrased and noisy-speech cases. Then evaluate reranking and/or
  hybrid lexical + semantic retrieval specifically against the observed paraphrase failures, rather
  than adding complexity by default.
- **Threshold:** calibrate the distance gate on a development set, keep it in an appropriate
  out-of-domain role, and never read embedding distance as a probability.
- **Check architecture:** measure the incremental value and latency of the relevance,
  generation-fallback and groundedness stages, then consider simplifying or combining redundant
  checks, using smaller judge models, or redesigning the flow to reduce sequential LLM latency.
  Currently relevance appears essential, the fallback never fired and groundedness never fired
  naturally, so any simplification should be measurement-driven. Define explicit fail-closed behavior
  when a judge is unavailable or returns malformed output (today malformed output already leads to
  `HANDOFF`; outages are not handled).
- **PII:** design detection for ASR transcript formats (digit words, grouped numbers, self-corrections,
  hesitations, mixed spoken/digit forms). Define corpus-ingestion sanitization separately from query
  masking, distinguishing public service numbers from personal identifiers.
- **Voice:** streaming ASR and streaming TTS (time-to-first-audio), shorter voice-oriented answers,
  telephony and noisy-audio testing, barge-in, and Swiss German evaluation if the use case needs it.
- **Guardrails:** replace or extend pattern checks according to an actual risk model, maintain
  domain-specific policies, and test bypass cases instead of claiming complete protection.
- **Provider:** another provider behind `LLMClient` (e.g. AWS Bedrock) would need a concrete adapter,
  configuration and authentication, model-compatibility and JSON-output testing, and latency/cost
  evaluation.

---

## 9. Setup and run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                  # install dependencies from uv.lock
export OPENAI_API_KEY=...                # or put it in a .env file (git-ignored)
# optional: export OPENAI_MODEL=...      # default: gpt-4.1-mini
```

**Data.** `data/corpus.jsonl` is committed. To regenerate it, clone the source dataset into the
git-ignored `data/raw/` first:

```bash
git clone --depth 1 https://github.com/DFKI-NLP/faq-rewrites-llms.git data/raw
uv run python prepare_data.py            # 1,680 raw records → 56 unique IDs → 50 FAQs
```

**Index.** The Chroma index is not committed. Build it once (it deletes and rebuilds `data/chroma/`):

```bash
uv run python build_index.py
uv run python retrieval.py "Kann ich beim Umzug meine Rufnummer behalten?"   # retrieval only
```

**Typed query** (the pipeline is a Python API; there is no chat CLI):

```bash
uv run python -c '
from pipeline import process_query
r = process_query("Wer bezahlt die Rechnung für die PlusKarten?")
print(r["status"], r["answer"])
'
```

**Audio query with spoken answer** (writes to the git-ignored `out/`):

```bash
uv run python -c '
from pipeline import process_audio, speak_result
r = speak_result(process_audio("audio/01_umzug.wav"), "out/answer.wav")
print(r["status"], r["transcript"], r["tts"]["output_path"])
'
```

**Test audio.** `audio/` contains four committed synthetic samples generated from `audio_samples.txt`.
`uv run python generate_test_audio.py` skips existing files; `--force` regenerates them.

**Evaluation** (calls the OpenAI API for every non-blocked case):

```bash
uv run python eval.py
```

`eval_cases.json` is the frozen test set; `eval_results.csv` is generated output (no full answers, no
raw PII).

**Experiments.** Retrieval-representation and embedding-model comparisons:
`uv run python -m experiments.compare_retrieval`, `uv run python -m experiments.compare_models`.

---

## 10. Dataset and license

- Source: the DFKI / Deutsche Telekom German FAQ dataset from
  [DFKI-NLP/faq-rewrites-llms](https://github.com/DFKI-NLP/faq-rewrites-llms), licensed
  **CC BY-SA 4.0**. `data/corpus.jsonl` is an adapted version under the same license.
- Preprocessing: the source repeats the same 56 FAQ instances across 30 LLM-evaluation files
  (1,680 records). Only the human-written reference question/answer pairs are kept, deduplicated by
  `instance_id` (56), then by exact whitespace-normalized question+answer (50). `source_instance_ids`
  keeps provenance for merged records. One FAQ is indexed as one document.
- Attribution, transformation notes and the paper citation (Gabryszak et al., INLG 2024) are in
  [`LICENSE-NOTICE.md`](LICENSE-NOTICE.md).
- The audio under `audio/` is synthetic TTS speech with fake data only; it contains no real customer
  recordings.
- Code is licensed under MIT (`LICENSE`).

---

## 11. Repository layout

```text
prepare_data.py          corpus preparation
build_index.py           embed corpus → Chroma
retrieval.py             retrieve(query, top_k)
guards.py                PII masking, injection detection
llm.py                   LLMClient + OpenAI client
pipeline.py              process_query / process_audio / speak_result
asr.py                   faster-whisper transcription
tts.py                   OpenAI TTS
eval.py                  evaluation runner
eval_cases.json          frozen evaluation set
eval_results.csv         generated evaluation output
generate_test_audio.py   synthetic test-audio generator
audio_samples.txt        source text for the test audio
audio/                   committed synthetic test audio (4 WAV files)
data/corpus.jsonl        cleaned 50-record FAQ corpus
data/chroma/             local vector index (git-ignored, built by build_index.py)
experiments/             exploratory model/representation comparisons (not runtime code)
LICENSE-NOTICE.md        dataset attribution and citation
```
