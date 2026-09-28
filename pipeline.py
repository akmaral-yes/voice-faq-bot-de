"""Text pipeline with guards, gating and HANDOFF.

query -> PII masking -> injection check (BLOCKED_INJECTION)
      -> retrieval top-3 -> out-of-domain distance gate (HANDOFF)
      -> context-relevance check (HANDOFF) -> generation -> fallback check (HANDOFF)
      -> groundedness check (HANDOFF) -> OK

An accepted request makes 3 sequential LLM calls (relevance, generation,
groundedness). That is deliberate for clarity and measurability; a production
voicebot would likely need to cut this latency (e.g. combine or parallelize calls).
"""

import json

from guards import detect_injection, mask_pii
from llm import LLMClient, OpenAIClient
from retrieval import retrieve

# Provisional coarse out-of-domain gate.
# - Model: intfloat/multilingual-e5-base; Chroma cosine distance; LOWER = closer.
# - Manual observations so far: relevant queries had top-1 distances ~0.11-0.17,
#   the one clearly off-topic query (apple cake) ~0.23. 0.20 sits between them.
# - Based on very little data; must be calibrated in Stage 6 on a larger eval set.
# - Only meant to catch clearly out-of-domain questions. Raw vector distance is
#   NOT a calibrated confidence probability, and this gate cannot detect
#   "right topic, wrong FAQ" - that is what the context-relevance check is for.
OUT_OF_DOMAIN_MAX_DISTANCE = 0.20

INSUFFICIENT_CONTEXT_MESSAGE = "Dazu liegen mir leider nicht genügend Informationen vor."
HANDOFF_MESSAGE = "Dazu verbinde ich Sie mit einem Kundenberater."

GENERATION_PROMPT = """Du bist ein Kundenservice-Assistent. Beantworte die Frage des Kunden
ausschließlich auf Grundlage der unten stehenden FAQ-Einträge.

Regeln:
- Verwende nur Informationen aus den FAQ-Einträgen.
- Erfinde keine Informationen und nutze kein Wissen außerhalb der FAQ-Einträge.
- Wenn die FAQ-Einträge die Frage nicht beantworten, antworte genau:
  "{fallback}"
- Antworte kurz, freundlich und auf Deutsch.

FAQ-Einträge:

{context}

Frage des Kunden: {query}

Antwort:"""

# Two separate checks, deliberately not combined:
# - context relevance (before generation): do the retrieved FAQs contain enough
#   information to answer the user's question?
# - groundedness (after generation): is the generated answer supported by those FAQs?

RELEVANCE_PROMPT = """Du prüfst, ob FAQ-Einträge eine Kundenfrage beantworten können.

Aufgabe: Enthalten die FAQ-Einträge genügend Informationen, um die eigentliche Frage
des Kunden zu beantworten?
- Beurteile nur die FAQ-Einträge; es gibt noch keine Antwort zu prüfen.
- Ein Eintrag zum selben Thema reicht nicht, wenn er die konkrete Frage nicht beantwortet.
- Begründe deine Entscheidung ausschließlich anhand der bereitgestellten
  FAQ-Einträge und der Kundenfrage. Verwende kein externes Wissen.

Antworte ausschließlich mit JSON in genau dieser Form:
{{"relevant": true oder false, "explanation": "kurze Begründung"}}

FAQ-Einträge:

{context}

Frage des Kunden: {query}"""

GROUNDEDNESS_PROMPT = """Du prüfst, ob eine Antwort durch FAQ-Einträge belegt ist.

Aufgabe: Ist jede sachliche Aussage in der Antwort durch die FAQ-Einträge gestützt?
- Prüfe nur, ob die Aussagen belegt sind, nicht, ob die Antwort zu einer Kundenfrage passt.
- Begrüßungen, Höflichkeitsfloskeln, Überleitungen und Sätze wie
  "Bei weiteren Fragen helfen wir Ihnen gern." sind keine sachlichen Aussagen.
- Sinngemäße Umformulierungen von FAQ-Inhalten gelten als belegt.
- Liste jede nicht belegte sachliche Aussage in "unsupported_claims" auf.

Antworte ausschließlich mit JSON in genau dieser Form:
{{"grounded": true oder false, "unsupported_claims": ["..."]}}

FAQ-Einträge:

{context}

Antwort:
{answer}"""


def format_context(faqs):
    blocks = []
    for i, faq in enumerate(faqs, start=1):
        blocks.append(
            f"FAQ {i}\nID: {faq['instance_id']}\nFrage: {faq['question']}\nAntwort: {faq['answer']}"
        )
    return "\n\n---\n\n".join(blocks)


def parse_json(text):
    """Parse an LLM JSON reply, tolerating ```json fences. Returns dict or None."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def check_context_relevance(masked_query, faqs, llm: LLMClient):
    """Returns {"relevant": bool, "explanation": str} or None if unparseable."""
    reply = llm.generate(RELEVANCE_PROMPT.format(context=format_context(faqs), query=masked_query))
    data = parse_json(reply)
    if data is None or not isinstance(data.get("relevant"), bool):
        return None
    return {"relevant": data["relevant"], "explanation": str(data.get("explanation", ""))}


def generate_answer(masked_query, faqs, llm: LLMClient):
    prompt = GENERATION_PROMPT.format(
        fallback=INSUFFICIENT_CONTEXT_MESSAGE, context=format_context(faqs), query=masked_query
    )
    return llm.generate(prompt)


def check_groundedness(answer, faqs, llm: LLMClient):
    """Returns {"grounded": bool, "unsupported_claims": [str]} or None if unparseable.

    Conservative: listed unsupported claims count as not grounded even if the
    model also said grounded=true.
    """
    reply = llm.generate(GROUNDEDNESS_PROMPT.format(context=format_context(faqs), answer=answer))
    data = parse_json(reply)
    if data is None or not isinstance(data.get("grounded"), bool):
        return None
    claims = data.get("unsupported_claims", [])
    if not isinstance(claims, list):
        return None
    claims = [str(c) for c in claims]
    return {"grounded": data["grounded"] and not claims, "unsupported_claims": claims}


def process_query(text, llm: LLMClient = None):
    masked_query, pii_types = mask_pii(text)  # nothing below sees the raw text
    result = {
        "status": "OK",
        "masked_query": masked_query,
        "pii_types_found": pii_types,
        "reasons": detect_injection(masked_query),
        "retrieved_faqs": [],
        "top_1_distance": None,
        "answer": None,
        # internal/debug fields, not customer-facing:
        "relevance_check": None,
        "rejected_answer": None,
        "groundedness_check": None,
    }

    if result["reasons"]:
        result["status"] = "BLOCKED_INJECTION"
        return result

    def handoff(reason):
        result["status"] = "HANDOFF"
        result["reasons"].append(reason)
        result["answer"] = HANDOFF_MESSAGE
        return result

    faqs = retrieve(masked_query, top_k=3)
    result["retrieved_faqs"] = [
        {"instance_id": f["instance_id"], "question": f["question"], "distance": f["distance"]}
        for f in faqs
    ]
    result["top_1_distance"] = faqs[0]["distance"]

    if result["top_1_distance"] > OUT_OF_DOMAIN_MAX_DISTANCE:
        return handoff("handoff:out_of_domain")

    llm = llm or OpenAIClient()

    relevance = check_context_relevance(masked_query, faqs, llm)
    result["relevance_check"] = relevance
    if relevance is None:
        return handoff("handoff:relevance_unparseable")
    if not relevance["relevant"]:
        return handoff("handoff:insufficient_context")

    answer = generate_answer(masked_query, faqs, llm)
    if answer.strip() == INSUFFICIENT_CONTEXT_MESSAGE:  # secondary safety net
        result["rejected_answer"] = answer
        return handoff("handoff:insufficient_context")

    groundedness = check_groundedness(answer, faqs, llm)
    result["groundedness_check"] = groundedness
    if groundedness is None:
        result["rejected_answer"] = answer
        return handoff("handoff:groundedness_unparseable")
    if not groundedness["grounded"]:
        result["rejected_answer"] = answer
        return handoff("handoff:not_grounded")

    result["answer"] = answer
    return result
