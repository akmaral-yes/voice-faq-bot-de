"""Text pipeline: PII masking -> injection check -> retrieval -> LLM generation.

No relevance threshold, handoff or groundedness check yet (later stages).
"""

from guards import detect_injection, mask_pii
from llm import LLMClient, OpenAIClient
from retrieval import retrieve

PROMPT_TEMPLATE = """Du bist ein Kundenservice-Assistent. Beantworte die Frage des Kunden
ausschließlich auf Grundlage der unten stehenden FAQ-Einträge.

Regeln:
- Verwende nur Informationen aus den FAQ-Einträgen.
- Erfinde keine Informationen und nutze kein Wissen außerhalb der FAQ-Einträge.
- Wenn die FAQ-Einträge die Frage nicht beantworten, antworte genau:
  "Dazu liegen mir leider nicht genügend Informationen vor."
- Antworte kurz, freundlich und auf Deutsch.

FAQ-Einträge:

{context}

Frage des Kunden: {query}

Antwort:"""


def format_context(faqs):
    blocks = []
    for i, faq in enumerate(faqs, start=1):
        blocks.append(
            f"FAQ {i}\nID: {faq['instance_id']}\nFrage: {faq['question']}\nAntwort: {faq['answer']}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(masked_query, llm: LLMClient):
    faqs = retrieve(masked_query, top_k=3)
    prompt = PROMPT_TEMPLATE.format(context=format_context(faqs), query=masked_query)
    return llm.generate(prompt), faqs


def process_query(text, llm: LLMClient = None):
    masked_query, pii_types = mask_pii(text)  # nothing below sees the raw text
    result = {
        "status": "OK",
        "masked_query": masked_query,
        "pii_types_found": pii_types,
        "retrieved_faqs": [],
        "answer": None,
        "reasons": detect_injection(masked_query),
    }

    if result["reasons"]:
        result["status"] = "BLOCKED_INJECTION"
        return result

    answer, faqs = generate_answer(masked_query, llm or OpenAIClient())
    result["answer"] = answer
    result["retrieved_faqs"] = [
        {"instance_id": f["instance_id"], "question": f["question"], "distance": f["distance"]}
        for f in faqs
    ]
    return result
