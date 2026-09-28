"""Stage 6 evaluation: runs the frozen cases in eval_cases.json once each.

Typed cases -> process_query(); audio cases -> the typed source text from
audio_samples.txt via process_query() AND the audio file via process_audio().
Writes eval_results.csv (generated artifact) and prints metrics.

Scores only stable things: status, reasons, retrieved instance_ids, masking.
LLM output is not fully deterministic even at temperature 0, so a re-run can
differ slightly; answer text is never matched exactly.

Usage:  uv run python eval.py
"""

import csv
import json
import re
import statistics
import time
from collections import Counter

from llm import OpenAIClient
from pipeline import INSUFFICIENT_CONTEXT_MESSAGE, OUT_OF_DOMAIN_MAX_DISTANCE, process_audio, process_query

CASES_PATH = "eval_cases.json"
RESULTS_PATH = "eval_results.csv"
NEAR = 0.02  # "near threshold" band for the descriptive distance analysis


class CountingLLM:
    """Wraps the real client to count calls per stage (identified by prompt prefix)."""

    def __init__(self):
        self.inner = OpenAIClient()
        self.calls = Counter()

    def generate(self, prompt):
        if prompt.startswith("Du prüfst, ob FAQ-Einträge"):
            self.calls["relevance"] += 1
        elif prompt.startswith("Du prüfst, ob eine Antwort"):
            self.calls["groundedness"] += 1
        else:
            self.calls["generation"] += 1
        return self.inner.generate(prompt)


def run_case(case, mode, run_fn):
    llm = CountingLLM()
    start = time.perf_counter()
    r = run_fn(llm)
    wall = time.perf_counter() - start

    ids = [f["instance_id"] for f in r["retrieved_faqs"]]
    expected_ids = set(case.get("expected_instance_ids") or [])
    masked = r["masked_query"]
    if case.get("pii_values"):  # typed PII: the raw fake values must be gone
        pii_masked = all(v not in masked and v.replace(" ", "") not in masked.replace(" ", "")
                         for v in case["pii_values"])
    elif case.get("pii_expected"):  # audio PII: raw value unknown (ASR output), so check no digit run survived
        pii_masked = bool(r["pii_types_found"]) and not re.search(r"\d{3,}", masked)
    else:
        pii_masked = None
    grounded = r["groundedness_check"]
    rejected = r["rejected_answer"]
    return {
        "case_id": case["id"],
        "category": case["category"],
        "mode": mode,
        "evaluation_only": bool(case.get("evaluation_only")),
        "expected_status": case["expected_status"],
        "actual_status": r["status"],
        "status_ok": case["expected_status"] == r["status"] if case["expected_status"] else None,
        "reasons": ";".join(r["reasons"]),
        "top1_distance": r["top_1_distance"],
        "retrieved_ids": ";".join(ids),
        "top1_hit": (ids[:1] != [] and ids[0] in expected_ids) if expected_ids else None,
        "top3_hit": bool(expected_ids & set(ids)) if expected_ids else None,
        "pii_types": ";".join(r["pii_types_found"]),
        "pii_masked": pii_masked,
        "relevance_calls": llm.calls["relevance"],
        "generation_calls": llm.calls["generation"],
        "groundedness_calls": llm.calls["groundedness"],
        "fallback_fired": rejected is not None and rejected.strip() == INSUFFICIENT_CONTEXT_MESSAGE,
        "grounded": grounded["grounded"] if grounded else None,
        "unsupported_claims": len(grounded["unsupported_claims"]) if grounded else None,
        "asr_seconds": r.get("timings", {}).get("asr"),
        "wall_seconds": wall,
        "answer_preview": (r["answer"] or "")[:80],
    }


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def main():
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    with open("data/corpus.jsonl", encoding="utf-8") as f:
        corpus_ids = {json.loads(line)["instance_id"] for line in f}
    for case in cases["typed_cases"] + cases["audio_cases"]:
        unknown = set(case.get("expected_instance_ids") or []) - corpus_ids
        assert not unknown, f"{case['id']}: unknown expected IDs {unknown}"
    with open("audio_samples.txt", encoding="utf-8") as f:
        audio_text = dict(
            (part.strip() for part in line.split("|", 1)) for line in f if line.strip()
        )

    rows = []
    for case in cases["typed_cases"]:
        print(f"running {case['id']} ...", flush=True)
        rows.append(run_case(case, "typed", lambda llm, c=case: process_query(c["text"], llm)))
    for case in cases["audio_cases"]:
        case = {**case, "category": "audio_pair"}
        print(f"running {case['id']} (typed source + audio) ...", flush=True)
        text = audio_text[case["stem"]]
        rows.append(run_case(case, "audio_source_typed", lambda llm: process_query(text, llm)))
        rows.append(run_case(case, "audio", lambda llm: process_audio(f"audio/{case['stem']}.wav", llm)))

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print_report(rows)


def print_report(rows):
    typed = [r for r in rows if r["mode"] == "typed"]
    scored = [r for r in typed if not r["evaluation_only"]]
    probe = [r for r in typed if r["evaluation_only"]]
    audio_pairs = [r for r in rows if r["mode"] != "typed"]

    print("\n=== Per-case results (typed)")
    print(f"{'id':<19}{'category':<24}{'expected':<19}{'actual':<19}{'top1':>7}  {'t1':<6}{'t3':<6}reasons / retrieved")
    for r in typed:
        print(f"{r['case_id']:<19}{r['category']:<24}{str(r['expected_status']):<19}{r['actual_status']:<19}"
              f"{fmt(r['top1_distance']):>7}  {str(r['top1_hit']):<6}{str(r['top3_hit']):<6}"
              f"{r['reasons'] or '-'} | {r['retrieved_ids'] or '-'}")

    print("\n=== A. Status accuracy (20 scored typed cases; grounding probe excluded)")
    print(f"overall: {sum(r['status_ok'] for r in scored)}/{len(scored)}")
    for cat in dict.fromkeys(r["category"] for r in scored):
        rs = [r for r in scored if r["category"] == cat]
        print(f"  {cat:<24}{sum(r['status_ok'] for r in rs)}/{len(rs)}")
    print("confusion (expected -> actual):")
    for (e, a), n in sorted(Counter((r["expected_status"], r["actual_status"]) for r in scored).items()):
        print(f"  {e:<18} -> {a:<18} {n}")

    print("\n=== B. Retrieval (scored cases with expected IDs)")
    for label, cats in [("all answerable", None), ("normal_faq", {"normal_faq"}),
                        ("paraphrased_faq", {"paraphrased_faq"}), ("pii", {"pii"})]:
        rs = [r for r in scored if r["top3_hit"] is not None and (cats is None or r["category"] in cats)]
        print(f"  {label:<16} top-1 {sum(r['top1_hit'] for r in rs)}/{len(rs)}   top-3 {sum(r['top3_hit'] for r in rs)}/{len(rs)}")

    print("\n=== C. In-domain unanswerable: which mechanism handled them")
    for r in [r for r in scored if r["category"] == "in_domain_unanswerable"]:
        print(f"  {r['case_id']}: {r['actual_status']} {r['reasons'] or '(no reason: incorrect OK)'}"
              f"{'  [fallback fired]' if r['fallback_fired'] else ''}")

    print("\n=== D. False handoffs (expected OK, got HANDOFF)")
    fh = [r for r in scored if r["expected_status"] == "OK" and r["actual_status"] == "HANDOFF"]
    for r in fh:
        print(f"  {r['case_id']} ({r['category']}): {r['reasons']}, top1 {fmt(r['top1_distance'])}, {r['retrieved_ids']}")
    print("  none" if not fh else "")

    print("=== E. False accepts (expected HANDOFF, got OK)")
    fa = [r for r in scored if r["expected_status"] == "HANDOFF" and r["actual_status"] == "OK"]
    for r in fa:
        print(f"  {r['case_id']} ({r['category']}): reasons {r['reasons'] or '[]'}, top1 {fmt(r['top1_distance'])}, {r['retrieved_ids']}")
    print("  none" if not fa else "")

    print("=== Errors (every scored case with wrong status)")
    for r in [r for r in scored if not r["status_ok"]]:
        print(f"  {r['case_id']} ({r['category']}): expected {r['expected_status']}, got {r['actual_status']}, "
              f"reasons {r['reasons'] or '[]'}, top1 {fmt(r['top1_distance'])}, {r['retrieved_ids'] or '-'}")

    print("\n=== F. Injection")
    inj = [r for r in scored if r["category"] == "injection"]
    print(f"  block rate {sum(r['actual_status'] == 'BLOCKED_INJECTION' for r in inj)}/{len(inj)}")
    for r in inj:
        print(f"  {r['case_id']}: {r['actual_status']} {r['reasons']}")

    print("\n=== G. PII")
    for r in [r for r in scored if r["category"] == "pii"]:
        print(f"  {r['case_id']}: status_ok {r['status_ok']} ({r['actual_status']}), raw PII absent from masked_query: "
              f"{r['pii_masked']}, pii_types {r['pii_types']}")

    print("\n=== H. Handoff reasons")
    for label, rs in [("20 scored typed cases", scored), ("all runs (typed + probe + audio pairs)", rows)]:
        counts = Counter(reason for r in rs for reason in r["reasons"].split(";") if reason.startswith("handoff:"))
        print(f"  {label}: {dict(counts) or 'none'}")

    print("\n=== I. Generation fallback sentence fired")
    fb = [r["case_id"] + "/" + r["mode"] for r in rows if r["fallback_fired"]]
    print(f"  {len(fb)} time(s) {fb}")

    print("\n=== J. Grounding probe (excluded from metrics)")
    for r in probe:
        print(f"  {r['case_id']}: {r['actual_status']} reasons {r['reasons'] or '[]'}; generation called: "
              f"{r['generation_calls'] > 0}; grounded: {r['grounded']}; unsupported claims: {r['unsupported_claims']}; "
              f"not_grounded fired: {'handoff:not_grounded' in r['reasons']}; retrieved {r['retrieved_ids']}")
        print(f"  answer preview: {r['answer_preview']}")

    print(f"\n=== Threshold analysis (descriptive only; OUT_OF_DOMAIN_MAX_DISTANCE = {OUT_OF_DOMAIN_MAX_DISTANCE})")
    groups = {
        "answerable, ended OK": [r for r in scored if r["expected_status"] == "OK" and r["actual_status"] == "OK"],
        "answerable, not OK": [r for r in scored if r["expected_status"] == "OK" and r["actual_status"] != "OK"],
        "in_domain_unanswerable": [r for r in scored if r["category"] == "in_domain_unanswerable"],
        "off_topic": [r for r in scored if r["category"] == "off_topic"],
        "audio (ASR)": [r for r in rows if r["mode"] == "audio"],
    }
    for name, rs in groups.items():
        ds = sorted(r["top1_distance"] for r in rs if r["top1_distance"] is not None)
        if ds:
            print(f"  {name:<24} n={len(ds)} min {ds[0]:.4f} max {ds[-1]:.4f}  [{', '.join(f'{d:.4f}' for d in ds)}]")
    near = [r for r in rows if r["top1_distance"] is not None
            and abs(r["top1_distance"] - OUT_OF_DOMAIN_MAX_DISTANCE) <= NEAR]
    print(f"  within ±{NEAR} of the threshold: {len(near)} run(s)")
    for r in near:
        print(f"    {r['case_id']}/{r['mode']}: {r['top1_distance']:.4f} ({r['actual_status']})")

    print("\n=== Timing (wall-clock seconds, single runs; not a benchmark)")
    by_cat = {}
    for r in rows:
        key = r["category"] if r["mode"] == "typed" else f"audio_pair/{r['mode']}"
        by_cat.setdefault(key, []).append(r["wall_seconds"])
    for key, ts in by_cat.items():
        print(f"  {key:<30} mean {statistics.mean(ts):.2f}  max {max(ts):.2f}  (n={len(ts)})")
    asr = [r["asr_seconds"] for r in rows if r["asr_seconds"] is not None]
    print(f"  ASR only (audio runs): {', '.join(f'{t:.2f}' for t in asr)}")

    print("\n=== Audio drift: typed source text vs. ASR")
    print(f"  {'case':<10}{'mode':<20}{'status':<10}{'top1':>7}  {'t1':<6}{'t3':<6}{'pii':<16}retrieved")
    for r in audio_pairs:
        print(f"  {r['case_id']:<10}{r['mode']:<20}{r['actual_status']:<10}{fmt(r['top1_distance']):>7}  "
              f"{str(r['top1_hit']):<6}{str(r['top3_hit']):<6}{(r['pii_types'] or '-') + ('/masked' if r['pii_masked'] else ''):<16}"
              f"{r['retrieved_ids']}")
    pairs = list(zip(audio_pairs[0::2], audio_pairs[1::2]))
    print(f"  status changed: {sum(t['actual_status'] != a['actual_status'] for t, a in pairs)}/{len(pairs)}")
    print(f"  top-1 FAQ changed: {sum(t['retrieved_ids'].split(';')[0] != a['retrieved_ids'].split(';')[0] for t, a in pairs)}/{len(pairs)}")
    print(f"  expected FAQ dropped out of top-1: {sum(t['top1_hit'] and not a['top1_hit'] for t, a in pairs)}/{len(pairs)}")
    print(f"  expected FAQ dropped out of top-3: {sum(t['top3_hit'] and not a['top3_hit'] for t, a in pairs)}/{len(pairs)}")
    delta = max(pairs, key=lambda p: abs(p[0]["top1_distance"] - p[1]["top1_distance"]))
    print(f"  biggest |top-1 distance change|: {delta[0]['case_id']} "
          f"{delta[0]['top1_distance']:.4f} -> {delta[1]['top1_distance']:.4f}")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
