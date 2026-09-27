"""Build data/corpus.jsonl from the DFKI FAQ rewrite dataset.

The source has 30 JSON files (one per LLM/prompt setup), each containing the
same 56 FAQ instances. We keep only the human-written reference Q/A.

Actual source schema (differs slightly from the dataset README):
  instance["reference"]["content"]["question" | "answer"]
  instance["usecase"]   (not "use_case")
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw/data/faq-data")
OUT_PATH = Path("data/corpus.jsonl")


def extract_record(instance):
    """Return a clean record, or None if the instance is malformed."""
    try:
        content = instance["reference"]["content"]
        record = {
            "instance_id": instance["instance_id"].strip(),
            "question": content["question"].strip(),
            "answer": content["answer"].strip(),
            "use_case": instance["usecase"].strip(),
        }
    except (KeyError, TypeError, AttributeError):
        return None
    if not all(record.values()):
        return None
    return record


def normalize(text):
    """Collapse whitespace; used only for duplicate detection."""
    return re.sub(r"\s+", " ", text).strip()


def dedup_exact_pairs(records):
    """Merge records whose normalized (question, answer) pair is identical.

    The first record is kept; every record gets `source_instance_ids`.
    """
    merged = {}  # (question, answer) -> record
    for record in records:
        key = (normalize(record["question"]), normalize(record["answer"]))
        if key in merged:
            merged[key]["source_instance_ids"].append(record["instance_id"])
        else:
            merged[key] = {**record, "source_instance_ids": [record["instance_id"]]}
    return list(merged.values())


def same_question_different_answer(records):
    """Group records that share a normalized question but differ in answer."""
    by_question = {}
    for record in records:
        by_question.setdefault(normalize(record["question"]), []).append(record)
    return {q: group for q, group in by_question.items() if len(group) > 1}


def main():
    raw_count = 0
    skipped = 0
    records = {}  # instance_id -> record; first occurrence wins

    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            instances = json.loads(path.read_text(encoding="utf-8"))["instances"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Skipping unreadable file {path.name}: {e}")
            continue

        for instance in instances:
            raw_count += 1
            record = extract_record(instance)
            if record is None:
                skipped += 1
                continue
            records.setdefault(record["instance_id"], record)

    unique_by_id = list(records.values())
    corpus = dedup_exact_pairs(unique_by_id)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for record in corpus:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Raw records found:     {raw_count}")
    print(f"Malformed / skipped:   {skipped}")
    print(f"Unique instance_ids:   {len(unique_by_id)}")
    print(f"After exact Q+A dedup: {len(corpus)}")

    print("\nMerged (identical question + answer):")
    for record in corpus:
        if len(record["source_instance_ids"]) > 1:
            print(f"  {record['source_instance_ids']}  {record['question']!r}")
    print("\nSame question, different answers (kept separately):")
    conflicts = same_question_different_answer(corpus)
    for question, group in conflicts.items():
        print(f"  {question!r}: {[r['instance_id'] for r in group]}")
    if not conflicts:
        print("  (none)")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
