"""Stage 2B experiment (MiniLM): question+answer embeddings vs. question-only embeddings.

Both collections are temporary and in-memory, so this experiment is independent
of the persistent project index. The stored document is question + answer in
both; only the embedded text differs. Cosine distance (lower = closer).

Run from the repository root:  uv run python -m experiments.compare_retrieval
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from build_index import CORPUS_PATH

MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TESTS = [
    ("Q1", "Ich ziehe nächsten Monat in eine andere Stadt – kann ich meine alte Telefonnummer mitnehmen?", "faq_erste_schritte-30"),
    ("Q2", "Ich hab aus Versehen die SMS mit dem Link für Netflix weggeklickt, was mach ich jetzt?", "faq_netflix-3"),
    ("Q3", "Meine Tochter soll eine zweite SIM-Karte über meinen Handyvertrag bekommen – wer zahlt dann dafür?", "faq_pluskarten-19"),
    ("Q4", "Wie backe ich einen Apfelkuchen ohne Eier?", None),
]


def load_records():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_memory_collection(name, records, embeddings):
    collection = chromadb.EphemeralClient().create_collection(
        name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    collection.add(
        ids=[r["instance_id"] for r in records],
        documents=[r["question"] + "\n" + r["answer"] for r in records],
        embeddings=embeddings,
        metadatas=[{"instance_id": r["instance_id"], "question": r["question"]} for r in records],
    )
    return collection


def search(collection, query_embedding, top_k):
    result = collection.query(
        query_embeddings=query_embedding, n_results=top_k, include=["metadatas", "distances"]
    )
    return [
        {"rank": i, "instance_id": m["instance_id"], "question": m["question"], "distance": d}
        for i, (m, d) in enumerate(zip(result["metadatas"][0], result["distances"][0]), start=1)
    ]


def expected_rank(hits, expected):
    return next((h["rank"], h["distance"]) for h in hits if h["instance_id"] == expected)


def run_comparison(strategies, header):
    """strategies: {name: search_fn(query) -> hits}. Prints top-3 and a summary table."""
    summary = []
    for name, query, expected in TESTS:
        print(f"\n{name}: {query}  (expected: {expected})")
        row = [name, expected or "—"]
        for strategy, search_fn in strategies.items():
            hits = search_fn(query)
            print(f"  [{strategy}]")
            for h in hits[:3]:
                mark = " <-- expected" if h["instance_id"] == expected else ""
                print(f"    {h['rank']}. {h['distance']:.4f}  {h['instance_id']:<24} {h['question'][:60]}{mark}")
            if expected:
                rank, distance = expected_rank(hits, expected)
                print(f"    expected rank {rank} (distance {distance:.4f})")
                row.append(str(rank))
            else:
                row.append(f"top-1 {hits[0]['distance']:.4f}")
        summary.append(row)

    print(f"\n| Query | Expected | {' | '.join(header)} |")
    print("|---|---|" + "---|" * len(header))
    for row in summary:
        print("| " + " | ".join(row) + " |")


def main():
    records = load_records()
    minilm = SentenceTransformer(MINILM_MODEL_NAME)

    def embed(texts):
        return minilm.encode(texts, normalize_embeddings=True).tolist()

    qa = build_memory_collection(
        "minilm_qa", records, embed([r["question"] + "\n" + r["answer"] for r in records])
    )
    question_only = build_memory_collection(
        "minilm_question_only", records, embed([r["question"] for r in records])
    )

    run_comparison(
        {
            "Q+A": lambda q: search(qa, embed([q]), top_k=50),
            "Question-only": lambda q: search(question_only, embed([q]), top_k=50),
        },
        header=["Q+A rank", "Question-only rank"],
    )


if __name__ == "__main__":
    main()
