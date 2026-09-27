"""Stage 2C experiment: MiniLM vs. multilingual-e5-base, both on question + answer.

Both collections are temporary and in-memory, so this experiment is independent
of the persistent project index. E5 inputs use the model card's "query: " /
"passage: " prefixes. All embeddings are L2-normalized; cosine distance (lower = closer).

Run from the repository root:  uv run python -m experiments.compare_models
"""

import time

from sentence_transformers import SentenceTransformer

from experiments.compare_retrieval import (
    MINILM_MODEL_NAME,
    build_memory_collection,
    load_records,
    run_comparison,
    search,
)

E5_MODEL_NAME = "intfloat/multilingual-e5-base"


def count_truncated(model, texts):
    limit = model.max_seq_length
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    return sum(n > limit for n in lengths), max(lengths)


def timed_encode(model, texts):
    model.encode(texts[:1], normalize_embeddings=True)  # warm-up
    start = time.perf_counter()
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    return embeddings, time.perf_counter() - start


def main():
    records = load_records()
    documents = [r["question"] + "\n" + r["answer"] for r in records]

    minilm = SentenceTransformer(MINILM_MODEL_NAME)
    minilm_embeddings, minilm_corpus_s = timed_encode(minilm, documents)
    minilm_collection = build_memory_collection("minilm_qa", records, minilm_embeddings)

    e5 = SentenceTransformer(E5_MODEL_NAME)
    passages = ["passage: " + d for d in documents]
    e5_embeddings, e5_corpus_s = timed_encode(e5, passages)
    e5_collection = build_memory_collection("e5_qa", records, e5_embeddings)

    query_times = {"MiniLM Q+A": [], "E5 Q+A": []}

    def timed(name, fn):
        def wrapper(query):
            start = time.perf_counter()
            hits = fn(query)
            query_times[name].append(time.perf_counter() - start)
            return hits
        return wrapper

    run_comparison(
        {
            "MiniLM Q+A": timed("MiniLM Q+A", lambda q: search(
                minilm_collection, minilm.encode([q], normalize_embeddings=True).tolist(), top_k=50)),
            "E5 Q+A": timed("E5 Q+A", lambda q: search(
                e5_collection, e5.encode(["query: " + q], normalize_embeddings=True).tolist(), top_k=50)),
        },
        header=["MiniLM Q+A rank", "E5 Q+A rank"],
    )

    for name, model, texts in [("MiniLM", minilm, documents), ("E5", e5, passages)]:
        truncated, longest = count_truncated(model, texts)
        print(f"\n{name}: max_seq_length {model.max_seq_length}, "
              f"truncated {truncated} of {len(texts)} (longest {longest} tokens)")
    print(f"Corpus embedding (50 docs): MiniLM {minilm_corpus_s:.2f}s, E5 {e5_corpus_s:.2f}s")
    for name, times in query_times.items():
        print(f"Mean query time (embed + search), {name}: {1000 * sum(times) / len(times):.0f} ms")


if __name__ == "__main__":
    main()
