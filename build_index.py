"""Embed data/corpus.jsonl and persist it as a Chroma collection.

One FAQ = one document; document text = question + "\\n" + answer (no chunking).
embed_passages() adds the E5 "passage: " prefix; the stored document has no prefix.
Deletes and recreates the whole index directory on every run, so each record is
indexed exactly once and no files from an earlier build remain.
"""

import json
import shutil

import chromadb

from retrieval import CHROMA_PATH, COLLECTION_NAME, MODEL_NAME, embed_passages

CORPUS_PATH = "data/corpus.jsonl"


def main():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    documents = [r["question"] + "\n" + r["answer"] for r in records]
    embeddings = embed_passages(documents)

    # delete_collection() leaves old HNSW folders on disk, so wipe the directory instead
    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # we always pass our own embeddings
    )

    collection.add(
        ids=[r["instance_id"] for r in records],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "instance_id": r["instance_id"],
                "source_instance_ids": r["source_instance_ids"],
                "use_case": r["use_case"],
                "question": r["question"],
                "answer": r["answer"],
            }
            for r in records
        ],
    )

    print(f"Embedding model:  {MODEL_NAME}")
    print(f"Chroma path:      {CHROMA_PATH}")
    print(f"Distance metric:  {collection.configuration['hnsw']['space']}")
    print(f"Corpus records:   {len(records)}")
    print(f"Collection size:  {collection.count()}")


if __name__ == "__main__":
    main()
