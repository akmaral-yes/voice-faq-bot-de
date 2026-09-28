"""Query the persistent Chroma FAQ index (built by build_index.py).

Embedding model: intfloat/multilingual-e5-base (768-dim, max 512 tokens).
As the model card requires, corpus texts are prefixed with "passage: " and
queries with "query: "; embeddings are L2-normalized.

Distance metric: cosine distance = 1 - cosine_similarity.
Range 0..2; LOWER means a closer match (0 = same direction).
The raw distance is returned unchanged; no threshold is applied here.

Usage:  uv run python retrieval.py "Ihre Frage"
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "faq"

_model = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts):
    texts = ["passage: " + t for t in texts]
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query):
    return get_model().encode(["query: " + query], normalize_embeddings=True).tolist()


def get_collection():
    """Open the existing collection; fails if build_index.py has not been run."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    return _collection


def retrieve(query: str, top_k: int = 3):
    result = get_collection().query(
        query_embeddings=embed_query(query),
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    hits = []
    for rank, (meta, distance) in enumerate(
        zip(result["metadatas"][0], result["distances"][0]), start=1
    ):
        hits.append({
            "rank": rank,
            "instance_id": meta["instance_id"],
            "source_instance_ids": meta["source_instance_ids"],
            "question": meta["question"],
            "answer": meta["answer"],
            "use_case": meta["use_case"],
            "distance": distance,
        })
    return hits


if __name__ == "__main__":
    for hit in retrieve(" ".join(sys.argv[1:])):
        print(f"{hit['rank']}. {hit['distance']:.4f}  {hit['instance_id']}  {hit['question']}")
