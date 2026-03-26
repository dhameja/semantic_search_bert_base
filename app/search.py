


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from app.embedding import get_embedding
from app.data import sentences


# Precompute embeddings (like vector DB)
sentence_embeddings = np.vstack(
    [get_embedding(s) for s in sentences]
)


def semantic_search(query):

    query_embedding = get_embedding(query)

    scores = cosine_similarity(
        query_embedding,
        sentence_embeddings
    )[0]

    results = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {"sentence": s, "score": float(score)}
        for s, score in results
    ]