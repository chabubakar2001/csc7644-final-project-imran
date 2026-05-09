import faiss
import numpy as np


class Retriever:
    """
    Simple FAISS retriever.
    """

    def __init__(self, embeddings, chunks):

        self.chunks = chunks

        self.index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:

            results.append(
                self.chunks[idx]
            )

        return results