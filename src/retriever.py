"""
FAISS retriever for semantic search.
"""

import faiss
import numpy as np


class Retriever:
    """
    FAISS-based retriever using L2 distance.
    """

    def __init__(self, embeddings, chunks):
        """
        Initialize retriever.

        Args:
            embeddings: NumPy array of text embeddings.
            chunks: List of chunk dictionaries.
        """
        self.chunks = chunks
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=3):
        """
        Search for top-k relevant chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of chunks to retrieve.

        Returns:
            List of retrieved chunk dictionaries with scores.
        """
        query_embedding = np.array([query_embedding], dtype="float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index >= 0:
                chunk = self.chunks[index].copy()
                chunk["score"] = float(distance)
                results.append(chunk)

        return results