"""
Main RAG pipeline.
"""

import numpy as np

from embeddings import create_embedding
from retriever import Retriever
from prompts import build_prompt


class RAGPipeline:
    """
    RAG system for PDF-based research paper question answering.
    """

    def __init__(self, client, chunks):
        """
        Create embeddings and initialize retriever.

        Args:
            client: OpenAI client.
            chunks: List of chunk dictionaries.
        """
        self.client = client
        self.chunks = chunks

        print("Creating embeddings...")

        embedding_list = []

        for chunk in chunks:
            embedding = create_embedding(client, chunk["text"])
            embedding_list.append(embedding)

        embeddings = np.array(embedding_list, dtype="float32")

        self.retriever = Retriever(embeddings, chunks)

    def answer_question(self, question, top_k=3):
        """
        Answer a user question using retrieved chunks.

        Args:
            question: User question.
            top_k: Number of chunks to retrieve.

        Returns:
            Tuple containing answer and retrieved chunks.
        """
        query_embedding = create_embedding(self.client, question)

        retrieved_chunks = self.retriever.search(
            query_embedding,
            top_k=top_k,
        )

        prompt = build_prompt(question, retrieved_chunks)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the supplied paper "
                        "evidence and cite chunk IDs."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content

        return answer, retrieved_chunks