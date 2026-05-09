import numpy as np

from embeddings import create_embedding
from retriever import Retriever
from prompts import build_prompt


class RAGPipeline:
    """
    Main RAG system.
    """

    def __init__(self, client, chunks):

        self.client = client
        self.chunks = chunks

        print("Creating embeddings...")

        embedding_list = []

        for chunk in chunks:

            embedding = create_embedding(
                client,
                chunk
            )

            embedding_list.append(embedding)

        embeddings = np.array(
            embedding_list,
            dtype="float32"
        )

        self.retriever = Retriever(
            embeddings,
            chunks
        )

    def answer_question(self, question):

        query_embedding = create_embedding(
            self.client,
            question
        )

        retrieved_chunks = self.retriever.search(
            query_embedding,
            top_k=3
        )

        prompt = build_prompt(
            question,
            retrieved_chunks
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        return answer, retrieved_chunks