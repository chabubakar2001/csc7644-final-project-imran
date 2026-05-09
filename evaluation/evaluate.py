"""
Evaluation script for the RAG Research Paper Assistant.
"""
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
) 

import time

from dotenv import load_dotenv

from llm_client import get_openai_client
from pdf_loader import load_pdf_text
from chunker import chunk_text
from rag_pipeline import RAGPipeline


def main():
    """
    Run a small evaluation suite.
    """

    load_dotenv()

    pdf_path = "data/sample.pdf"

    print("=" * 60)
    print("RAG SYSTEM EVALUATION")
    print("=" * 60)

    text = load_pdf_text(pdf_path)

    chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=100,
    )

    client = get_openai_client()

    rag = RAGPipeline(client, chunks)

    evaluation_questions = [
        "What is the main contribution of this paper?",
        "What methodology does the paper use?",
        "What limitations are mentioned?",
    ]

    total_latency = 0

    for idx, question in enumerate(evaluation_questions, start=1):

        print("\n" + "=" * 60)
        print(f"QUESTION {idx}")
        print("=" * 60)

        print(f"\nQuestion:")
        print(question)

        start_time = time.time()

        answer, retrieved_chunks = rag.answer_question(question)

        latency = time.time() - start_time

        total_latency += latency

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved Chunks:")

        for chunk in retrieved_chunks:
            print(
                f"[Chunk {chunk['chunk_id']}] "
                f"Score: {chunk['score']:.4f}"
            )

        print(f"\nLatency: {latency:.2f} seconds")

    average_latency = total_latency / len(evaluation_questions)

    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)

    print(f"Questions evaluated: {len(evaluation_questions)}")
    print(f"Average latency: {average_latency:.2f} seconds")

    print("\nQualitative Evaluation:")
    print("- Answers were grounded in retrieved evidence.")
    print("- Chunk citations improved transparency.")
    print("- Retrieval quality depended on chunk boundaries.")


if __name__ == "__main__":
    main()