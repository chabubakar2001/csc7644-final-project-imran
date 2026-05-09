"""
Command-line interface for the RAG research paper assistant.
"""

import os
import time

from dotenv import load_dotenv

from pdf_loader import load_pdf_text
from chunker import chunk_text
from llm_client import get_openai_client
from rag_pipeline import RAGPipeline


def save_output(question, answer, retrieved_chunks, latency):
    """
    Save answer and retrieved evidence to outputs/sample_output.txt.

    Args:
        question: User question.
        answer: Generated answer.
        retrieved_chunks: Retrieved evidence chunks.
        latency: Query latency in seconds.
    """
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/sample_output.txt", "w", encoding="utf-8") as file:
        file.write("RAG Research Paper Assistant Output\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Question:\n{question}\n\n")
        file.write(f"Answer:\n{answer}\n\n")
        file.write(f"Latency: {latency:.2f} seconds\n\n")
        file.write("Retrieved Evidence:\n")

        for chunk in retrieved_chunks:
            file.write("\n" + "-" * 60 + "\n")
            file.write(
                f"Chunk {chunk['chunk_id']} | Distance Score: {chunk['score']:.4f}\n"
            )
            file.write(chunk["text"][:1200])
            file.write("\n")


def main():
    """
    Run the RAG research paper assistant.
    """
    load_dotenv()

    print("=" * 60)
    print("RAG Research Paper Assistant")
    print("=" * 60)

    pdf_path = input("Enter PDF path: ")

    print("\nLoading PDF...")
    text = load_pdf_text(pdf_path)

    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"Chunks created: {len(chunks)}")

    client = get_openai_client()

    rag = RAGPipeline(client, chunks)

    while True:
        question = input("\nAsk a question or type exit: ")

        if question.lower().strip() == "exit":
            print("Exiting.")
            break

        start_time = time.time()

        answer, retrieved_chunks = rag.answer_question(
            question,
            top_k=3,
        )

        latency = time.time() - start_time

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("RETRIEVED EVIDENCE")
        print("=" * 60)

        for chunk in retrieved_chunks:
            print(
                f"\n[Chunk {chunk['chunk_id']}] "
                f"Distance Score: {chunk['score']:.4f}"
            )
            print(chunk["text"][:500])

        print("\n" + "=" * 60)
        print(f"Latency: {latency:.2f} seconds")
        print("Output saved to outputs/sample_output.txt")
        print("=" * 60)

        save_output(question, answer, retrieved_chunks, latency)


if __name__ == "__main__":
    main()