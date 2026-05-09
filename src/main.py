from dotenv import load_dotenv

from pdf_loader import load_pdf_text
from chunker import chunk_text
from llm_client import get_openai_client
from rag_pipeline import RAGPipeline


def main():

    load_dotenv()

    print("=" * 60)
    print("RAG Research Paper Assistant")
    print("=" * 60)

    pdf_path = input("Enter PDF path: ")

    print("\nLoading PDF...")

    text = load_pdf_text(pdf_path)

    chunks = chunk_text(text)

    print(f"Chunks created: {len(chunks)}")

    client = get_openai_client()

    rag = RAGPipeline(
        client,
        chunks
    )

    while True:

        question = input(
            "\nAsk a question (or type exit): "
        )

        if question.lower() == "exit":
            break

        answer, retrieved_chunks = rag.answer_question(
            question
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("RETRIEVED CHUNKS")
        print("=" * 60)

        for idx, chunk in enumerate(retrieved_chunks):

            print(f"\nChunk {idx + 1}:\n")
            print(chunk[:500])


if __name__ == "__main__":
    main()