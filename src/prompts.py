"""
Prompt templates for grounded RAG answers.
"""


def build_prompt(question, contexts):
    """
    Build prompt with retrieved evidence.

    Args:
        question: User question.
        contexts: Retrieved chunks.

    Returns:
        Prompt string.
    """
    context_text = ""

    for chunk in contexts:
        context_text += (
            f"\n[Chunk {chunk['chunk_id']}]\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are a careful research paper assistant.

Answer the user question using ONLY the retrieved evidence below.
Do not invent unsupported claims. If the evidence is insufficient, say so.

When you use information from a chunk, cite it using the format [Chunk X].

Retrieved Evidence:
{context_text}

User Question:
{question}

Required format:
Short Answer:
Evidence-Based Explanation:
Limitations or Uncertainty:
"""

    return prompt.strip()