def build_prompt(question, contexts):
    """
    Build grounded RAG prompt.
    """

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a research paper assistant.

Answer the user's question ONLY using the context below.

If the answer is not contained in the context, say:
"I could not find enough information in the paper."

Context:
{context_text}

Question:
{question}

Answer:
"""

    return prompt