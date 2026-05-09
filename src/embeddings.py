import numpy as np


def create_embedding(client, text):
    """
    Create embedding for text.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return np.array(embedding, dtype="float32")