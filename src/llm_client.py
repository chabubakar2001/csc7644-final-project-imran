import os

from openai import OpenAI


def get_openai_client():
    """
    Create OpenAI client.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    return client