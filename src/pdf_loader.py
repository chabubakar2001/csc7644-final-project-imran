import fitz


def load_pdf_text(pdf_path):
    """
    Extract text from PDF file.
    """

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    return text