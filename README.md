# RAG-Enabled Research Paper Assistant

## Overview

This project is a Retrieval-Augmented Generation (RAG) application developed as the final project for CSC 7644: Applied LLM Development.

The system helps users read and analyze technical research papers more efficiently by answering questions using the actual content of an uploaded PDF. Instead of relying on a prompt-only chatbot, the application retrieves relevant paper sections before generating grounded responses with supporting evidence.

The goal is to reduce hallucinations while improving the reliability and usefulness of LLM-generated answers for academic reading and literature review tasks.

---

# Key Features

- PDF text extraction using PyMuPDF
- Overlapping text chunking
- OpenAI embedding generation
- FAISS vector retrieval
- Grounded question answering using GPT-4o-mini
- Evidence-based responses with chunk citations
- Latency measurement
- Output logging for evaluation and demonstration

---

# Tech Stack

## Language
- Python 3.13

## Libraries and APIs
- OpenAI API
- FAISS
- NumPy
- python-dotenv
- PyMuPDF

---

# High-Level Architecture

```text
Research Paper PDF
        ↓
Text Extraction
        ↓
Chunking with Overlap
        ↓
OpenAI Embeddings
        ↓
FAISS Vector Index
        ↓
User Question
        ↓
Retriever
        ↓
GPT-4o-mini
        ↓
Grounded Answer + Evidence