# Project Setup Guide

## Installation

To install the required dependencies for this project, simply run the following command:


### .env Configuration

Before running the app, ensure you configure your OpenAI API Key in the `.env` file.

1. Create a `.env` file in the root of the project directory.
2. Add your OpenAI API Key in the `.env` file like this:


Once this is done, you're all set for the magic to happen when you run the project!

## File Structure Overview

- **RAG.ipynb**: This notebook contains the core Retrieval-Augmented Generation (RAG) logic. It includes:
  - Preprocessing the Bangla PDF to extract text using Tesseract OCR.
  - Chunking and vectorizing the document content.
  - Generating answers using a retrieval-augmented generation pipeline.

- **app.py**: This file contains the Streamlit frontend for user interaction. It allows you to upload a PDF, interact with the RAG system, and view answers to your queries in both English and Bengali.

## Run the Streamlit App

- Configure `.env` with your OpenAI API Key.
- Run the following command to start the Streamlit frontend:



## Bangla PDF Text Extraction (Why We Use Tesseract)

### Problem with Traditional PDF Loaders

Traditional loaders like `PyPDFLoader` or `pdfplumber` do not work well for Bangla PDFs. Many HSC-style Bangla documents use:

- Non-Unicode or embedded fonts (e.g., SutonnyMJ).
- Glyph-based rendering that causes unreadable text.

As a result, you get corrupted or broken text when using:
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("HSC26-Bangla1st-Paper.pdf")
docs = loader.load() # ❌ Broken or unreadable Bangla output



### Why Not Use UnstructuredPDFLoader?

The `UnstructuredPDFLoader` from unstructured.io offers more advanced parsing. However, full capabilities (including OCR) are locked behind a paid API and require additional setup (e.g., Docker, authentication).

#### Solution: Tesseract OCR (Free & Accurate for Bangla)

To overcome these challenges, Tesseract OCR is used to extract text from Bangla PDFs as images. This solution:

- Bypasses font encoding issues.
- Provides clean, readable Bangla text.
- Is open-source and free.

#### Install Poppler for PDF → Image Conversion

- **Download Poppler for Windows:** Link (please insert actual download link)
- **Extract it to:** `C:\poppler-24.08.0`
- **Use this path in your code:**


poppler_path = r"C:/poppler-24.08.0/Library/bin"



#### Install Tesseract OCR with Bengali Language

- **Download Tesseract OCR:** UB Mannheim Tesseract Download (please insert actual download link)
- **Install it to:** `C:\Program Files\Tesseract-OCR`
- During installation, ensure **Bengali (ben)** language is selected.
- If Bengali (ben) is not available during installation, download `ben.traineddata` manually from [here](https://github.com/tesseract-ocr/tessdata) and place it in:

C:\Program Files\Tesseract-OCR\tessdata\

#### Add Tesseract to PATH

- Add `C:\Program Files\Tesseract-OCR` to your system environment variables (`PATH`).
- Restart your terminal and verify installation by running:




