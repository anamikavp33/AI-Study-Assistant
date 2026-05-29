# AI Study Assistant

An AI-powered RAG chatbot built using Python and Streamlit.

## Overview

AI Study Assistant helps students:

* Ask questions from uploaded PDFs
* Generate summaries
* Create quiz questions
* Learn concepts using AI

This project uses RAG (Retrieval Augmented Generation) architecture for intelligent document-based responses.

---

## Features

* AI chatbot
* PDF question answering
* Chat memory
* Quiz generator
* Notes summarizer
* RAG architecture
* ChromaDB vector database
* Streamlit web interface

---

## Technologies Used

* Python
* Streamlit
* OpenRouter API
* LangChain
* ChromaDB
* Sentence Transformers
* PyPDF
* HuggingFace Embeddings

---

## Project Structure

```text
AI_Study_Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── chroma_db/
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add:

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

## Run Project

```bash
streamlit run app.py
```

---

## Future Improvements

* Voice assistant
* Multi-PDF support
* AI flashcards
* Dark mode UI
* Online deployment
* User authentication

---

## Author

Anamika V P
