import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

# =========================
# OPENROUTER CLIENT
# =========================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# =========================
# STREAMLIT PAGE
# =========================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📘",
    layout="wide"
)

st.title("📘 AI Study Assistant")

st.markdown("""
### 🤖 Your AI Learning Companion

Upload PDF notes and:
- Ask questions
- Generate summaries
- Create quizzes
- Learn faster with AI
""")

st.write("Upload PDF notes and chat with AI.")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF notes",
    type="pdf"
)

quiz_button = st.sidebar.button("📝 Generate Quiz")

summary_button = st.sidebar.button("📘 Summarize Notes")

clear_button = st.sidebar.button("🗑️ Clear Chat")

if st.session_state.get("messages"):

    chat_text = "\n\n".join(
        [f"{msg['role'].upper()}: {msg['content']}"
         for msg in st.session_state.messages]
    )

    st.sidebar.download_button(
        label="📥 Download Chat",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Clear chat
if clear_button:
    st.session_state.messages = []

# =========================
# PDF PROCESSING
# =========================

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Read PDF
        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector database
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        st.session_state.vector_db = vector_db

        st.sidebar.success("✅ PDF processed successfully!")

        st.sidebar.info(f"📄 Pages: {len(pdf_reader.pages)}")

        st.sidebar.info(f"🧩 Chunks Created: {len(chunks)}")

# =========================
# SUMMARY FEATURE
# =========================

if summary_button and st.session_state.vector_db:

    with st.spinner("Generating summary..."):

        docs = st.session_state.vector_db.similarity_search(
            "summary of document",
            k=5
        )

        context = ""

        for doc in docs:
            context += doc.page_content + "\n"

        with st.expander("📄 Retrieved Context"):
            st.write(context)

        prompt = f"""
        Summarize these notes clearly for students.

        Notes:
        {context}
        """

        completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        summary = completion.choices[0].message.content

        st.subheader("📘 Notes Summary")

        st.write(summary)

        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )


# =========================
# QUIZ GENERATOR
# =========================

if quiz_button and st.session_state.vector_db:

    with st.spinner("Generating quiz..."):

        docs = st.session_state.vector_db.similarity_search(
            "important concepts",
            k=5
        )

        context = ""

        for doc in docs:
            context += doc.page_content + "\n"

        prompt = f"""
        Create 10 multiple choice questions from these notes.

        Notes:
        {context}
        """

        completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        quiz = completion.choices[0].message.content

        st.subheader("📝 Quiz Questions")

        st.write(quiz)

# =========================
# DISPLAY OLD CHAT
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# CHAT INPUT
# =========================

user_input = st.chat_input("Ask questions from your PDF...")

# =========================
# AI CHATBOT
# =========================

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    try:

        context = ""

        # Retrieve relevant chunks
        if st.session_state.vector_db is not None:

            docs = st.session_state.vector_db.similarity_search(
                user_input,
                k=3
            )

            for doc in docs:
                context += doc.page_content + "\n"

        # Prompt
        prompt = f"""
       
You are an intelligent AI Study Assistant.

Instructions:
1. Explain concepts in simple student-friendly language.
2. Use bullet points whenever appropriate.
3. Give examples if helpful.
4. If the answer exists in the PDF context, prioritize that information.
5. If the PDF does not contain the answer, use your general knowledge.
6. Keep answers clear, structured, and easy to understand.

PDF Context:
{context}

Question:
{user_input}
"""

        # AI Response
        completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI study assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        ai_reply = completion.choices[0].message.content

        # Show AI response
        with st.chat_message("assistant"):
            st.write(ai_reply)

        # Save AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_reply
        })

    except Exception as e:

        st.error(f"Error: {e}")