import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import google.generativeai as genai
from streamlit_option_menu import option_menu

os.environ["TRANSFORMERS_NO_TF"] = "1"

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Brochure QA",
        options=["📄 Upload & Read", "🔍 Ask a Question"],
        icons=["file-earmark-arrow-up", "chat-dots"],
        default_index=0
    )

# --- Constants ---
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
api_key = st.secrets["api"].get("GEMINIKEY", os.environ.get("GEMINIKEY"))
genai.configure(api_key=api_key)

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# --- Embedding Function ---
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).numpy()

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# --- Chunking Text ---
def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# --- GenAI Setup ---
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    gen_model = genai.GenerativeModel("models/gemini-1.5-flash")
else:
    gen_model = None

# --- Session State for Data ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# --- Upload & Read PDF ---
if selected == "📄 Upload & Read":
    st.title("📄 Upload Product Brochure")
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_pdf:
        text = extract_text_from_pdf(uploaded_pdf)
        st.session_state.pdf_text = text

        st.subheader("📘 Extracted Text Snippet")
        with st.expander("📝 View Extracted PDF Text"):
            st.text(st.session_state.pdf_text[:1000] + "...\n\n(truncated)")

        st.subheader("🔗 Generating Chunks & Embeddings")
        chunks = chunk_text(text, max_words=100)
        embeddings = [get_embedding(chunk)[0] for chunk in chunks]

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

        st.success("Chunks and embeddings stored in memory ✅")

# --- Ask a Question ---
elif selected == "🔍 Ask a Question":
    st.title("🔍 Ask about the Brochure")

    if not st.session_state.chunks:
        st.warning("Please upload a brochure and generate embeddings first.")
    else:
        query = st.text_input("Ask a question about the product:")
        if query:
            q_embed = get_embedding(query)
            sims = cosine_similarity([q_embed[0]], st.session_state.embeddings)[0]
            top_k = 3
            top_indices = sims.argsort()[-top_k:][::-1]

            context = "\n".join([st.session_state.chunks[i] for i in top_indices])

            prompt = f"""
            You are a helpful assistant. Answer the following question based on the given brochure content.

            Brochure Content:
            {context}

            Question: {query}
            """

            if gen_model:
                response = gen_model.generate_content(prompt)
                answer = response.text
            else:
                answer = "❌ Gemini API Key not configured."

            st.subheader("📬 Answer")
            st.write(answer)

            with st.expander("🔎 Top Matching Chunks"):
                for i in top_indices:
                    st.markdown(f"**Chunk {i+1} (score: {sims[i]:.2f})**")
                    st.code(st.session_state.chunks[i])

st.caption("📁 Brochure QA module powered by embeddings + Gemini")
