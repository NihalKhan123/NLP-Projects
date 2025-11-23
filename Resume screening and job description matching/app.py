import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for PDF reading
import os
import tempfile

# -----------------------------
# Load BERT Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# -----------------------------
# PDF → Text Extractor
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


# -----------------------------
# Compute Cosine Similarity
# -----------------------------
def compute_similarity(job_desc, resume_texts, resume_names):
    jd_embedding = model.encode(job_desc, convert_to_tensor=True)

    results = []
    for text, name in zip(resume_texts, resume_names):
        emb = model.encode(text, convert_to_tensor=True)
        score = float(util.cos_sim(jd_embedding, emb))
        results.append((name, score))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:5]  # Top 5 resumes


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume–Job Description Matcher", layout="wide")

st.title("📌 Resume Screener & Job Description Matcher")
st.write("Upload multiple resumes and enter a Job Description to get the Top 5 most relevant resumes.")

st.markdown("---")

# HR enters JD
job_description = st.text_area("✍️ Enter Job Description", height=200)

# HR uploads PDFs
uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🔍 Match Resumes"):
    if not job_description.strip():
        st.error("❌ Please enter a Job Description")
    elif not uploaded_files:
        st.error("❌ Please upload at least 1 resume")
    else:
        st.info("⏳")

        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            try:
                text = extract_text_from_pdf(file)
                resume_texts.append(text)
                resume_names.append(file.name)
            except:
                st.warning(f"⚠️ Could not read {file.name}")

        # Compute similarity
        top_results = compute_similarity(job_description, resume_texts, resume_names)

        st.success("✅ Matching Completed!")

        st.markdown("## 🏆 Top 5 Best Matching Resumes")

        for rank, (name, score) in enumerate(top_results, start=1):
            st.write(f"### #{rank}. **{name}** — Similarity Score: `{score:.4f}`")

        # Download results
        df = pd.DataFrame(top_results, columns=["Resume Name", "Similarity Score"])
        st.download_button(
            "⬇️ Download Results (CSV)",
            df.to_csv(index=False),
            "top_5_resume_matches.csv",
            "text/csv"
        )
