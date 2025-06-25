import os
import fitz  
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tempfile


model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        doc = fitz.open(tmp_file_path)
        text = " ".join([page.get_text() for page in doc])
        doc.close()  
    finally:
        os.remove(tmp_file_path)
    
    return text


def compute_similarity(resume_text, jd_text):
    resume_vec = model.encode([resume_text])[0]
    jd_vec = model.encode([jd_text])[0]
    return cosine_similarity([resume_vec], [jd_vec])[0][0]


st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI-Based Resume Screening System")


jd_input = st.text_area("🧾 Enter Job Description", height=250)


resumes = st.file_uploader("📤 Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)


if st.button("🔍 Analyze"):
    if not jd_input or not resumes:
        st.warning("Please provide both a job description and at least one resume.")
    else:
        scores = []
        for resume_file in resumes:
            resume_text = extract_text_from_pdf(resume_file)
            similarity = compute_similarity(resume_text, jd_input)
            scores.append({
                "filename": resume_file.name,
                "score": round(similarity * 100, 2)
            })

        
        scores.sort(key=lambda x: x["score"], reverse=True)

        
        st.subheader("📊 Resume Match Scores")
        for item in scores:
            st.write(f"**{item['filename']}** — Match Score: **{item['score']}%**")
