import os
import fitz  # PyMuPDF
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tempfile
import re
import pandas as pd
import plotly.express as px

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
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

# Compute cosine similarity
def compute_similarity(resume_text, jd_text):
    resume_vec = model.encode([resume_text])[0]
    jd_vec = model.encode([jd_text])[0]
    return cosine_similarity([resume_vec], [jd_vec])[0][0]

# Extract keywords from job description
def extract_keywords(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation
    keywords = set(text.split())
    stopwords = {
        "the", "and", "is", "in", "on", "for", "to", "a", "of", "with", "by", "an",
        "be", "as", "this", "that", "you", "your", "are", "or", "we", "us"
    }
    return keywords - stopwords

# Highlight keywords in resume
def highlight_keywords(resume_text, jd_keywords):
    resume_text = resume_text.strip()

    # Remove links & emails
    resume_text = re.sub(r'https?://\S+|www\.\S+', '', resume_text)
    resume_text = re.sub(r'\S+@\S+', '', resume_text)

    words = resume_text.split()
    highlighted_text = ""

    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        if clean_word in jd_keywords:
            highlighted_text += f"**:orange[{word}]** "
        else:
            highlighted_text += word + " "

    return highlighted_text

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI-Based Resume Screening System")

jd_input = st.text_area("🧾 Enter Job Description", height=250)
resumes = st.file_uploader("📤 Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)

if st.button("🔍 Analyze"):
    if not jd_input or not resumes:
        st.warning("Please provide both a job description and at least one resume.")
    else:
        scores = []
        jd_keywords = extract_keywords(jd_input)

        for resume_file in resumes:
            resume_text = extract_text_from_pdf(resume_file)
            similarity = compute_similarity(resume_text, jd_input)
            scores.append({
                "Resume": resume_file.name,
                "Match Score": round(similarity * 100, 2),
                "Text": resume_text
            })

        scores.sort(key=lambda x: x["Match Score"], reverse=True)

        st.subheader("📊 Resume Match Scores")
        for item in scores:
            st.write(f"**{item['Resume']}** — Match Score: **{item['Match Score']}%**")
            st.markdown("**🧠 Matched Keywords Highlighted:**")
            highlighted_resume = highlight_keywords(item["Text"], jd_keywords)
            st.markdown(highlighted_resume)
            st.markdown("---")

        # 📈 Better Plotly Chart
        st.subheader("📈 Resume Role Fit Score (out of 100%)")

        chart_data = pd.DataFrame(scores)[["Resume", "Match Score"]]
        auto_range = [0, max(100, max(chart_data["Match Score"]) + 10)]

        fig = px.bar(
            chart_data,
            x="Match Score",
            y="Resume",
            orientation='h',
            text="Match Score",
            color="Match Score",
            color_continuous_scale='Blues',
            title="🎯 Resume Role Fit Score (Out of 100%)"
        )

        fig.update_layout(
            xaxis=dict(title="Match Score (%)", range=auto_range),
            yaxis=dict(title="Resumes"),
            title_font_size=20,
            plot_bgcolor="white",
            showlegend=False,
            height=300 if len(chart_data) == 1 else 400 + len(chart_data) * 20,
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='inside',
            marker_line_width=0.5
        )

        if len(chart_data) == 1:
            fig.update_coloraxes(showscale=False)

        st.plotly_chart(fig, use_container_width=False)

