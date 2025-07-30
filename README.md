# 📄 AI Resume Screening System

A smart AI-powered tool that compares multiple resumes to a job description using NLP techniques and ranks them based on similarity score.

---

## 🔍 About the Project

This project helps HR teams and recruiters automatically screen resumes by comparing them with a job description using **sentence embeddings** and **cosine similarity**. It extracts text from PDF resumes and calculates how closely they match the job description using advanced **NLP** models.

This project was built as part of **HackWithInfy 2025** by Ashritha for the **Digital Specialist Role**.

---

## 🚀 Features

- 📤 Upload one or multiple PDF resumes
- 📄 Paste the job description (JD)
- 📊 View match scores ranked by similarity
- 💻 Simple and elegant interface built using Streamlit

---

## 🛠️ Technologies Used

- Python 🐍
- [Streamlit](https://streamlit.io/) – for UI
- [Sentence Transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) – for embeddings
- Cosine Similarity from scikit-learn
- PyMuPDF (fitz) – for extracting PDF content

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/ashritha18953/ai_resumee_screening.git
cd ai_resumee_screening
