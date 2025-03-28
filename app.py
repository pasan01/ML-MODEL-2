import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber
from docx import Document

# Load the pre-trained model and vectorizer
model_path = 'resume_analysis_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Streamlit app title and description
st.title("Resume Job Role Prediction")
st.write("Upload a resume (PDF, DOCX, or TXT), and the app will predict the job role most suited for the candidate.")

# Upload resume file
uploaded_file = st.file_uploader("Upload a Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Determine file type and extract text
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode('utf-8')

    # Show the resume content
    st.write("Resume content:")
    st.write(resume_text)

    # Transform the uploaded resume using the trained TF-IDF vectorizer
    transformed_resume = tfidf_vectorizer.transform([resume_text])

    # Predict the role using the pre-trained model
    predicted_role = model.predict(transformed_resume)

    # Display the prediction result
    st.write(f"Predicted job role: **{predicted_role[0]}**")
