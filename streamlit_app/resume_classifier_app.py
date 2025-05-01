import streamlit as st
import pandas as pd
import re
import pickle
import os
import chardet
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load models and vectorizer
log_model = pickle.load(open('log_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Utility Functions
def clean_resume(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def generate_wordcloud(category, df):
    text = " ".join(df[df['Category'] == category]['cleaned_resume'])
    wordcloud = WordCloud(width=800, height=400, background_color='#f8f9fa', colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def extract_text(file, file_type):
    if file_type == 'pdf':
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    elif file_type == 'docx':
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_type == 'txt':
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            return raw_data.decode('ISO-8859-1')
    else:
        return ""

# Custom CSS Styling
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #00008B;
        }
        .stApp {
            max-width: 1000px;
            margin: auto;
            padding: 2rem;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        h1, h2, h3 {
            color: #1f4e79;
        }
        .stButton>button {
            background-color: #1f4e79;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stSidebar {
            background-color:   #28a745!important;
        }
        .metric-label {
            font-size: 1rem;
            color: #1f4e79;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #f0f2f6;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📄 AI-Powered Resume Classifier")
st.markdown("Upload your resume and let the AI predict the job category it fits best into. Choose between **Logistic Regression** and **Random Forest** classifiers.")

# Sidebar
st.sidebar.header("Upload Your Resume")
uploaded_file = st.sidebar.file_uploader("Supported formats: TXT, PDF, DOCX", type=["txt", "pdf", "docx"])

# Resume Handling
if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()
    resume_text = extract_text(uploaded_file, ext)

    if resume_text:
        st.success("✅ Resume uploaded and processed successfully!")
        cleaned_resume = clean_resume(resume_text)
        cleaned_resume = remove_stopwords(cleaned_resume)
        resume_vectorized = tfidf.transform([cleaned_resume])

        # Predict with models
        log_pred = log_model.predict(resume_vectorized)
        rf_pred = rf_model.predict(resume_vectorized)

        # Display results
        st.subheader("🧠 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Logistic Regression Prediction**")
            st.metric(label="Predicted Category", value=log_pred[0])
        with col2:
            st.markdown("**Random Forest Prediction**")
            st.metric(label="Predicted Category", value=rf_pred[0])

        with st.expander("📝 Preview of Uploaded Resume"):
            st.code(resume_text[:1500], language='text')
    else:
        st.error("Could not extract text from the uploaded file.")

# WordCloud Visualization
df = pd.read_csv("ResumeDataSet.csv")
df['cleaned_resume'] = df['Resume'].apply(clean_resume).apply(remove_stopwords)

with st.expander("📊 Explore WordCloud for Data Science"):
    generate_wordcloud("Data Science", df)

st.markdown("---")
st.markdown("Made using Streamlit")
