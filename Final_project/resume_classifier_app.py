import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import chardet
import os

# Use absolute paths or paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_model_path = os.path.join(BASE_DIR, 'log_model.pkl')
log_model = pickle.load(open(log_model_path, 'rb'))
# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load pre-trained models and vectorizer
log_model = pickle.load(open('log_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Function to clean the resume text
def clean_resume(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text.strip()

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Function to generate a word cloud for a specific category
def generate_wordcloud(category, df):
    text = " ".join(df[df['Category'] == category]['cleaned_resume'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {category} Resumes")
    st.pyplot(plt)

# App title
st.title("Resume Classifier")

# Sidebar for user input
st.sidebar.header("Upload Resume")
uploaded_file = st.sidebar.file_uploader("Choose a resume file", type=["txt", "pdf", "docx"])

# Variable to track if file is processed successfully
file_processed = False

if uploaded_file and not file_processed:
    # Set the flag to True so the file is not processed again
    file_processed = True
    
    # Detect file encoding using chardet
    raw_data = uploaded_file.read()
    detected_encoding = chardet.detect(raw_data)['encoding']
    
    # Attempt to decode using the detected encoding, fall back to 'ISO-8859-1' if needed
    try:
        if detected_encoding is None:
            detected_encoding = 'utf-8'
        resume_text = raw_data.decode(detected_encoding)
        st.success("File uploaded and processed successfully!")
    except UnicodeDecodeError:
        st.error("There was an issue decoding the file with the detected encoding. Trying with ISO-8859-1...")
        resume_text = raw_data.decode('ISO-8859-1')
        st.success("File uploaded and processed successfully!")

    # Clean and preprocess resume text
    cleaned_resume = clean_resume(resume_text)
    cleaned_resume = remove_stopwords(cleaned_resume)

    # Vectorize the resume text using the pre-trained TF-IDF vectorizer
    resume_vectorized = tfidf.transform([cleaned_resume])

    # Predict with Logistic Regression
    log_pred = log_model.predict(resume_vectorized)
    # Predict with Random Forest
    rf_pred = rf_model.predict(resume_vectorized)

    # Display predictions
    st.write(f"Prediction using Logistic Regression: {log_pred[0]}")
    st.write(f"Prediction using Random Forest: {rf_pred[0]}")

# Load the dataset for visualization
df = pd.read_csv("C:\\Users\\majok\\Downloads\\ResumeDataSet.csv")

# Clean the dataset
df['cleaned_resume'] = df['Resume'].apply(clean_resume)
df['cleaned_resume'] = df['cleaned_resume'].apply(remove_stopwords)

# Display word cloud for Data Science category
generate_wordcloud('Data Science', df)

# Upload the resume file again (this will not trigger an upload if the file has already been processed)
if uploaded_file and not file_processed:
    # Read the raw data from the file
    raw_data = uploaded_file.read()

    # Use chardet to detect the encoding
    detected_encoding = chardet.detect(raw_data)['encoding']

    # If encoding is None, default to 'utf-8'
    if detected_encoding is None:
        detected_encoding = 'utf-8'

    # Decode the file using the detected encoding
    resume_text = raw_data.decode(detected_encoding)

    # Display the resume text (if it's text-based)
    st.text_area("Resume Text", resume_text, height=300)
