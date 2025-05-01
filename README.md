
# Resume Screening App

A simple web application that categorizes uploaded resumes using Natural Language Processing (NLP) and Machine Learning. Users can upload `.pdf` or `.txt` files, and the app predicts the job role or category based on the resume content.

---
# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#Use)
- [Technologies Used](#technologies-used)

---
## Overview
This project streamlines the recruitment process by automating the **resume screening** phase. The app processes and classifies resumes using **NLP** techniques and a **multi-class machine learning classifier**, enabling faster and more consistent candidate filtering.

This app uses:

- **TF-IDF Vectorization** for text feature extraction
- **Random Forest** or **K-Nearest Neighbors (KNN)** classifiers for prediction
- **Streamlit** for the interactive web interface

It includes:

- A **training script** (`train_models.py`) for preprocessing and training the model
- A **Streamlit app** (`resume_classifier_app.py`) for deployment and user interaction

---

## Prerequisites

Make sure the following are installed:

- Python 3.7+
- pip (Python package manager)

Optional but recommended:

- A virtual environment tool (`venv`, `virtualenv`, or `conda`)

---

## Getting Started

### Step 1: Download and Navigate

After downloading the project folder named `Final-AI-project-using-Machine-learning`, open your terminal and navigate to the Streamlit app directory:

```bash
cd Final-AI-project-using-Machine-learning/streamlit_app
```

### Step 2: Setup Your Environment

(Optional) Create and activate a virtual environment:

```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

Run the model training script to prepare your classifier:

```bash
python train_models.py
```

### Step 5: Run the App

Start the Streamlit app using:

```bash
python -m streamlit run resume_classifier_app.py
```

Example:

```bash
Final-AI-project-using-Machine-learning\streamlit_app> python -m streamlit run resume_classifier_app.py
```

### Step 6: Use the App

- Upload a `.pdf` or `.txt` resume
- The app will predict the associated category or job role

---

## How It Works

The app pipeline includes:

- Text cleaning (stopword removal, lemmatization, etc.)
- TF-IDF vectorization
- Multi-class classification using Random Forest or KNN
- The trained model and vectorizer are saved and used in the app

---

## 🛠 Troubleshooting

- **Streamlit not found?**  
  Run: `pip install streamlit`

- **NLTK data errors?**  
  Ensure required corpora are downloaded in the notebook or:

  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

---

## License

This project is for educational and demo purposes.

---

### System Components

#### Training Notebook – `resume_screening.ipynb`
- Preprocesses the data (cleaning, label encoding, stopword removal, lemmatization).
- Extracts features using **TF-IDF**.
- Trains a **K-Nearest Neighbors** model with `OneVsRestClassifier`.
- Saves the trained model and vectorizer.

#### 🌐 Web App – `resume_classifier_app.py`
- Built with **Streamlit**.
- Allows resume file upload.
- Predicts job category from resume content using the trained model.

---

## Features

- Upload resumes in `.pdf` or `.txt` format.
- Automated screening with **instant predictions**.
- Uses **TF-IDF** and **KNN** for accurate classification.
- Clean and intuitive web interface built with **Streamlit**.

---

## Getting Started

Streamlit Web App
Run the Streamlit web app using the following command:

```bash
python -m streamlit run resume_classifier_app.py

```
Access the app in your web browser by following the link provided in the terminal.

Upload a resume (in .txt or .pdf format) to the app.

The app will process the resume and predict the associated category or job role.
