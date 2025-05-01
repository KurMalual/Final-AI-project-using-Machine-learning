# Resume Screening App

A simple web application that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to automatically screen and categorize resumes. Users can upload `.txt` or `.pdf` files, and the system predicts the most suitable **job role or category** based on the content.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Overview

This project streamlines the recruitment process by automating the **resume screening** phase. The app processes and classifies resumes using **NLP** techniques and a **multi-class machine learning classifier**, enabling faster and more consistent candidate filtering.

### System Components

#### Training Notebook – `resume_screening.ipynb`
- Preprocesses the data (cleaning, label encoding, stopword removal, lemmatization).
- Extracts features using **TF-IDF**.
- Trains a **K-Nearest Neighbors** model with `OneVsRestClassifier`.
- Saves the trained model and vectorizer.

#### 🌐 Web App – `app.py`
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