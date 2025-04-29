import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
# Load the dataset
df = pd.read_csv("C:\\Users\\majok\\Downloads\\Final_project\\ResumeDataSet.csv")

# Step 1: Remove exact duplicates and duplicates in 'Resume' column
df = df.drop_duplicates()
df = df.drop_duplicates(subset='Resume')

# Step 2: Text Normalization - Clean resume text
def clean_resume(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text.strip()

df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Step 3: Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['cleaned_resume'] = df['cleaned_resume'].apply(remove_stopwords)

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y
)

# Step 6: Train Logistic Regression Model
log_model = LogisticRegression(max_iter=100)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Step 7: Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=150)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Step 8: Model Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=1))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Step 9: Evaluate both models
evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)

# Step 10: Save the trained models and vectorizer for later use
with open('log_model.pkl', 'wb') as f:
    pickle.dump(log_model, f)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
