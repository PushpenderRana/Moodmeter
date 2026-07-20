import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


# Load Dataset

df = pd.read_csv("preprocessed_reviews.csv")

# Create Sentiment Labels

import re

def rating_to_sentiment(rating):

    match = re.search(r"\d+", str(rating))

    if match:
        rating = int(match.group())
    else:
        return None

    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

# Features and Target

df["Sentiment"] = df["Rating"].apply(rating_to_sentiment)

df = df.dropna(subset=["Sentiment"])

X = df["Cleaned_Review"]
y = encoder.fit_transform(df["Sentiment"])

# TF-IDF Vectorization

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

print("TF-IDF Shape:", X.shape)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


# Save Model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

print("\n✅ Model Saved Successfully!")