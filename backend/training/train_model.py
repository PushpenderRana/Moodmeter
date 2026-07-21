import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "preprocessing" / "preprocessed_reviews.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

print("Dataset Shape:", df.shape)

# Create Sentiment Labels

def rating_to_sentiment(rating):

    rating = float(rating)

    if rating <= 2:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    else:
        return "Positive"

df["sentiment"] = df["Rate"].apply(rating_to_sentiment)

print("\nSentiment Distribution")

print(df["sentiment"].value_counts())



from sklearn.utils import resample

negative = df[df["sentiment"] == "Negative"]
neutral = df[df["sentiment"] == "Neutral"]
positive = df[df["sentiment"] == "Positive"]

target = min(len(negative), len(neutral), len(positive))

negative = resample(
    negative,
    replace=False,
    n_samples=target,
    random_state=42
)

neutral = resample(
    neutral,
    replace=False,
    n_samples=target,
    random_state=42
)

positive = resample(
    positive,
    replace=False,
    n_samples=target,
    random_state=42
)

df = pd.concat([negative, neutral, positive])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df["sentiment"].value_counts())

# Remove Missing Values

df = df.dropna(subset=["Cleaned_Review"])

# Features & Labels

X = df["Cleaned_Review"]

encoder = LabelEncoder()

y = encoder.fit_transform(df["sentiment"])

print("\nClasses")

print(encoder.classes_)

# TF-IDF

vectorizer = TfidfVectorizer(

    max_features=15000,

    ngram_range=(1,2),

    min_df=5,

    max_df=0.95,

    sublinear_tf=True

)

X = vectorizer.fit_transform(X)

print("\nTF-IDF Shape:", X.shape)

print("Vocabulary Size:", len(vectorizer.vocabulary_))



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples:", X_train.shape[0])

print("Testing Samples :", X_test.shape[0])

# Logistic Regression

model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced"

)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed")



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")

print(round(accuracy,4))

print("\nClassification Report")

print(classification_report(y_test,y_pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test,y_pred))



MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(

    model,

    MODEL_DIR / "sentiment_model.pkl"

)

joblib.dump(

    vectorizer,

    MODEL_DIR / "tfidf_vectorizer.pkl"

)

joblib.dump(

    encoder,

    MODEL_DIR / "label_encoder.pkl"

)

