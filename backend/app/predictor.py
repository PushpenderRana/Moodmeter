import joblib
from pathlib import Path
from app.complaint_classifier import classify_complaint
from app.urgency_detector import detect_urgency
from app.routing_agent import route_review

from preprocessing.text_processing import preprocess

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

print("Loading predictor...")

print("Loading model...")
model = joblib.load(MODEL_DIR / "sentiment_model.pkl")

print("Loading vectorizer...")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

print("Loading encoder...")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

print("Predictor loaded.")
def predict_sentiment(review):

    cleaned = preprocess(review)
    

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)
    print("Prediction:", prediction)
    print("Probabilities:", model.predict_proba(vector))
    print("Classes:", encoder.classes_)

    sentiment = encoder.inverse_transform(prediction)[0]

    category = classify_complaint(review)

    urgency = detect_urgency(review)

    routing = route_review(
        category,
        urgency["urgent"]
    )

    return {

        "sentiment": sentiment,

        "category": category,

        "urgent": urgency["urgent"],

        "matched_keywords": urgency["matched_keywords"],

        "assigned_team": routing["assigned_team"],

        "priority": routing["priority"]

    }

