import re
import string
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# Download NLTK Resources
# ==========================================

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ==========================================
# Stopwords
# ==========================================

stop_words = set(stopwords.words("english"))

# Keep Negation Words
negations = {
    "not",
    "no",
    "nor",
    "don't",
    "didn't",
    "doesn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "can't",
    "couldn't",
    "shouldn't",
    "wouldn't",
    "never"
}

stop_words -= negations

# ==========================================
# Lemmatizer
# ==========================================

lemmatizer = WordNetLemmatizer()

# ==========================================
# Text Preprocessing Function
# ==========================================

def preprocess(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove Emojis / Non ASCII Characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove Numbers
    text = re.sub(r"\d+", " ", text)

    # Remove Punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove Extra Spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove Stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Remove Single Characters
    tokens = [word for word in tokens if len(word) > 1]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    df = pd.read_csv("cleaned_reviews.csv")

    print("=" * 60)
    print("Cleaning Reviews...")
    print("=" * 60)

    df["Cleaned_Review"] = df["Review"].apply(preprocess)

    print(df[["Review", "Cleaned_Review"]].head())

    df.to_csv("preprocessed_reviews.csv", index=False)

    print("\n✅ Preprocessing Completed")

    print("Saved as preprocessed_reviews.csv")