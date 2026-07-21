import pandas as pd

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Amazon_Reviews.csv", encoding="latin1")

print("=" * 60)
print("Original Shape:", df.shape)
print("=" * 60)

# Remove Duplicates
df = df.drop_duplicates()

# Remove Spaces
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Convert Rating
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")

# Remove Missing Values
df = df.dropna(subset=["Review", "Rate"])

# Remove Empty Reviews
df = df[df["Review"].str.strip() != ""]

# Reset Index
df.reset_index(drop=True, inplace=True)

print("\nFinal Shape:", df.shape)

print(df["Rate"].value_counts().sort_index())

df.to_csv("cleaned_reviews.csv", index=False)

print("✅ cleaned_reviews.csv saved")