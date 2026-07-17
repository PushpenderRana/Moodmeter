import pandas as pd

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("Amazon_Reviews.csv")

print("=" * 50)
print("Original Shape:", df.shape)
print("=" * 50)

# ==========================
# Remove Duplicate Rows
# ==========================
duplicates = df.duplicated().sum()
print(f"Duplicate Rows: {duplicates}")

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# ==========================
# Remove Leading/Trailing Spaces
# ==========================
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# ==========================
# Missing Values
# ==========================
print("\nMissing Values")
print(df.isnull().sum())

print("\nMissing Percentage")
print((df.isnull().sum() / len(df)) * 100)

# ==========================
# Remove Rows with Missing Review Text or Rating
# ==========================
df = df.dropna(subset=["Review Text", "Rating"])

print("\nShape after removing missing Review Text & Rating:")
print(df.shape)

# ==========================
# Remove Empty Reviews
# ==========================
df = df[df["Review Text"].str.strip() != ""]

print("Shape after removing empty reviews:")
print(df.shape)

# ==========================
# Reset Index
# ==========================
df = df.reset_index(drop=True)

# ==========================
# Dataset Information
# ==========================
print("\nFinal Shape:", df.shape)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

print("\nFirst 5 Rows")
print(df.head())

# ==========================
# Save Clean Dataset
# ==========================
df.to_csv("cleaned_reviews.csv", index=False)

print("\n✅ Clean dataset saved as 'cleaned_reviews.csv'")