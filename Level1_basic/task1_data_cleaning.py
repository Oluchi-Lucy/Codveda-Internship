import pandas as pd

df = pd.read_csv("data/sentiment_dataset.csv")
print("Before cleaning:", df.shape)

# Remove leftover index columns
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
print("After removing index columns:", df.shape)

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Standardize text formatting
text_columns = ["Sentiment", "Platform", "Country", "User", "Hashtags", "Text"]
for col in text_columns:
    df[col] = df[col].str.strip()
print("\nTask 1 cleaning complete. Final shape:", df.shape)
# Check Timestamp format
print("\nTimestamp before conversion (sample):")
print(df["Timestamp"].head())

# Convert to proper datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
print("\nTimestamp after conversion (sample):")
print(df["Timestamp"].head())
print("\nData type of Timestamp:", df["Timestamp"].dtype)