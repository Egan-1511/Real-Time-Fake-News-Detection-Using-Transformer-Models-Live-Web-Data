import pandas as pd
import re

# Load datasets
real_df = pd.read_csv('archive (2)\True.csv')
fake_df = pd.read_csv('archive (2)\Fake.csv')

# Add labels
real_df['label'] = 0
fake_df['label'] = 1

# Concatenate and shuffle
df = pd.concat([real_df, fake_df], ignore_index=True)
df = df[['title', 'label']].dropna()
df = df.sample(frac=1).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

# Apply cleaning
df['title'] = df['title'].apply(clean_text)

# Save cleaned dataset
df.to_csv('archive (2)/cleaned_fake_news.csv', index=False)

print("✅ Dataset cleaned and saved as 'archive (2)/cleaned_fake_news.csv'")
