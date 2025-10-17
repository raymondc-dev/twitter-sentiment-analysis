import pandas as pd
import re
import string
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    return text

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = ["id", "entity", "sentiment", "text"]
    df["text"] = df["text"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_and_clean("data/twitter_training.csv")
    print(df.head())
