import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

def train_model(data_path):
    df = pd.read_csv(data_path)
    df.columns = ["id", "entity", "sentiment", "text"]
    
    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, "models/sentiment_model.pkl")
    print("Model saved to models/sentiment_model.pkl")

if __name__ == "__main__":
    train_model("data/twitter_training.csv")
