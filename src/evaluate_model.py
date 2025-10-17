import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df.columns = ["id", "entity", "sentiment", "text"]

    preds = model.predict(df["text"])
    print(classification_report(df["sentiment"], preds))

    cm = confusion_matrix(df["sentiment"], preds, labels=["Positive","Negative","Neutral"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive","Negative","Neutral"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Twitter Sentiment Model")
    plt.savefig("screenshots/confusion_matrix.png")
    print("📊 Saved to screenshots/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model("models/sentiment_model.pkl", "data/twitter_validation.csv")
