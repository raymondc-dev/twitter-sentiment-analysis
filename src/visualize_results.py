import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(path):
    df = pd.read_csv(path)
    df.columns = ["id", "entity", "sentiment", "text"]

    plt.figure(figsize=(6,4))
    sns.countplot(x="sentiment", data=df, palette="pastel")
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("screenshots/class_distribution.png")
    print("Saved to screenshots/class_distribution.png")

if __name__ == "__main__":
    plot_distribution("data/twitter_training.csv")
