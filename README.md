# Twitter Sentiment Analysis

## What It Does
A Natural Language Processing project that performs **entity-level sentiment analysis** on multilingual tweets.  
There were two Kaggle datasets, only for training, and one for validation. The instructions were to predict whether a tweet expresses a **positive**, **negative**, or **neutral** sentiment toward a given entity. 

This was my first major project working with a Kaggle set. Normalizing and cleaning data, then using NLP techniques to analyze it. 

Dataset: [Twitter Entity Sentiment Analysis (Kaggle)](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)

---

## Tech Stack
- Python
- pandas, numpy (data handling)
- scikit-learn (TF-IDF + Logistic Regression)
- NLTK (text cleaning)
- matplotlib, seaborn (visualization)
- pytest (testing)

---

## How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/<raymondc-dev>/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
Install dependencies

```bash
Copy code
pip install -r requirements.txt
Run scripts

```bash
Copy code
python src/clean_data.py
python src/train_model.py
python src/evaluate_model.py
python src/visualize_results.py
Run tests

```bash
Copy code
pytest

---------------------------------------

Social media platforms like Twitter hold massive volumes of unstructured text data.
Being able to quickly identify public sentiment around a company, product, or event helps:

Businesses track brand reputation

Investors gauge market perception

Researchers analyze sociopolitical trends

This project aims to demonstrate how Natural Language Processing (NLP) techniques can extract structured insights from noisy, real-world tweets.

Data Understanding & Cleaning

Dataset: Twitter Entity Sentiment Analysis (Kaggle)

Data Challenges

The dataset contained tweets in multiple languages.

Sentiment labels had mixed formats (e.g., Pos, positive, irrelevant).

Tweets included URLs, mentions, hashtags, and emojis.

Some tweets were irrelevant or neutral about the entity.

Cleaning Process

Converted all text to lowercase.

Removed URLs, mentions (@user), hashtags (#topic), and special characters.

Normalized sentiment labels into three main categories: Positive, Negative, Neutral.

Dropped null or empty tweets.

Example cleaning function:

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-záéíóúüñçàèìòùâêîôûäëïöûß ]", " ", text)
    return text.strip()

🧠 Modeling Approach

Feature extraction: TF-IDF (bi-gram range, 5,000–20,000 features)

Model: Logistic Regression (chosen for interpretability and speed)

Evaluation metric: Accuracy and F1-score on validation set

You can retrain the model using:

python src/train_model.py


The trained pipeline is stored in models/sentiment_model.pkl (excluded from repo via .gitignore).

📊 Data Visualization & Insights
Sentiment Distribution (Training)

Neutral tweets dominate, showing how often users discuss entities without strong polarity.

Negative sentiment tends to be underrepresented — common in real-world datasets.

Model Confusion Matrix (Validation)

The model performs best on clear positive/negative cases but occasionally misclassifies neutral tweets.

Sarcasm and multilingual nuances remain challenging areas.

⚙️ How to Run
git clone https://github.com/<username>/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -r requirements.txt
python src/clean_data.py
python src/train_model.py
python src/evaluate_model.py
python src/visualize_results.py
pytest
