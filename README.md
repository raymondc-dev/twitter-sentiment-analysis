# Twitter Sentiment Analysis

## What It Does
A Natural Language Processing project that performs **entity-level sentiment analysis** on multilingual tweets.  
It predicts whether a tweet expresses a **positive**, **negative**, or **neutral** sentiment toward a given entity.

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
