# Sentiment Analyzer

A modular, end-to-end NLP pipeline for sentiment classification — built with text preprocessing, tokenization, TF-IDF feature extraction, and a Multinomial Naive Bayes classifier.

```
Raw Text
   ↓
Text Preprocessing
   ↓
Tokenization & Cleaning
   ↓
Feature Extraction (TF-IDF)
   ↓
Multinomial Naive Bayes Classifier
   ↓
Sentiment Prediction
   ↓
Performance Evaluation
```

## Overview

This project classifies text (reviews, comments, tweets, etc.) into sentiment categories. It's built around clean, single-responsibility classes rather than one long script, so each stage of the pipeline can be tested, reused, or swapped out independently.

## Features

- **Robust text cleaning** — strips HTML tags, URLs, bracketed content, and punctuation
- **Contraction expansion** — handles standard contractions (`don't` → `do not`) and broken/mojibake encodings often found in scraped data
- **Garbage token filtering** — removes non-vowel junk and alphanumeric noise
- **Optional spelling correction** via TextBlob
- **Tokenization & stopword removal** using NLTK
- **TF-IDF + Multinomial Naive Bayes** classifier with stratified k-fold cross-validation
- **Model persistence** — save and reload trained pipelines with `joblib`
- **Evaluation suite** — accuracy, classification report, and confusion matrix

## Project Structure

```
Sentiment_Analyzer/
├── sentiment_pipeline.py   # Core pipeline (all classes)
└── README.md
```

| Class | Responsibility |
|---|---|
| `TextPreprocessor` | Cleans raw text: lowercasing, HTML/URL removal, contraction expansion, garbage filtering |
| `Tokenizer` | Tokenizes text and removes stopwords |
| `SentimentModel` | TF-IDF + Multinomial Naive Bayes pipeline (train, predict, save, load) |
| `ModelEvaluator` | Computes accuracy, classification report, and confusion matrix |
| `SentimentAnalyzer` | Orchestrates the full pipeline end-to-end |

## Installation

```bash
git clone https://github.com/anirban1800d/Sentiment_Analyzer.git
cd Sentiment_Analyzer
pip install scikit-learn nltk textblob joblib pandas
```

NLTK resources (`punkt_tab`, `stopwords`) are downloaded automatically on first run.

## Usage

### Training

```python
from sentiment_pipeline import SentimentAnalyzer
import pandas as pd

df = pd.read_csv("your_data.csv")  # must contain text + label columns

analyzer = SentimentAnalyzer()
analyzer.train(df['Comments'], df['target'])
analyzer.save("multinomial_nb_pipeline.pkl")
```

### Inference

```python
analyzer = SentimentAnalyzer()
analyzer.load("multinomial_nb_pipeline.pkl")

preds, probs = analyzer.predict(["I really loved this product!"])
print(preds)   # e.g. ['pos']
print(probs)   # class probabilities
```

### Evaluation

```python
results = analyzer.evaluate(df_test['Comments'], df_test['target'])
```

## Tech Stack

- Python 3
- scikit-learn
- NLTK
- TextBlob
- joblib

## Roadmap

- [ ] Swap TF-IDF for word embeddings (Word2Vec / GloVe)
- [ ] Add a deep learning baseline (LSTM / transformer-based)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Streamlit demo app for live predictions

## License

This project is open source and available under the [MIT License](LICENSE).
