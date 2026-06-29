"""
Sentiment Analysis Pipeline
============================

Pipeline stages (matches the architecture diagram):

    Raw Text
        ↓
    Text Preprocessing
        ↓
    Tokenization & Cleaning
        ↓
    Feature Extraction (TF-IDF / Count Vectorizer)
        ↓
    Multinomial Naive Bayes Classifier
        ↓
    Sentiment Prediction
        ↓
    Performance Evaluation

Each stage is implemented as its own class so the pipeline is modular,
testable, and easy to extend (swap TF-IDF for embeddings, swap NB for
Logistic Regression, etc.) without touching unrelated code.
"""

import re
import string

import joblib
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# One-time NLTK resource setup
# ---------------------------------------------------------------------------
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ===========================================================================
# STAGE 1: TEXT PREPROCESSING
# ===========================================================================
class TextPreprocessor:
    """
    Cleans raw text before tokenization:
        - lowercases
        - strips HTML tags, URLs, bracketed content
        - removes non-alphabetic characters
        - expands contractions (he's -> he is)
        - removes junk / garbage tokens
        - optional spelling correction
    """

    CONTRACTIONS = {
        r"\bhe's\b": "he is",
        r"\bthere's\b": "there is",
        r"\bwe're\b": "we are",
        r"\bthat's\b": "that is",
        r"\bwon't\b": "will not",
        r"\bcan't\b": "cannot",
        r"\bain't\b": "am not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bhaven't\b": "have not",
        r"\bhasn't\b": "has not",
        r"\bshouldn't\b": "should not",
        r"\bwouldn't\b": "would not",
        r"\bcouldn't\b": "could not",
        r"\bi'm\b": "i am",
        r"\bi've\b": "i have",
        r"\bi'd\b": "i would",
        r"\bi'll\b": "i will",
        r"\byou're\b": "you are",
        r"\byou've\b": "you have",
        r"\byou'd\b": "you would",
        r"\byou'll\b": "you will",
        r"\bthey're\b": "they are",
        r"\bthey've\b": "they have",
        r"\bthey'd\b": "they would",
        r"\bthey'll\b": "they will",
        r"\bit's\b": "it is",
        r"\bit'll\b": "it will",
        r"\bwe've\b": "we have",
        r"\bwe'll\b": "we will",
        r"\bwe'd\b": "we would",
        r"\blet's\b": "let us",
        r"\by'all\b": "you all",
        r"\bwho's\b": "who is",
        r"\bwhat's\b": "what is",
        r"\bwhere's\b": "where is",
        r"\bhere's\b": "here is",
        r"\bwould've\b": "would have",
        r"\bshould've\b": "should have",
        r"\bcould've\b": "could have",
        # broken / mojibake encodings (common in scraped tweet data)
        r"don\x89\u00db\u00aat|don\u00e5\u00aat": "do not",
        r"can\x89\u00db\u00aat": "cannot",
        r"it\x89\u00db\u00aas": "it is",
        r"i\x89\u00db\u00aam": "i am",
        r"i\x89\u00db\u00aave": "i have",
        r"you\x89\u00db\u00aare": "you are",
        r"you\x89\u00db\u00aave": "you have",
        r"you\x89\u00db\u00aall": "you will",
        r"doesn\x89\u00db\u00aat": "does not",
        r"wouldn\x89\u00db\u00aat": "would not",
        r"that\x89\u00db\u00aas": "that is",
        r"here\x89\u00db\u00aas": "here is",
    }

    def __init__(self, correct_spelling: bool = False):
        """
        Parameters
        ----------
        correct_spelling : bool
            Whether to run TextBlob spelling correction. This is slow on
            large datasets, so it defaults to off.
        """
        self.correct_spelling = correct_spelling

    # -- individual steps -------------------------------------------------

    def expand_contractions(self, text: str) -> str:
        for pattern, repl in self.CONTRACTIONS.items():
            text = re.sub(pattern, repl, text)
        return text

    def clean_text(self, text: str) -> str:
        text = str(text).lower()

        text = re.sub(r"<.*?>", " ", text)                     # HTML tags
        text = re.sub(r"(http\S+|www\S+|\w+\.\w+/\w+)", " ", text)  # URLs
        text = re.sub(r"\(.*?\)", " ", text)                    # bracketed text
        text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
        text = re.sub(r"\b[a-z]{1,2}\b", " ", text)             # drop 1-2 letter junk
        text = re.sub(r"\s+", " ", text).strip()                # normalize spaces

        return text

    def remove_garbage_words(self, text: str) -> str:
        """Drops tokens with no vowels (e.g. 'rmtrgf') and alphanumeric junk."""
        clean_words = []
        for w in text.split():
            if not re.search(r"[aeiou]", w):
                continue
            if re.search(r"\d", w):
                continue
            clean_words.append(w)
        return " ".join(clean_words)

    def remove_punctuation(self, text: str) -> str:
        for ch in string.punctuation:
            if ch in text:
                text = text.replace(ch, "")
        return text

    def correct_text_spelling(self, text: str) -> str:
        return str(TextBlob(text).correct())

    # -- orchestrator -------------------------------------------------------

    def transform(self, text: str) -> str:
        """Runs the full preprocessing chain on a single string."""
        text = self.expand_contractions(str(text).lower())
        text = self.clean_text(text)
        text = self.remove_garbage_words(text)
        text = self.remove_punctuation(text)
        if self.correct_spelling:
            text = self.correct_text_spelling(text)
        return text

    def transform_batch(self, texts):
        """Applies `transform` to an iterable (e.g. a pandas Series) of texts."""
        return [self.transform(t) for t in texts]


# ===========================================================================
# STAGE 2: TOKENIZATION & CLEANING
# ===========================================================================
class Tokenizer:
    """
    Tokenizes cleaned text and removes stopwords.
    """

    def __init__(self, language: str = "english"):
        self.language = language
        self._stopwords = set(stopwords.words(language))

    def tokenize(self, text: str) -> list:
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        return [w for w in tokens if w not in self._stopwords]

    def transform(self, text: str) -> list:
        """Tokenizes then strips stopwords, returning a list of tokens."""
        tokens = self.tokenize(text)
        return self.remove_stopwords(tokens)

    def transform_batch(self, texts):
        return [self.transform(t) for t in texts]


# ===========================================================================
# STAGES 3-5: FEATURE EXTRACTION + NAIVE BAYES + PREDICTION
# ===========================================================================
class SentimentModel:
    """
    Wraps TF-IDF feature extraction + Multinomial Naive Bayes into a single
    trainable / saveable sklearn Pipeline.
    """

    def __init__(
        self,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        alpha: float = 0.0010,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                sublinear_tf=True,
                norm="l2",
            )),
            ("nb", MultinomialNB(alpha=alpha)),
        ])

    def cross_validate(self, X, y, n_splits: int = 5, scoring: str = "accuracy"):
        """Runs stratified K-fold CV and returns the array of fold scores."""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        scores = cross_val_score(self.pipeline, X, y, cv=skf, scoring=scoring)
        print(f"Mean CV {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores

    def fit(self, X, y):
        """Trains the pipeline on the full dataset."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, path: str = "multinomial_nb_pipeline.pkl"):
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = "multinomial_nb_pipeline.pkl"):
        """Loads a previously saved pipeline into a new SentimentModel wrapper."""
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance


# ===========================================================================
# STAGE 6: PERFORMANCE EVALUATION
# ===========================================================================
class ModelEvaluator:
    """Computes and prints standard classification metrics."""

    @staticmethod
    def evaluate(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        return {
            "accuracy": acc,
            "report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }


# ===========================================================================
# ORCHESTRATOR: ties every stage together end-to-end
# ===========================================================================
class SentimentAnalyzer:
    """
    End-to-end orchestrator that mirrors the diagram:

        Raw Text -> Preprocessing -> Tokenization & Cleaning ->
        Feature Extraction -> Naive Bayes -> Prediction -> Evaluation

    Note: TF-IDF works directly on raw-ish strings, so tokenization here is
    primarily used for stopword-cleaned text inspection / alternate
    vectorizers. The model itself trains on the *preprocessed string*
    (not the token list), matching standard TfidfVectorizer usage.
    """

    def __init__(self, correct_spelling: bool = False):
        self.preprocessor = TextPreprocessor(correct_spelling=correct_spelling)
        self.tokenizer = Tokenizer()
        self.model = SentimentModel()
        self.evaluator = ModelEvaluator()

    def prepare_text(self, raw_texts):
        """Raw Text -> Preprocessing -> cleaned strings ready for TF-IDF."""
        return self.preprocessor.transform_batch(raw_texts)

    def tokenize_text(self, cleaned_texts):
        """Cleaned strings -> Tokenization & stopword removal -> token lists."""
        return self.tokenizer.transform_batch(cleaned_texts)

    def train(self, raw_texts, labels, cross_validate: bool = True):
        cleaned = self.prepare_text(raw_texts)
        if cross_validate:
            self.model.cross_validate(cleaned, labels)
        self.model.fit(cleaned, labels)
        return self

    def predict(self, raw_texts):
        cleaned = self.prepare_text(raw_texts)
        return self.model.predict(cleaned), self.model.predict_proba(cleaned)

    def evaluate(self, raw_texts, true_labels):
        preds, _ = self.predict(raw_texts)
        return self.evaluator.evaluate(true_labels, preds)

    def save(self, path: str = "multinomial_nb_pipeline.pkl"):
        self.model.save(path)

    def load(self, path: str = "multinomial_nb_pipeline.pkl"):
        self.model = SentimentModel.load(path)
        return self


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================
if __name__ == "__main__":
    import pandas as pd

    # df_set = pd.read_csv("your_data.csv")  # must have 'Comments' and 'target'

    analyzer = SentimentAnalyzer()

    # ---- Training ----
    # analyzer.train(df_set['Comments'], df_set['target'])
    # analyzer.save("multinomial_nb_pipeline.pkl")

    # ---- Inference on new text ----
    # analyzer.load("multinomial_nb_pipeline.pkl")
    # preds, probs = analyzer.predict(["I really loved this product!"])
    # print(preds, probs)
    pass
