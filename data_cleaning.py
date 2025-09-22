import pandas as pd
import spacy
from nltk.corpus import stopwords
from .utils import vprint

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Tokenize, lemmatize, remove stopwords & short words"""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stop_words and len(token) > 2
    ]
    return tokens

def clean_text(df, text_column="review_text", verbose=False):
    vprint("🔹 Cleaning text data...", verbose)
    df = df.copy()
    df["tokens"] = df[text_column].apply(preprocess)
    vprint("✅ Text cleaning done.", verbose)
    return df
