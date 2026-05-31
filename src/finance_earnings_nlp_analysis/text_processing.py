import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .config import DEFAULT_STOP_EXTRA

STOPWORDS = set(ENGLISH_STOP_WORDS) | DEFAULT_STOP_EXTRA


def split_sentences(text: str):
    normalized = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', normalized) if len(s.strip()) > 25]


def tokenize(text: str):
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    return [t.strip("'-") for t in toks if len(t) > 2 and t not in STOPWORDS]
