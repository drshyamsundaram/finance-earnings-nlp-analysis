import re
from collections import Counter
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import POSITIVE_WORDS, NEGATIVE_WORDS, HEDGE_WORDS, THEME_KEYWORDS
from .text_processing import tokenize


def term_frequency(tokens):
    return pd.DataFrame(Counter(tokens).most_common(50), columns=['term', 'count'])


def phrase_analysis(sentences):
    docs = [' '.join(sentences[i:i+6]) for i in range(0, len(sentences), 6) if sentences[i:i+6]]
    if len(docs) < 2:
        docs = sentences[:]
    vec = CountVectorizer(stop_words='english', ngram_range=(2, 3), min_df=1, token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\-]+\b')
    X = vec.fit_transform(docs)
    counts = X.sum(axis=0).A1
    phrases = vec.get_feature_names_out()
    rows = sorted([(phrases[i], int(counts[i])) for i in range(len(phrases)) if counts[i] > 1], key=lambda x: (-x[1], x[0]))[:50]
    return pd.DataFrame(rows if rows else [('no recurring phrase', 1)], columns=['phrase', 'count'])


def sentiment_analysis(sentences):
    rows = []
    for sentence in sentences:
        toks = tokenize(sentence)
        pos = sum(1 for t in toks if t in POSITIVE_WORDS)
        neg = sum(1 for t in toks if t in NEGATIVE_WORDS)
        rows.append({'sentence': sentence, 'positive_hits': pos, 'negative_hits': neg, 'score': pos - neg})
    return pd.DataFrame(rows)


def uncertainty_analysis(sentences):
    rows = []
    for sentence in sentences:
        toks = tokenize(sentence)
        hedge = sum(1 for t in toks if t in HEDGE_WORDS)
        rows.append({'sentence': sentence, 'hedge_hits': hedge, 'is_forward_looking': hedge > 0})
    return pd.DataFrame(rows)


def theme_analysis(tokens):
    token_counts = Counter(tokens)
    rows = [{'theme': theme, 'term_hits': sum(token_counts[k] for k in kws)} for theme, kws in THEME_KEYWORDS.items()]
    return pd.DataFrame(rows).sort_values('term_hits', ascending=False)


def topic_model_analysis(sentences):
    chunks = [' '.join(sentences[i:i+8]) for i in range(0, len(sentences), 8) if sentences[i:i+8]]
    if len(chunks) < 3:
        return pd.DataFrame([{'topic': 'Topic 1', 'terms': 'insufficient chunks'}])
    vec = CountVectorizer(stop_words='english', max_features=300)
    X = vec.fit_transform(chunks)
    lda = LatentDirichletAllocation(n_components=min(4, len(chunks)), random_state=42)
    lda.fit(X)
    names = vec.get_feature_names_out()
    rows = []
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[-8:][::-1]
        rows.append({'topic': f'Topic {i+1}', 'terms': ', '.join(names[j] for j in idx)})
    return pd.DataFrame(rows)


def kpi_extraction(sentences):
    pats = [r'\$\d+(?:\.\d+)? billion', r'\d+% (?:growth|increase|tailwind)', r'earnings per share .*?\$\d+(?:\.\d+)?']
    rows = []
    for sentence in sentences:
        if any(re.search(p, sentence.lower()) for p in pats) or any(tok in sentence.lower() for tok in ['revenue','margin','cash flow','backlog','capex','eps']):
            rows.append({'kpi_sentence': sentence})
    return pd.DataFrame(rows).drop_duplicates().head(50)
