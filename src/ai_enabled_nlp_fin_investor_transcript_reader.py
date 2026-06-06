# -*- coding: utf-8 -*-
"""
## AI Retrieval Augmented Generation (RAG) Pipeline

This notebook demonstrates the setup and execution of a basic Retrieval Augmented Generation (RAG) pipeline. The primary goal is to enable a Large Language Model (LLM) to generate informed answers by first retrieving relevant information from a custom knowledge base.

### Enhancing for AI RAG:

To enable Retrieval Augmented Generation (RAG), we need to set up a pipeline that can:
1.  **Load documents**: Ingest documents (PDFs, text files) from a specified folder.
2.  **Split documents**: Break down larger documents into smaller, manageable chunks.
3.  **Generate embeddings**: Convert these chunks into numerical vector representations.
4.  **Store embeddings**: Save the chunks and their embeddings in a vector database.
5.  **Retrieve relevant chunks**: Given a query, find the most relevant chunks from the database.
6.  **Generate response**: Use a Large Language Model (LLM) to generate an answer based on the retrieved chunks and the query.

### Sample Use Case:
Imagine you are an analyst trying to understand the latest sentiment and market reaction to a company's earnings. Your primary input data is an **Alphabet investor report (a comprehensive financial document)**. To get a holistic view, you want to augment this with real-time, unstructured data from external sources. The RAG documents in this pipeline would represent **recent articles from media or social media reports** pertaining to Alphabet, providing crucial context and varying perspectives that the official investor report might not cover.

The RAG pipeline involves several key steps:

1.  **Document Loading:** Ingesting various document types (e.g., PDF, TXT) from a specified local folder (`rag_data`).
2.  **Text Splitting:** Breaking down these documents into smaller, manageable `chunks` to fit within the context window limits of LLMs and to improve the granularity of retrieval.
3.  **Embedding Generation:** Converting each text `chunk` into a high-dimensional numerical vector (embedding) using a pre-trained model (`HuggingFaceEmbeddings`). These embeddings capture the semantic meaning of the text.
4.  **Vector Store Creation:** Storing these embeddings in an efficient vector database (`FAISS`) that allows for rapid similarity searches.
5.  **Information Retrieval:** Given a user `query`, performing a similarity search in the vector store to find the most semantically relevant document `chunks`.
6.  **LLM Integration & Generation:** Using a small, local LLM (`distilgpt2` or `TinyLlama`) to synthesize a coherent and factual answer based on the original `query` and the `context` provided by the retrieved document `chunks`.

!pip install pandas scikit-learn PyMuPDF langchain-community langchain-core langchain-text-splitters faiss-cpu sentence-transformers transformers langchain-huggingface
"""

import os
from pathlib import Path
import json
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import RAG components for type hinting, though the actual objects will be passed
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSCRIPT_PATH = REPO_ROOT / 'data' / 'Alphabet-2026_Q1_Earnings_Transcript.txt'
DEFAULT_FIN_NLP_CONFIG_FILE = REPO_ROOT / 'fin_nlp.json'
DEFAULT_AI_CONFIG_PATH = REPO_ROOT / 'ai.json'
DEFAULT_QUERY_FILE = REPO_ROOT / 'query.json'
DEFAULT_RAG_DATA_FOLDER = REPO_ROOT / 'rag_data'
DEFAULT_OUTPUT_BASE_DIR = REPO_ROOT / 'output'
DEFAULT_VECTOR_DB_PERSIST_DIR = REPO_ROOT / 'vecdb'
DEFAULT_EXPORT_DIR = REPO_ROOT / 'vecdb_exported'


def load_nlp_config(config_path: Path) -> dict:
    """Loads NLP configuration from a JSON file or returns defaults."""
    default_config = {
        'default_stop_words': list(set(ENGLISH_STOP_WORDS) | {
            'alphabet', 'google', 'q1', 'quarter', 'year', 'years', 'company', 'business',
            'thanks', 'thank', 'operator', 'welcome', 'everyone', 'today', 'good', 'afternoon'
        }),
        'positive_words': ['growth','strong','momentum','improved','profit','record','accelerated','opportunity','opportunities','innovation','efficient','leadership','resilient','outstanding','terrific'],
        'negative_words': ['risk','pressure','cost','loss','decline','headwind','uncertainty','regulatory','slowdown','constraint','constrained'],
        'hedge_words': ['expect','expects','expected','may','could','might','should','looking','ahead','plan','plans','planning','outlook','forecast','guidance','would'],
        'theme_keywords': {
            'AI & Cloud': ['ai','cloud','gemini','model','models','compute','infrastructure','tpu','gpu','agents'],
            'Advertising': ['ads','advertising','search','youtube','retail','commerce','monetization'],
            'Financials': ['revenue','margin','income','cash','capex','backlog','eps','dividend'],
            'Operations': ['efficiency','servers','data','centers','network','deployment'],
            'Risk & Outlook': ['risk','outlook','uncertainty','regulatory','pressure','demand']
        },
        'risk_tags': {
            'Demand': ['demand','slowdown','backlog','usage'],
            'Margins': ['margin','profit','cost','depreciation'],
            'CapEx': ['capex','infrastructure','servers','data centers','compute'],
            'Regulation': ['regulatory','legal'],
            'Competition': ['competitive','market'],
            'Product Momentum': ['gemini','search','youtube','cloud','subscriptions']
        },
        'entity_pattern': r'\b(?:Gemini|YouTube|Search|Cloud|TPU|TPUs|Wiz|Waymo|BigQuery|Workspace|Google Cloud|AI Mode|AI Overviews|Google One)\b'
    }

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        # Merge loaded config with defaults, ensuring defaults are used if keys are missing
        for key, default_value in default_config.items():
            if key not in loaded_config:
                loaded_config[key] = default_value
            elif isinstance(default_value, list) and isinstance(loaded_config[key], list):
                # Ensure no duplicates if merging lists, but preserve order/original if not list of strings
                loaded_config[key] = list(set(loaded_config[key] + default_value))
        return loaded_config
    else:
        # Save defaults if config didn't exist
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config

class FinanceEarningsNLPAnalysis:
    def __init__(self, earnings_file: str, output_dir: str = 'output', config_file: str = 'fin_nlp.json',
                 rag_db: FAISS = None, rag_generator: pipeline = None,
                 llm_generation_params: dict = None, similarity_search_k: int = 5):
        earnings_path = Path(earnings_file)
        self.earnings_file = earnings_path if earnings_path.is_absolute() else REPO_ROOT / earnings_path

        output_path = Path(output_dir)
        self.output_dir = output_path if output_path.is_absolute() else REPO_ROOT / output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_path = Path(config_file)
        self.config_path = config_path if config_path.is_absolute() else REPO_ROOT / config_path

        self.text = ''
        self.sentences = []
        self.tokens = []
        self.sections = []
        self.speaker_blocks = []

        # RAG components
        self.rag_db = rag_db
        self.rag_generator = rag_generator
        self.llm_generation_params = llm_generation_params if llm_generation_params is not None else {}
        self.similarity_search_k = similarity_search_k

        self._load_config()

    def _load_config(self):
        """Loads configuration from a JSON file or sets defaults."""
        config = load_nlp_config(self.config_path)
        self.default_stop = set(config.get('default_stop_words', []))
        self.positive_words = set(config.get('positive_words', []))
        self.negative_words = set(config.get('negative_words', []))
        self.hedge_words = set(config.get('hedge_words', []))
        self.theme_keywords = config.get('theme_keywords', {})
        self.risk_tags = config.get('risk_tags', {})
        self.entity_pattern = re.compile(config.get('entity_pattern', ''))

    def _save_config(self):
        """Saves the current configuration to a JSON file."""
        config = {
            'default_stop_words': list(self.default_stop),
            'positive_words': list(self.positive_words),
            'negative_words': list(self.negative_words),
            'hedge_words': list(self.hedge_words),
            'theme_keywords': self.theme_keywords,
            'risk_tags': self.risk_tags,
            'entity_pattern': self.entity_pattern.pattern
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

    def load_text(self) -> str:
        """Loads text from the earnings file (PDF or TXT)."""
        suffix = self.earnings_file.suffix.lower()
        if suffix == '.pdf':
            try:
                import fitz
                doc = fitz.open(str(self.earnings_file))
                self.text = '\n'.join(page.get_text('text') for page in doc)
            except Exception as exc:
                raise RuntimeError(f'Unable to parse PDF: {exc}')
        else:
            self.text = self.earnings_file.read_text(encoding='utf-8', errors='ignore')
        return self.text

    def sentence_split(self, text: str):
        """Splits text into sentences."""
        normalized = re.sub(r'\s+', ' ', text).strip()
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', normalized) if len(s.strip()) > 25]

    def tokenize(self, text: str):
        """Tokenizes text into words, removing stop words."""
        toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
        return [t.strip("'-'") for t in toks if len(t) > 2 and t not in self.default_stop]

    def prepare(self):
        """Prepares the text by loading, splitting, and tokenizing."""
        self.load_text()
        self.sentences = self.sentence_split(self.text)
        self.tokens = self.tokenize(self.text)
        self.sections = self.extract_sections()
        self.speaker_blocks = self.extract_speakers()

    def extract_sections(self):
        """Extracts sections like 'Prepared Remarks' and 'Q&A'."""
        lowered = self.text.lower()
        marker = lowered.find('your first question')
        if marker != -1:
            return [
                {'section': 'Prepared Remarks', 'text': self.text[:marker]},
                {'section': 'Q&A', 'text': self.text[marker:]}
            ]
        return [{'section': 'Full Transcript', 'text': self.text}]

    def extract_speakers(self):
        """Identifies speaker blocks and extracts their dialogue."""
        blocks, current = [], None
        for line in self.text.splitlines():
            t = line.strip()
            if not t:
                continue
            if ':' in t and len(t) < 140 and any(name in t for name in ['Pichai', 'Schindler', 'Ashkenazi', 'Friedland', 'Operator', 'Nowak']):
                current = {'speaker': t.split(':')[0].strip(), 'lines': []}
                blocks.append(current)
            elif current is not None:
                current['lines'].append(t)
        rows = []
        for item in blocks:
            joined = ' '.join(item['lines']).strip()
            if joined:
                rows.append({'speaker': item['speaker'], 'text': joined, 'word_count': len(self.tokenize(joined))})
        return rows

    def term_frequency_analysis(self):
        """Analyzes the frequency of terms."""
        df = pd.DataFrame(Counter(self.tokens).most_common(50), columns=['term','count'])
        df.to_csv(self.output_dir / 'term_frequency.csv', index=False)
        return df

    def phrase_analysis(self):
        """Identifies and counts recurring phrases."""
        docs = [' '.join(self.sentences[i:i+6]) for i in range(0, len(self.sentences), 6) if self.sentences[i:i+6]]
        if len(docs) < 2:
            docs = self.sentences[:]
        vec = CountVectorizer(stop_words='english', ngram_range=(2,3), min_df=1, token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\-]+\b')
        X = vec.fit_transform(docs)
        counts = X.sum(axis=0).A1
        phrases = vec.get_feature_names_out()
        data = sorted([(phrases[i], int(counts[i])) for i in range(len(phrases)) if counts[i] > 1], key=lambda x: (-x[1], x[0]))[:50]
        df = pd.DataFrame(data if data else [('no recurring phrase',1)], columns=['phrase','count'])
        df.to_csv(self.output_dir / 'phrase_analysis.csv', index=False)
        return df

    def sentiment_analysis(self):
        """Performs sentiment analysis based on positive/negative word hits."""
        rows = []
        for s in self.sentences:
            toks = self.tokenize(s)
            pos = sum(1 for t in toks if t in self.positive_words)
            neg = sum(1 for t in toks if t in self.negative_words)
            rows.append({'sentence': s, 'positive_hits': pos, 'negative_hits': neg, 'score': pos-neg})
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'sentiment_analysis.csv', index=False)
        return df

    def uncertainty_analysis(self):
        """Detects uncertainty and forward-looking statements."""
        rows = []
        for s in self.sentences:
            toks = self.tokenize(s)
            hedge = sum(1 for t in toks if t in self.hedge_words)
            rows.append({'sentence': s, 'hedge_hits': hedge, 'is_forward_looking': hedge > 0})
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'uncertainty_analysis.csv', index=False)
        return df

    def speaker_analysis(self):
        """Analyzes contributions and top terms per speaker."""
        rows = []
        for block in self.speaker_blocks:
            toks = self.tokenize(block['text'])
            rows.append({'speaker': block['speaker'], 'word_count': len(toks), 'top_terms': ', '.join(t for t,_ in Counter(toks).most_common(8))})
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'speaker_analysis.csv', index=False)
        return df

    def section_analysis(self):
        """Provides word and unique term counts per transcript section."""
        rows = []
        for sec in self.sections:
            toks = self.tokenize(sec['text'])
            rows.append({'section': sec['section'], 'word_count': len(toks), 'unique_terms': len(set(toks))})
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'section_analysis.csv', index=False)
        return df

    def theme_analysis(self):
        """Identifies and counts predefined themes."""
        token_counts = Counter(self.tokens)
        rows = [{'theme': theme, 'term_hits': sum(token_counts[k] for k in kws)} for theme, kws in self.theme_keywords.items()]
        df = pd.DataFrame(rows).sort_values('term_hits', ascending=False)
        df.to_csv(self.output_dir / 'theme_analysis.csv', index=False)
        return df

    def risk_opportunity_analysis(self):
        """Classifies sentences as risk or opportunity based on keywords."""
        rows = []
        for s in self.sentences:
            tags = [tag for tag, kws in self.risk_tags.items() if any(k in s.lower() for k in kws)]
            if tags:
                kind = 'Opportunity' if any(w in s.lower() for w in ['growth','opportunity','strong','momentum','record']) else 'Risk'
                rows.append({'sentence': s, 'tags': ', '.join(tags), 'classification': kind})
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'risk_opportunity_analysis.csv', index=False)
        return df

    def key_quote_analysis(self):
        """Extracts key quotes using TF-IDF scores."""
        vec = TfidfVectorizer(stop_words='english')
        X = vec.fit_transform(self.sentences)
        scores = X.sum(axis=1).A1
        df = pd.DataFrame({'sentence': self.sentences, 'score': scores}).sort_values('score', ascending=False).head(25)
        df.to_csv(self.output_dir / 'key_quote_analysis.csv', index=False)
        return df

    def entity_analysis(self):
        """Identifies and counts specified entities."""
        df = pd.DataFrame(Counter(self.entity_pattern.findall(self.text)).most_common(), columns=['entity','count'])
        df.to_csv(self.output_dir / 'entity_analysis.csv', index=False)
        return df

    def kpi_extraction(self):
        """Extracts sentences containing key performance indicators."""
        pats = [r'\$\d+(?:\.\d+)? billion', r'\d+% (?:growth|increase|tailwind)', r'earnings per share .*?\$\d+(?::\d+)?', r'operating margin .*?\d+(?::\d+)?%', r'capex .*?\$\d+(?::\d+)? billion']
        rows = []
        for s in self.sentences:
            if any(re.search(p, s.lower()) for p in pats) or any(tok in s.lower() for tok in ['revenue','margin','cash flow','backlog','capex','eps']):
                rows.append({'kpi_sentence': s})
        df = pd.DataFrame(rows).drop_duplicates().head(50)
        df.to_csv(self.output_dir / 'kpi_extraction.csv', index=False)
        return df

    def topic_model_analysis(self):
        """Performs Latent Dirichlet Allocation (LDA) for topic modeling."""
        chunks = [' '.join(self.sentences[i:i+8]) for i in range(0, len(self.sentences), 8) if self.sentences[i:i+8]]
        if len(chunks) < 3:
            df = pd.DataFrame([{'topic':'Topic 1','terms':'insufficient chunks'}])
        else:
            vec = CountVectorizer(stop_words='english', max_features=300)
            X = vec.fit_transform(chunks)
            lda = LatentDirichletAllocation(n_components=min(4, len(chunks)), random_state=42)
            lda.fit(X)
            names = vec.get_feature_names_out()
            rows = []
            for i, comp in enumerate(lda.components_):
                idx = comp.argsort()[-8:][::-1]
                rows.append({'topic': f'Topic {i+1}', 'terms': ', '.join(names[j] for j in idx)})
            df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'topic_model_analysis.csv', index=False)
        return df

    def semantic_search_index(self):
        """Creates an index for semantic search using tokenized sentences."""
        df = pd.DataFrame({'sentence': self.sentences, 'tokens': [' '.join(self.tokenize(s)) for s in self.sentences]})
        df.to_csv(self.output_dir / 'semantic_search_index.csv', index=False)
        return df

    def cooccurrence_analysis(self):
        """Analyzes co-occurring term pairs within sentences."""
        pairs = Counter()
        for s in self.sentences:
            toks = list(dict.fromkeys(self.tokenize(s)))[:12]
            for i in range(len(toks)):
                for j in range(i+1, len(toks)):
                    pairs[' | '.join(sorted([toks[i], toks[j]]))] += 1
        df = pd.DataFrame(pairs.most_common(50), columns=['pair','count'])
        df.to_csv(self.output_dir / 'cooccurrence_analysis.csv', index=False)
        return df

    def comparative_placeholder_analysis(self):
        """Placeholder for comparative analysis with prior quarter transcripts."""
        df = pd.DataFrame([{'comparison':'prior_quarter_comparison','status':'placeholder','note':'load prior quarter transcript to enable delta analysis'}])
        df.to_csv(self.output_dir / 'comparative_analysis.csv', index=False)
        return df

    def _load_queries(self, file_path: str) -> list:
        """Loads a list of queries from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        print(f"Loaded {len(queries)} queries from {file_path}")
        return queries

    def run_rag_query(self, query_text: str, query_id: int = 0):
        """Processes a single RAG query using the internal db and generator, saving results to output_dir."""
        if not self.rag_db or not self.rag_generator:
            print("RAG components (db or generator) not initialized for FinanceEarningsNLPAnalysis.")
            return

        print(f"\nProcessing RAG query {query_id}: '{query_text}'")
        retrieved_docs = self.rag_db.similarity_search(query_text, k=self.similarity_search_k)

        context = "\n".join([doc.page_content for doc in retrieved_docs])
        # Enhanced prompt for comparative queries
        prompt = f"Analyze the following context to identify differences and similarities related to the question. Clearly state the points of comparison and contrast. Context: {context}\n\nQuestion: {query_text}\n\nAnswer:"

        print(f"Generating response...")
        try:
            # Pass generation parameters from config to the generator
            response = self.rag_generator(prompt, **self.llm_generation_params)
            generated_text = response[0]['generated_text']
        except Exception as e:
            print(f"Error during LLM generation for query '{query_text}': {e}")
            generated_text = f"Error: LLM generation failed. Details: {e}"

        print("--- Generated Answer ---")
        print(generated_text)

        # Ensure output directory exists (already done in __init__)

        # Save retrieved documents
        retrieved_docs_path = self.output_dir / f'rag_retrieved_documents_{query_id}.txt'
        with open(retrieved_docs_path, 'w', encoding='utf-8') as f:
            f.write(context)
        print(f"Retrieved documents saved to: {retrieved_docs_path}")

        # Save generated answer
        generated_answer_path = self.output_dir / f'rag_generated_answer_{query_id}.txt'
        with open(generated_answer_path, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"Generated answer saved to: {generated_answer_path}")

    def export_vector_store(self, persist_dir: str = './vecdb_exported'):
        """Exports the FAISS vector store to a specified directory."""
        if self.rag_db:
            persist_path = Path(persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.rag_db.save_local(persist_dir)
            print(f"Vector store exported to '{persist_dir}'.")
        else:
            print("No RAG vector store initialized to export.")

    def build_report_payload(self, analyses):
        """Aggregates analysis results into a JSON-compatible payload."""
        sent = analyses['sentiment']
        return {
            'document': self.earnings_file.name,
            'sentence_count': len(self.sentences),
            'token_count': len(self.tokens),
            'unique_terms': len(set(self.tokens)),
            'top_terms': analyses['terms'].head(12).to_dict(orient='records'),
            'top_phrases': analyses['phrases'].head(12).to_dict(orient='records'),
            'sentiment_summary': {'positive': int((sent['score'] > 0).sum()), 'neutral': int((sent['score'] == 0).sum()), 'negative': int((sent['score'] < 0).sum())},
            'themes': analyses['themes'].to_dict(orient='records'),
            'speakers': analyses['speakers'].to_dict(orient='records'),
            'sections': analyses['sections'].to_dict(orient='records'),
            'quotes': analyses['quotes'].head(10).to_dict(orient='records'),
            'entities': analyses['entities'].to_dict(orient='records'),
            'kpis': analyses['kpis'].head(15).to_dict(orient='records'),
            'risk_opportunities': analyses['risk_opportunities'].head(20).to_dict(orient='records'),
            'topics': analyses['topics'].to_dict(orient='records'),
            'cooccurrence': analyses['cooccurrence'].head(20).to_dict(orient='records'),
            'uncertainty': analyses['uncertainty'].head(20).to_dict(orient='records'),
            'semantic_index': analyses['semantic'].head(120).to_dict(orient='records')
        }

    def build_html_report(self, report_data: dict) -> str:
        """Generates a comprehensive HTML report from analysis data."""
        payload = json.dumps(report_data)
        return """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Finance Earnings NLP analysis</title><link href=\"https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&f[]=zodiak@400,700&display=swap\" rel=\"stylesheet\"><script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script><style>:root{--bg:#f7f6f2;--surface:#fff;--surface2:#f2efe8;--text:#1f1d18;--muted:#6d685f;--border:#d7d2c7;--primary:#01696f;--good:#437a22;--warn:#d19900;--bad:#a13544}[data-theme=dark]{--bg:#171614;--surface:#1d1c1a;--surface2:#23211e;--text:#ece8e0;--muted:#a39d93;--border:#38342f;--primary:#58a1ab;--good:#7fbc5b;--warn:#efc35e;--bad:#e37b93}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Satoshi,sans-serif;line-height:1.55}.wrap{max-width:1280px;margin:0 auto;padding:20px}.topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;position:sticky;top:0;background:color-mix(in srgb,var(--bg) 90%,transparent);backdrop-filter:blur(10px);padding:16px 20px;border-bottom:1px solid var(--border)}h1,h2,h3{font-family:Zodiak,serif;margin:0}p{color:var(--muted)}.hero,.grid,.stats{display:grid;gap:18px}.hero{grid-template-columns:1.5fr 1fr;margin-top:18px}.grid{grid-template-columns:1fr 1fr;margin-top:18px}.stats{grid-template-columns:repeat(4,1fr);margin-top:18px}.card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:24px}.stat{background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:14px}.stat strong{display:block;font-size:1.4rem;color:var(--text)}.tag{display:inline-block;background:var(--surface2);color:var(--primary);padding:5px 9px;border-radius:999px;margin:4px 6px 0 0;font-size:12px;font-weight:700}.note{background:var(--surface2);border-left:3px solid var(--primary);padding:14px 16px;border-radius:12px;margin-top:10px}button,input{border:1px solid var(--border);background:var(--surface);color:var(--text);padding:10px 14px;border-radius:12px}table{width:100%;border-collapse:collapse}th,td{padding:10px 8px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}th{color:var(--muted);font-size:.88rem}canvas{max-height:320px}.quotes{display:grid;gap:10px}.quote{background:var(--surface2);border-left:3px solid var(--primary);padding:12px 14px;border-radius:12px}.search-box{width:100%;margin-top:12px}@media (max-width:960px){.hero,.grid,.stats{grid-template-columns:1fr}.topbar{position:static}}</style></head><body><div class=\"topbar\"><div><h1>Finance Earnings NLP analysis</h1><p>Interactive analyst workspace for transcript-driven NLP exploration.</p></div><button id=\"themeToggle\">Toggle theme</button></div><div class=\"wrap\"><section class=\"hero\"><div class=\"card\"><div class=\"tag\">Transcript</div><h2 style=\"margin-top:12px;font-size:2.2rem\">Comprehensive earnings NLP analysis</h2><p>Includes speaker, section, KPI, sentiment, uncertainty, topics, entities, co-occurrence, semantic index, and risk-opportunity layers.</p><div class=\"stats\" id=\"stats\"></div></div><div class=\"card\"><h3>Analysis layers</h3><div id=\"themeTags\"></div><div class=\"note\"><strong>Modular design</strong><br>Each NLP layer is generated independently in Python and exported as its own CSV for analyst workflows.</div></div></section><section class=\"grid\"><div class=\"card\"><h2>Top terms</h2><canvas id=\"termsChart\"></canvas></div><div class=\"card\"><h2>Sentiment mix</h2><canvas id=\"sentimentChart\"></canvas></div></section><section class=\"grid\"><div class=\"card\"><h2>Theme hits</h2><table><thead><tr><th>Theme</th><th>Hits</th></tr></thead><tbody id=\"themeTable\"></tbody></table></div><div class=\"card\"><h2>Speaker analysis</h2><table><thead><tr><th>Speaker</th><th>Words</th><th>Top terms</th></tr></thead><tbody id=\"speakerTable\"></tbody></table></div></section><section class=\"grid\"><div class=\"card\"><h2>Section analysis</h2><table><thead><tr><th>Section</th><th>Words</th><th>Unique terms</th></tr></thead><tbody id=\"sectionTable\"></tbody></table></div><div class=\"card\"><h2>Entity analysis</h2><table><thead><tr><th>Entity</th><th>Count</th></tr></thead><tbody id=\"entityTable\"></tbody></table></div></section><section class=\"grid\"><div class=\"card\"><h2>Topic clusters</h2><div class=\"quotes\" id=\"topicList\"></div></div><div class=\"card\"><h2>Recurring phrases</h2><table><thead><tr><th>Phrase</th><th>Count</th></tr></thead><tbody id=\"phraseTable\"></tbody></table></div></section><section class=\"grid\"><div class=\"card\"><h2>KPI extraction</h2><div class=\"quotes\" id=\"kpiList\"></div></div><div class=\"card\"><h2>Risk and opportunity</h2><div class=\"quotes\" id=\"riskList\"></div></div></section><section class=\"grid\"><div class=\"card\"><h2>Key quotes</h2><div class=\"quotes\" id=\"quoteList\"></div></div><div class=\"card\"><h2>Co-occurrence pairs</h2><table><thead><tr><th>Pair</th><th>Count</th></tr></thead><tbody id=\"coTable\"></tbody></table></div></section><section class=\"grid\"><div class=\"card\"><h2>Semantic search index</h2><input id=\"semanticSearch\" class=\"search-box\" placeholder=\"Search transcript concepts like capex, TPUs, Gemini, margin\" /><div class=\"quotes\" id=\"semanticResults\"></div></div><div class=\"card\"><h2>Uncertainty and forward-looking sentences</h2><div class=\"quotes\" id=\"uncertaintyList\"></div></div></section></div><script>const report=__REPORT_DATA__;const statRows=[['Document',report.document],['Sentences',String(report.sentence_count)],['Tokens',String(report.token_count)],['Unique terms',String(report.unique_terms)],['Positive sentences',String(report.sentiment_summary.positive)],['Neutral sentences',String(report.sentiment_summary.neutral)],['Negative sentences',String(report.sentiment_summary.negative)],['Top terms',String(report.top_terms.length)]];statRows.forEach(function(item){const div=document.createElement('div');div.className='stat';div.innerHTML='<span>'+item[0]+'</span><strong>'+item[1]+'</strong>';document.getElementById('stats').appendChild(div)});report.themes.forEach(function(t){const tag=document.createElement('span');tag.className='tag';tag.textContent=t.theme;document.getElementById('themeTags').appendChild(tag)});document.getElementById('themeTable').innerHTML=report.themes.map(x=>'<tr><td>'+x.theme+'</td><td>'+x.term_hits+'</td></tr>').join('');document.getElementById('speakerTable').innerHTML=report.speakers.map(x=>'<tr><td>'+x.speaker+'</td><td>'+x.word_count+'</td><td>'+x.top_terms+'</td></tr>').join('');document.getElementById('sectionTable').innerHTML=report.sections.map(x=>'<tr><td>'+x.section+'</td><td>'+x.word_count+'</td><td>'+x.unique_terms+'</td></tr>').join('');document.getElementById('entityTable').innerHTML=report.entities.map(x=>'<tr><td>'+x.entity+'</td><td>'+x.count+'</td></tr>').join('');document.getElementById('phraseTable').innerHTML=report.top_phrases.map(x=>'<tr><td>'+x.phrase+'</td><td>'+x.count+'</td></tr>').join('');document.getElementById('coTable').innerHTML=report.cooccurrence.map(x=>'<tr><td>'+x.pair+'</td><td>'+x.count+'</td></tr>').join('');report.topics.forEach(x=>{const d=document.createElement('div');d.className='quote';d.innerHTML='<strong>'+x.topic+'</strong><br>'+x.terms;document.getElementById('topicList').appendChild(d)});report.kpis.forEach(x=>{const d=document.createElement('div');d.className='quote';d.innerHTML='<strong>KPI match</strong><br>'+x.kpi_sentence;document.getElementById('kpiList').appendChild(d)});report.risk_opportunities.forEach(x=>{const d=document.createElement('div');d.className='quote';d.innerHTML='<strong>'+x.classification+'</strong> · '+x.tags+'<br>'+x.sentence;document.getElementById('riskList').appendChild(d)});report.quotes.forEach(x=>{const d=document.createElement('div');d.className='quote';d.textContent=item.sentence;wrap.appendChild(d)})}renderSemanticResults('');document.getElementById('semanticSearch').addEventListener('input',function(e){renderSemanticResults(e.target.value)});let dark=window.matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.setAttribute('data-theme',dark?'dark':'light');document.getElementById('themeToggle').onclick=function(){dark=!dark;document.documentElement.setAttribute('data-theme',dark?'dark':'light')};</script></body></html>""".replace('__REPORT_DATA__', payload)

    def run(self):
        """Executes all NLP analyses and generates the HTML report."""
        self.prepare()
        analyses = {
            'terms': self.term_frequency_analysis(),
            'phrases': self.phrase_analysis(),
            'sentiment': self.sentiment_analysis(),
            'uncertainty': self.uncertainty_analysis(),
            'speakers': self.speaker_analysis(),
            'sections': self.section_analysis(),
            'themes': self.theme_analysis(),
            'risk_opportunities': self.risk_opportunity_analysis(),
            'quotes': self.key_quote_analysis(),
            'entities': self.entity_analysis(),
            'kpis': self.kpi_extraction(),
            'topics': self.topic_model_analysis(),
            'semantic': self.semantic_search_index(),
            'cooccurrence': self.cooccurrence_analysis(),
            'comparative': self.comparative_placeholder_analysis(),
        }
        payload = self.build_report_payload(analyses)
        html = self.build_html_report(payload)
        html_path = self.output_dir / 'finance-earnings-nlp-analysis.html'
        html_path.write_text(html, encoding='utf-8')
        manifest = pd.DataFrame(
            [
                {'artifact': 'finance-earnings-nlp-analysis.html', 'type': 'interactive_html'},
                {'artifact': 'term_frequency.csv', 'type': 'csv'},
                {'artifact': 'phrase_analysis.csv', 'type': 'csv'},
                {'artifact': 'sentiment_analysis.csv', 'type': 'csv'},
                {'artifact': 'uncertainty_analysis.csv', 'type': 'csv'},
                {'artifact': 'speaker_analysis.csv', 'type': 'csv'},
                {'artifact': 'section_analysis.csv', 'type': 'csv'},
                {'artifact': 'theme_analysis.csv', 'type': 'csv'},
                {'artifact': 'risk_opportunity_analysis.csv', 'type': 'csv'},
                {'artifact': 'key_quote_analysis.csv', 'type': 'csv'},
                {'artifact': 'entity_analysis.csv', 'type': 'csv'},
                {'artifact': 'kpi_extraction.csv', 'type': 'csv'},
                {'artifact': 'topic_model_analysis.csv', 'type': 'csv'},
                {'artifact': 'semantic_search_index.csv', 'type': 'csv'},
                {'artifact': 'cooccurrence_analysis.csv', 'type': 'csv'},
                {'artifact': 'comparative_analysis.csv', 'type': 'csv'},
            ]
        )
        manifest.to_csv(self.output_dir / 'analysis_manifest.csv', index=False)
        return {'html': str(html_path), 'manifest': str(self.output_dir / 'analysis_manifest.csv')}

def run_finance_earnings_nlp_analysis(pdf_path: str, output_dir: str = str(DEFAULT_OUTPUT_BASE_DIR), config_file: str = str(DEFAULT_FIN_NLP_CONFIG_FILE)):
    """Convenience function to run the NLP analysis from a PDF."""
    return FinanceEarningsNLPAnalysis(pdf_path, output_dir, config_file).run()

def main(earnings_file: str = str(DEFAULT_TRANSCRIPT_PATH), config_file: str = str(DEFAULT_FIN_NLP_CONFIG_FILE)):
    """Main function to execute the analysis with a default file."""
    print(run_finance_earnings_nlp_analysis(earnings_file, config_file=config_file))

import json
from pathlib import Path

ai_config_path = DEFAULT_AI_CONFIG_PATH

# Load the existing configuration
if ai_config_path.exists():
    with open(ai_config_path, 'r', encoding='utf-8') as f:
        ai_config = json.load(f)
else:
    # Default configuration if file does not exist
    ai_config = {
        "llm_model_name": "distilgpt2",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "rag_data_folder": str(DEFAULT_RAG_DATA_FOLDER),
        "vector_db_persist_dir": str(DEFAULT_VECTOR_DB_PERSIST_DIR),
        "queries_file": str(DEFAULT_QUERY_FILE),
        "output_base_dir": str(DEFAULT_OUTPUT_BASE_DIR),
        "use_persistent_vector_db": True,
        "export_rag_vector_db": True,
        "export_dir": str(DEFAULT_EXPORT_DIR)
    }

# Add or update AI model hyperparameters
ai_config["llm_generation_params"] = {
    "max_new_tokens": 100,
    "num_return_sequences": 1,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95
}
ai_config["similarity_search_k"] = 5

# Save the updated configuration
with open(ai_config_path, 'w', encoding='utf-8') as f:
    json.dump(ai_config, f, indent=4)

print(f"Updated AI model hyperparameters in {ai_config_path}")

"""Now that the `ai.json` file includes the new hyperparameters, I will update the `FinanceEarningsNLPAnalysis` class and the `run_full_analysis_and_rag_flow` function to load and utilize these parameters."""



"""### Integrating RAG and Query Loading into a Full Flow

To make the RAG pipeline dynamic and part of the overall analysis, we'll introduce:

1.  A `query.json` file to store multiple queries.
2.  A `load_queries` function to read these queries.
3.  Refactored RAG setup and LLM processing into reusable functions.
4.  An orchestration block to run the NLP analysis, set up RAG, load queries, and process each one using the LLM, saving the outputs.
"""

import json
from pathlib import Path

# Create a dummy query.json file for demonstration
query_data = [
    "What were Alphabet's Q1 2024 earnings?",
    "What are the key differences between Alphabet's and Microsoft's AI strategy?",
    "Summarize the market reaction to Alphabet's recent announcements.",
    "What are the main risks and opportunities mentioned for Alphabet?"
]

query_file_path = DEFAULT_QUERY_FILE
with open(query_file_path, 'w', encoding='utf-8') as f:
    json.dump(query_data, f, indent=4)

print(f"Dummy queries saved to {query_file_path}")


from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import os
from pathlib import Path
import json

def load_documents_from_rag_folder(folder_path: str):
    """Loads documents from a specified RAG folder, handling PDF and TXT files."""
    folder_path = Path(folder_path)
    if not folder_path.is_absolute():
        folder_path = REPO_ROOT / folder_path

    documents = []
    if not folder_path.exists():
        print(f"RAG folder does not exist: {folder_path}")
        return documents

    for filename in os.listdir(folder_path):
        file_path = folder_path / filename
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif filename.endswith('.txt'):
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents from '{folder_path}'.")
    return documents

def setup_rag_pipeline(rag_folder: str, llm_model_name: str, embedding_model_name: str,
                       persist_dir: str, use_persistent_vector_db: bool):
    """Sets up the RAG pipeline: loads docs, splits, embeds, creates/loads vector store, and loads LLM."""
    rag_folder = Path(rag_folder)
    if not rag_folder.is_absolute():
        rag_folder = REPO_ROOT / rag_folder

    persist_path = Path(persist_dir)
    if not persist_path.is_absolute():
        persist_path = REPO_ROOT / persist_path

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if use_persistent_vector_db and persist_path.exists() and any(persist_path.iterdir()):
        print(f"Loading FAISS vector store from '{persist_dir}'...")
        db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
    else:
        # If not using persistent DB or DB doesn't exist, create a new one
        print("Creating new FAISS vector store...")
        # 1. Document Loading
        loaded_docs = load_documents_from_rag_folder(rag_folder)

        # 2. Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(loaded_docs)
        print(f"Split into {len(chunks)} chunks.")

        # 3. Embedding Generation and Vector Store
        db = FAISS.from_documents(chunks, embeddings)
        print("Vector store created.")

        # Save the new vector store if persistence is enabled
        if use_persistent_vector_db:
            db.save_local(persist_dir)
            print(f"Vector store saved to '{persist_dir}'.")

    # 4. LLM Integration
    print(f"Loading LLM: {llm_model_name}...")
    generator = pipeline('text-generation', model=llm_model_name)
    print("LLM loaded.")

    return db, generator


def run_full_analysis_and_rag_flow(earnings_file: str = str(DEFAULT_TRANSCRIPT_PATH),
                                   nlp_config_file: str = str(DEFAULT_FIN_NLP_CONFIG_FILE),
                                   ai_config_file: str = str(DEFAULT_AI_CONFIG_PATH)):
    """Runs the full NLP analysis and RAG flow with multiple queries, loading RAG config from ai_config_file."""

    ai_config_path = Path(ai_config_file)
    if not ai_config_path.is_absolute():
        ai_config_path = REPO_ROOT / ai_config_path

    # Load AI RAG configuration
    with open(ai_config_path, 'r', encoding='utf-8') as f:
        ai_config = json.load(f)

    llm_model_name = ai_config.get("llm_model_name", "distilgpt2")
    embedding_model_name = ai_config.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    rag_data_folder = ai_config.get("rag_data_folder", str(DEFAULT_RAG_DATA_FOLDER))
    queries_file = ai_config.get("queries_file", str(DEFAULT_QUERY_FILE))
    output_base_dir = ai_config.get("output_base_dir", str(DEFAULT_OUTPUT_BASE_DIR))
    vector_db_persist_dir = ai_config.get("vector_db_persist_dir", str(DEFAULT_VECTOR_DB_PERSIST_DIR))
    use_persistent_vector_db = ai_config.get("use_persistent_vector_db", False)
    export_rag_vector_db = ai_config.get("export_rag_vector_db", False)
    export_dir = ai_config.get("export_dir", str(DEFAULT_EXPORT_DIR))
    llm_generation_params = ai_config.get("llm_generation_params", {})
    similarity_search_k = ai_config.get("similarity_search_k", 5)

    rag_data_folder = Path(rag_data_folder)
    if not rag_data_folder.is_absolute():
        rag_data_folder = REPO_ROOT / rag_data_folder

    queries_file = Path(queries_file)
    if not queries_file.is_absolute():
        queries_file = REPO_ROOT / queries_file

    output_base_path = Path(output_base_dir)
    if not output_base_path.is_absolute():
        output_base_path = REPO_ROOT / output_base_path
    output_base_path.mkdir(parents=True, exist_ok=True)

    vector_db_persist_dir = Path(vector_db_persist_dir)
    if not vector_db_persist_dir.is_absolute():
        vector_db_persist_dir = REPO_ROOT / vector_db_persist_dir

    export_dir = Path(export_dir)
    if not export_dir.is_absolute():
        export_dir = REPO_ROOT / export_dir

    print("\n--- Setting Up RAG Pipeline ---")
    db, generator = setup_rag_pipeline(rag_folder=rag_data_folder, # Changed from rag_data_folder to rag_folder
                                       llm_model_name=llm_model_name,
                                       embedding_model_name=embedding_model_name,
                                       persist_dir=vector_db_persist_dir,
                                       use_persistent_vector_db=use_persistent_vector_db)
    print("RAG Pipeline Setup Complete.")

    print("\n--- Running Financial NLP Analysis ---")
    # Instantiate FinanceEarningsNLPAnalysis with RAG components and hyperparameters
    nlp_analysis = FinanceEarningsNLPAnalysis(earnings_file, output_base_path, nlp_config_file,
                                              rag_db=db, rag_generator=generator,
                                              llm_generation_params=llm_generation_params,
                                              similarity_search_k=similarity_search_k)
    analysis_results = nlp_analysis.run() # Run the core NLP analysis
    print("Financial NLP Analysis Complete.")
    print(analysis_results)


    print("\n--- Loading Queries ---")
    queries = nlp_analysis._load_queries(queries_file)
    print(f"Loaded {len(queries)} queries for RAG.")

    print("\n--- Processing RAG Queries with integrated NLP Analysis object ---")
    for i, query_text in enumerate(queries):
        nlp_analysis.run_rag_query(query_text, query_id=i+1)
    print("All RAG Queries Processed.")

    # Export the vector store if requested
    if export_rag_vector_db:
        print("\n--- Exporting RAG Vector Store ---")
        nlp_analysis.export_vector_store(persist_dir=export_dir)
        print("RAG Vector Store Export Complete.")

# Call the main orchestration function only when executed as a script
if __name__ == '__main__':
    run_full_analysis_and_rag_flow()

"""
5. LLM Integration for Generation

Finally, we can integrate a small to medium-sized LLM from Hugging Face to generate a coherent answer using the retrieved documents. For demonstration, we'll use a `text-generation` pipeline with a small model. You might need to pick a model that fits your resource constraints (e.g., `distilgpt2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
**Note**: Running a full LLM for generation might require significant computational resources. For complex RAG setups, consider using services that abstract away LLM hosting or fine-tuning. This example uses a very small model for illustrative purposes.
"""

import json
from pathlib import Path
import re

config_path = DEFAULT_FIN_NLP_CONFIG_FILE

# Load existing config or create a default structure
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
else:
    config = {
        'default_stop_words': [],
        'positive_words': [],
        'negative_words': [],
        'hedge_words': [],
        'theme_keywords': {},
        'risk_tags': {},
        'entity_pattern': ''
    }

# Define new terms to add
new_entities = [
    'Anthropic', 'OpenAI', 'Perplexity', 'Microsoft', 'Amazon', 'Nvidia', 'Apple', 'Meta', 'Tesla', 'Google', 'Alphabet',
    'Claude', 'ChatGPT', 'Sora', 'Copilot', 'Azure AI', 'AWS AI', 'Llama', 'Bard', 'DALL-E', 'Midjourney', 'Stable Diffusion',
    'AGI', 'LLM', 'Generative AI', 'AI Ethics', 'AI Governance', 'AI Regulation', 'Machine Learning', 'Deep Learning', 'Neural Networks'
]

# Merge new entities into the existing entity_pattern
current_entity_pattern = config.get('entity_pattern', '').replace(r'\b(?:', '').replace(r')\b', '')
existing_entities = [e.strip() for e in current_entity_pattern.split('|') if e.strip()]

all_entities = sorted(list(set(existing_entities + new_entities)))
config['entity_pattern'] = r'\b(?:' + '|'.join(re.escape(e) for e in all_entities) + r')\b'

# Enhance theme_keywords
if 'AI & Cloud' not in config['theme_keywords']:
    config['theme_keywords']['AI & Cloud'] = []
config['theme_keywords']['AI & Cloud'].extend([
    'claude', 'chatgpt', 'sora', 'copilot', 'azure ai', 'aws ai', 'llama', 'bard', 'dall-e', 'midjourney', 'stable diffusion',
    'agi', 'llm', 'generative ai', 'machine learning', 'deep learning', 'neural networks'
])
config['theme_keywords']['AI & Cloud'] = list(set(config['theme_keywords']['AI & Cloud'])) # Remove duplicates

# Add a new theme for AI Ethics & Governance
if 'AI Ethics & Governance' not in config['theme_keywords']:
    config['theme_keywords']['AI Ethics & Governance'] = []
config['theme_keywords']['AI Ethics & Governance'].extend([
    'ethics', 'governance', 'regulation', 'safety', 'bias', 'responsible ai', 'fairness', 'transparency', 'privacy'
])
config['theme_keywords']['AI Ethics & Governance'] = list(set(config['theme_keywords']['AI Ethics & Governance']))

# Enhance positive_words
if 'positive_words' not in config:
    config['positive_words'] = []
config['positive_words'].extend([
    'breakthrough', 'innovative', 'pioneering', 'leading', 'upside', 'expansion', 'accelerate', 'robust', 'stronger', 'future', 'potential', 'revolution', 'transformative', 'advantage', 'leadership', 'dominant'
])
config['positive_words'] = list(set(config['positive_words']))

# Enhance negative_words
if 'negative_words' not in config:
    config['negative_words'] = []
config['negative_words'].extend([
    'challenge', 'competition', 'scrutiny', 'concern', 'bottleneck', 'limitation', 'threat', 'monopoly', 'regulatory risk', 'litigation', 'controversy', 'slowdown', 'vulnerability'
])
config['negative_words'] = list(set(config['negative_words']))

# Save the updated configuration
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)

print(f"Updated configuration saved to {config_path}")

# Now, re-run the analysis to apply the new configuration
# For simplicity, we'll call the main function here, but in a real scenario, you might want to restart the kernel and run the analysis cell again.
# Note: This will re-run the entire analysis based on the updated config.
if __name__ == '__main__':
    main(str(DEFAULT_TRANSCRIPT_PATH), config_file=str(config_path))


"""
### Expanding with Investor Analyst Terms (WSJ, Financial Times Style)
To make the NLP analysis more attuned to the language used by financial analysts in publications like the Wall Street Journal and Financial Times, we'll further enrich the `fin_nlp.json` configuration. This involves adding more nuanced positive, negative, and neutral (hedge) words, as well as refining risk and opportunity tags to capture a broader spectrum of market sentiment and strategic implications.
"""

import json
from pathlib import Path
import re

config_path = DEFAULT_FIN_NLP_CONFIG_FILE

# Load the existing configuration
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
else:
    print(f"Configuration file not found at {config_path}. Cannot update. Please ensure it exists.")
    # Exit or handle error appropriately if config is strictly required
    exit()

# Add more sophisticated positive words
config['positive_words'].extend([
    'outperform', 'upside', 'accelerating', 'robust', 'stronger', 'gaining', 'momentum', 'efficiency', 'synergy', 'optimistic',
    'strategic', 'growth', 'expansion', 'resilient', 'innovative', 'breakthrough', 'pioneering', 'leading', 'future-proof',
    'competitive advantage', 'market leadership', 'dominant', 'value creation', 'sustainable', 'promising', 'favorable'
])
config['positive_words'] = list(set(config['positive_words']))

# Add more sophisticated negative words
config['negative_words'].extend([
    'underperform', 'downside', 'decelerating', 'fragile', 'weaker', 'losing', 'pressure', 'inefficiency', 'disruption', 'pessimistic',
    'challenge', 'headwind', 'uncertainty', 'regulatory risk', 'litigation', 'controversy', 'slowdown', 'vulnerability',
    'erosion', 'dilution', 'impairment', 'liability', 'recessionary', 'bearish', 'volatile', 'geopolitical risk', 'supply chain issues'
])
config['negative_words'] = list(set(config['negative_words']))

# Add more sophisticated hedge words (neutral/uncertainty)
config['hedge_words'].extend([
    'potential', 'likely', 'could', 'might', 'may', 'expect', 'anticipated', 'forecast', 'outlook', 'guidance', 'possible',
    'speculative', 'tentative', 'contingent', 'estimate', 'projected', 'subject to', 'depends on', 'if', 'assuming'
])
config['hedge_words'] = list(set(config['hedge_words']))

# Enhance risk_tags with more financial/market specific terms
if 'Market Sentiment' not in config['risk_tags']:
    config['risk_tags']['Market Sentiment'] = []
config['risk_tags']['Market Sentiment'].extend([
    'investor confidence', 'market reaction', 'stock performance', 'valuation', 'sentiment', 'share price', 'analyst ratings'
])
config['risk_tags']['Market Sentiment'] = list(set(config['risk_tags']['Market Sentiment']))

if 'Macroeconomic' not in config['risk_tags']:
    config['risk_tags']['Macroeconomic'] = []
config['risk_tags']['Macroeconomic'].extend([
    'inflation', 'interest rates', 'GDP', 'economic slowdown', 'recession', 'geopolitical', 'trade tensions', 'currency fluctuations'
])
config['risk_tags']['Macroeconomic'] = list(set(config['risk_tags']['Macroeconomic']))

# Save the updated configuration
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)

print(f"Updated configuration (with financial analyst terms) saved to {config_path}")

# Re-run the analysis to apply the new configuration
if __name__ == '__main__':
    main(str(DEFAULT_TRANSCRIPT_PATH), config_file=str(config_path))