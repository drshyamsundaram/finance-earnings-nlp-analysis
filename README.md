# Finance Earnings NLP analysis

A modular Python project for transcript-driven NLP analysis of earnings calls, with an interactive HTML report and CSV extracts for analysts.

This sample project is designed around the Alphabet Q1 2026 earnings transcript and its themes such as Search growth, Cloud acceleration, Gemini adoption, backlog expansion, capex intensity, and forward-looking management commentary. [file:2]

## Features

- PDF or text transcript ingestion.
- Modular NLP pipelines for terms, phrases, sentiment, uncertainty, speakers, sections, entities, topics, KPIs, co-occurrence, and semantic search index.
- Interactive HTML report with charts, tables, filters, and search.
- Individual CSV outputs for each analysis layer.
- CLI entrypoint for batch-style usage.
- Project structure suitable for GitHub publishing and extension.

## Project structure

```text
finance-earnings-nlp-analysis/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── finance_earnings_nlp_analysis/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── analyzer.py
│       ├── config.py
│       ├── io_utils.py
│       ├── text_processing.py
│       ├── metrics.py
│       ├── report_builder.py
│       └── export_utils.py
├── templates/
│   └── report_template.html
├── examples/
│   └── run_sample.py
├── tests/
│   └── test_smoke.py
├── docs/
│   └── architecture.md
├── data/
│   └── sample_transcript.txt
└── output/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required packages are `pandas`, `scikit-learn`, and `PyMuPDF`, because the code uses DataFrame exports, vectorizers and topic models from scikit-learn, and `fitz` for PDF parsing. [code_file:4]

## Quick start

Run against a transcript file:

```bash
python -m finance_earnings_nlp_analysis --input data/sample_transcript.txt --output output
```

Or run the example script:

```bash
python examples/run_sample.py
```

## Outputs

The project writes:

- `finance-earnings-nlp-analysis.html`
- `analysis_manifest.csv`
- `term_frequency.csv`
- `phrase_analysis.csv`
- `sentiment_analysis.csv`
- `uncertainty_analysis.csv`
- `speaker_analysis.csv`
- `section_analysis.csv`
- `theme_analysis.csv`
- `risk_opportunity_analysis.csv`
- `key_quote_analysis.csv`
- `entity_analysis.csv`
- `kpi_extraction.csv`
- `topic_model_analysis.csv`
- `semantic_search_index.csv`
- `cooccurrence_analysis.csv`
- `comparative_analysis.csv`

These align with the modular analysis layers already implemented in the current utility. [code_file:4]

## Architecture

### Core modules

| Module | Responsibility |
|---|---|
| `io_utils.py` | Load transcript text from PDF or TXT |
| `text_processing.py` | Sentence splitting, tokenization, stopword filtering |
| `metrics.py` | All NLP analysis layers and dataframe creation |
| `report_builder.py` | Build populated interactive HTML report |
| `export_utils.py` | Write CSV outputs and manifest |
| `cli.py` | Command-line entrypoint |
| `analyzer.py` | Orchestration class |

### Design principles

- Keep each analysis extract independent so analysts can reuse a single CSV without loading the whole pipeline.
- Keep report rendering separate from NLP computation.
- Make transcript ingestion pluggable for PDF, TXT, or future API-based sources.
- Preserve analyst-friendly outputs with transparent intermediate tables.

## Example use cases

- Earnings-call review for equity research.
- Management tone tracking across quarters.
- Cloud, AI, and advertising theme extraction.
- KPI sentence harvesting for financial models.
- Q&A analysis and speaker-level emphasis studies.

## Roadmap

- Prior-quarter comparative analysis and language-delta scoring.
- Better named entity recognition with spaCy or transformer models.
- Embedding-based semantic retrieval.
- Multi-company batch processing.
- Dashboard filters for prepared remarks versus Q&A.

## Notes

The sample transcript text included in `data/sample_transcript.txt` reflects the official Alphabet Q1 2026 transcript themes and figures such as Search growth of 19%, Cloud growth of 63%, Cloud revenue above $20 billion, and backlog near $462 billion. [file:2]
