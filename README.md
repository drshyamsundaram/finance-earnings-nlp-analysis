# Finance Earnings NLP Analysis

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/drshyamsundaram/finance-earnings-nlp-analysis)](https://github.com/drshyamsundaram/finance-earnings-nlp-analysis)
[![GitHub issues](https://img.shields.io/github/issues/drshyamsundaram/finance-earnings-nlp-analysis)](https://github.com/drshyamsundaram/finance-earnings-nlp-analysis/issues)
[![GitHub forks](https://img.shields.io/github/forks/drshyamsundaram/finance-earnings-nlp-analysis)](https://github.com/drshyamsundaram/finance-earnings-nlp-analysis/network)

A lightweight Python toolkit for earnings call transcript analysis with a polished HTML report and CSV exports.

This repository is built around earnings transcript NLP workflows, including phrase mining, sentiment, uncertainty, theme extraction, KPI discovery, and topic modeling.

## Highlights

- Import transcript content from `PDF` or `TXT`
- Generate modular NLP outputs for terms, phrases, sentiment, uncertainty, speakers, sections, entities, topics, and KPIs
- Export analysis results as CSV files
- Create an interactive HTML report with charts, filters, and summary cards
- Use the CLI for repeatable analysis

## Repository layout

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

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Run the analyzer against the included sample transcript:

```bash
python -m finance_earnings_nlp_analysis --input data/sample_transcript.txt --output output --template templates/report_template.html
```

Or run the sample helper script:

```bash
python examples/run_sample.py
```

## Generated outputs

The pipeline produces the following artifacts in `output/`:

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

## Recommended badges

Once you add CI and release automation, consider adding these badges for better project visibility:

- Build status: GitHub Actions, Azure Pipelines, or Travis CI
- Code coverage: Codecov, Coveralls, or SonarCloud
- PyPI release: version and download count
- Documentation status: Read the Docs or GitHub Pages
- Dependency health: `requires.io` or Snyk
- Security scan: GitHub Dependabot or GitHub Code Scanning
- Maintained: `maintained` or `status` badge

## Core architecture

| Module | Purpose |
|---|---|
| `io_utils.py` | Load transcripts from PDF or text files |
| `text_processing.py` | Sentence splitting, tokenization, and normalization |
| `metrics.py` | Compute analytics and create analysis dataframes |
| `report_builder.py` | Render the interactive HTML report |
| `export_utils.py` | Persist CSV files and output manifest |
| `cli.py` | Parse CLI arguments and launch analysis |
| `analyzer.py` | Orchestrate the workflow and build report payload |

## Typical workflows

- Review earnings call themes and sentiment trends
- Extract KPI-related sentences for financial modeling
- Compare prepared remarks versus Q&A sections
- Spot uncertainty and forward-looking language
- Build analyst-ready summary reports from transcripts

## Future enhancements

- Prior-quarter comparative analysis and delta scoring
- Improved named entity recognition using transformer models
- Embedding-based semantic retrieval and RAG pipelines
- Multi-company batch analysis support
- Dashboard controls for segment filtering and theme focus

## Notes

The sample transcript data is stored in `data/sample_transcript.txt` and is used for demonstration and testing.
