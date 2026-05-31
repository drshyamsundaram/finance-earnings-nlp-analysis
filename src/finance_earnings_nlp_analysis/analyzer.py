from pathlib import Path
import pandas as pd
from .io_utils import load_transcript
from .text_processing import split_sentences, tokenize
from .metrics import term_frequency, phrase_analysis, sentiment_analysis, uncertainty_analysis, theme_analysis, topic_model_analysis, kpi_extraction
from .report_builder import build_html
from .export_utils import export_frames


class FinanceEarningsNLPAnalyzer:
    def __init__(self, input_path: str, output_dir: str, template_path: str):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.template_path = template_path

    def run(self):
        text = load_transcript(self.input_path)
        sentences = split_sentences(text)
        tokens = tokenize(text)

        frames = {
            'term_frequency': term_frequency(tokens),
            'phrase_analysis': phrase_analysis(sentences),
            'sentiment_analysis': sentiment_analysis(sentences),
            'uncertainty_analysis': uncertainty_analysis(sentences),
            'theme_analysis': theme_analysis(tokens),
            'topic_model_analysis': topic_model_analysis(sentences),
            'kpi_extraction': kpi_extraction(sentences),
        }
        export_frames(frames, str(self.output_dir))

        payload = {
            'document': Path(self.input_path).name,
            'sentence_count': len(sentences),
            'token_count': len(tokens),
            'unique_terms': len(set(tokens)),
            'top_terms': frames['term_frequency'].head(12).to_dict(orient='records'),
            'top_phrases': frames['phrase_analysis'].head(12).to_dict(orient='records'),
            'sentiment_summary': {
                'positive': int((frames['sentiment_analysis']['score'] > 0).sum()),
                'neutral': int((frames['sentiment_analysis']['score'] == 0).sum()),
                'negative': int((frames['sentiment_analysis']['score'] < 0).sum())
            },
            'themes': frames['theme_analysis'].to_dict(orient='records'),
            'topics': frames['topic_model_analysis'].to_dict(orient='records'),
            'kpis': frames['kpi_extraction'].head(12).to_dict(orient='records')
        }

        html = build_html(payload, self.template_path)
        html_path = self.output_dir / 'finance-earnings-nlp-analysis.html'
        html_path.write_text(html, encoding='utf-8')
        return {'html': str(html_path), 'manifest': str(self.output_dir / 'analysis_manifest.csv')}
