from pathlib import Path
from finance_earnings_nlp_analysis.analyzer import FinanceEarningsNLPAnalyzer

root = Path(__file__).resolve().parents[1]
analyzer = FinanceEarningsNLPAnalyzer(
    input_path=str(root / 'data' / 'sample_transcript.txt'),
    output_dir=str(root / 'output'),
    template_path=str(root / 'templates' / 'report_template.html'),
)
print(analyzer.run())
