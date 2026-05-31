import argparse
from pathlib import Path
from .analyzer import FinanceEarningsNLPAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Finance Earnings NLP analysis')
    parser.add_argument('--input', required=True, help='Path to transcript PDF or TXT')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--template', default='templates/report_template.html', help='HTML template path')
    args = parser.parse_args()

    analyzer = FinanceEarningsNLPAnalyzer(args.input, args.output, args.template)
    result = analyzer.run()
    print(result)


if __name__ == '__main__':
    main()
