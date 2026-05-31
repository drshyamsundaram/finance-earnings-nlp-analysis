from finance_earnings_nlp_analysis.text_processing import split_sentences, tokenize


def test_text_processing_smoke():
    text = 'Revenue grew 20%. Cloud was strong. Guidance remains constructive.'
    assert len(split_sentences(text)) >= 2
    assert 'revenue' in tokenize(text)
