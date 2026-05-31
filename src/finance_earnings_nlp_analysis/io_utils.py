from pathlib import Path


def load_transcript(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == '.pdf':
        import fitz
        doc = fitz.open(str(p))
        return '\n'.join(page.get_text('text') for page in doc)
    return p.read_text(encoding='utf-8', errors='ignore')
