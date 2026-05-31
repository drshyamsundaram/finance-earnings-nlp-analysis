import json
from pathlib import Path


def build_html(payload: dict, template_path: str) -> str:
    template = Path(template_path).read_text(encoding='utf-8')
    return template.replace('__REPORT_DATA__', json.dumps(payload))
