
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

import fitz
import re
import nltk
from nltk.stem import WordNetLemmatizer
import json



lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):

    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text


def clean_text_remove_figures_tables(text):

    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[\u200b\ufeff]', '', text)
    text = re.sub(r'\r\n?', '\n', text)

    # Удаляем подписи к рисункам
    text = re.sub(r'(?im)^\s*(figure|fig\.?)\s*\d+[^.\n]*[\.\n]?', '', text)

    # Удаляем подписи к таблицам
    text = re.sub(r'(?im)^\s*(table|tab\.?)\s*\d+[^.\n]*[\.\n]?', '', text)

    # Фильтр табличных строк
    def is_tabular_line(line):
        stripped = line.strip()
        if not stripped:
            return False
        num_ratio = sum(c.isdigit() for c in stripped) / len(stripped)
        return (
            num_ratio > 0.4
            or '\t' in stripped
            or stripped.count('  ') > 3
            or re.search(r'\|', stripped)
        )

    lines = []
    for line in text.splitlines():
        if not is_tabular_line(line):
            lines.append(line)
        elif lines and lines[-1] != '':
            lines.append('')

    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_section_name(name):

    name = name.strip().lower()
    name = re.sub(r'[:.\-–—]+$', '', name)
    name = re.sub(r'\s+', ' ', name)


    words = [lemmatizer.lemmatize(w) for w in name.split()]
    name = " ".join(words)


    normalization_map = {
        "introduction": "introduction",
        "background": "introduction",
        "method": "methods",
        "methods": "methods",
        "material and method": "methods",
        "materials and method": "methods",
        "materials and methods": "methods",
        "result": "results",
        "results": "results",
        "result and discussion": "results",
        "results and discussion": "results",
        "discussion": "discussion",
        "general discussion": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "summary": "conclusion",
        "abstract": "abstract",
        "acknowledgment": "acknowledgments",
        "acknowledgments": "acknowledgments",
        "reference": "references",
        "references": "references",
    }

    return normalization_map.get(name, name)


def split_into_sections_full_robust_lists(text):
    """Разделяет текст на разделы по заголовкам."""
    text = re.sub(r'\n{2,}', '\n\n', text)

    headers = [
        r'abstract',
        r'introduction|background',
        r'materials\s+and\s+methods|methods?',
        r'results?',
        r'discussion',
        r'conclusion[s]?',
        r'references',
        r'acknowledg(?:ement|ements|e)?'
    ]
    headers_union = '|'.join(headers)

    pattern = rf'(?:^|\n)\s*(?:\d+\.?|[IVXLCM]+\.)?\s*(?P<header>{headers_union})\b[:.]?\s*'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    if not matches:
        print(" Заголовки не найдены ")
        return {"Full text": [text.strip()]}

    sections = {}
    for i, m in enumerate(matches):
        header_raw = m.group('header')
        normalized_header = normalize_section_name(header_raw)

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].lstrip()
        section_text = re.sub(r'\n{3,}', '\n\n', section_text)

        sections.setdefault(normalized_header, []).append(section_text)

    return sections




pdf_path = "/content/oph.pdf"

raw_text = extract_text_from_pdf(pdf_path)
clean_text = clean_text_remove_figures_tables(raw_text)
sections = split_into_sections_full_robust_lists(clean_text)
output_path = "results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=2)


for name, parts in sections.items():
    print(f"\n=== {name.upper()} ===")
    for i, content in enumerate(parts, 1):
        print(f"[Part {i}] {content[:400]} ...")
