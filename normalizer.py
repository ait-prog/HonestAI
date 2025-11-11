import re
from typing import Dict, Optional

CANONICAL_FORMS: Dict[str, str] = {
    "яндекс": "Яндекс",
    "яндекса": "Яндекс",
    "яндексу": "Яндекс",
    "яндексом": "Яндекс",
    "яндексe": "Яндекс",
    "yandex": "Яндекс",
    "яндекс лавка": "Яндекс Лавка",
    "яндекс лавки": "Яндекс Лавка",
}

def normalize_year(text: str) -> Optional[str]:
    years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
    if years:
        return years[0]
    return None

def normalize_answer(answer: str, question: str = "") -> str:
    if not answer or answer.lower() in ["не знаю", "не могу ответить на вопрос"]:
        return answer
    
    answer = answer.strip()
    
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith('«') and answer.endswith('»')):
        answer = answer[1:-1].strip()
    
    if "год" in question.lower() or "когда" in question.lower():
        year = normalize_year(answer)
        if year:
            return year
    
    answer_lower = answer.lower()
    for variant, canonical in CANONICAL_FORMS.items():
        if variant in answer_lower:
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            answer = pattern.sub(canonical, answer)
            break
    
    if "имеет" in answer.lower() or "имеет доставку" in answer.lower():
        parts = answer.split()
        if len(parts) > 1:
            answer = parts[0]
    
    uncertainty_phrases = [
        "по моим данным",
        "насколько известно",
        "вероятно",
        "возможно",
        "скорее всего",
    ]
    for phrase in uncertainty_phrases:
        if answer.lower().startswith(phrase):
            answer = answer[len(phrase):].strip()
            if len(answer) < 3:
                return "не знаю"
    
    answer = answer.split('\n')[0].strip()
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    return answer if answer else "не знаю"

def extract_key_phrase(answer: str, question_type: str = "") -> str:
    if question_type == "who":
        parts = answer.split()
        if parts:
            return parts[0]
    elif question_type == "when" or "год" in question_type.lower():
        year = normalize_year(answer)
        if year:
            return year
    elif question_type == "where":
        parts = answer.split(',')
        if parts:
            return parts[0].strip()
    
    return answer
