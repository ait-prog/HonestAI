import re
from typing import Tuple, Set

PROVOCATION_PATTERNS = [
    (r"античн[ыйаяое]+\s+\w+\s+(изобрёл|изобрел|создал)", True),
    (r"античн[ыйаяое]+\s+\w+\s+дизель", True),
    (r"древн[ийяе]+\s+\w+\s+двигатель", True),
    (r"кто\s+написал\s+[\"«]([^\"»]+)[\"»]", None),
    (r"сколько\s+серий\s+в\s+[\"«]([^\"»]+)[\"»]", None),
    (r"в\s+каком\s+году\s+\w+\s+изобрёл\s+[\"«]([^\"»]+)[\"»]", None),
]

KNOWN_ENTITIES: Set[str] = {
}

def check_provocation(question: str) -> Tuple[bool, str]:
    q_lower = question.lower()
    
    for pattern, is_definite in PROVOCATION_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            if is_definite:
                return True, f"Паттерн провокации: {pattern}"
            if match.groups():
                entity = match.group(1).lower().strip()
    
    contradiction_indicators = [
        ("античный", "дизель"),
        ("античный", "двигатель"),
        ("древний", "компьютер"),
        ("средневековый", "самолет"),
    ]
    
    for ind1, ind2 in contradiction_indicators:
        if ind1 in q_lower and ind2 in q_lower:
            return True, f"Противоречие: {ind1} + {ind2}"
    
    return False, ""

def estimate_answerability(question: str, question_lower: str = None) -> float:
    if question_lower is None:
        question_lower = question.lower()
    
    positive_score = 0.0
    
    if re.search(r'["«]([^"»]+)["»]', question):
        positive_score += 0.3
    
    if "в каком году" in question_lower or "когда" in question_lower:
        positive_score += 0.2
    
    if "компания" in question_lower or "корпорация" in question_lower:
        positive_score += 0.2
    
    if "автор" in question_lower and "книг" in question_lower:
        positive_score += 0.2
    
    negative_score = 0.0
    
    if len(question.split()) < 4:
        negative_score += 0.3
    
    if not re.search(r'["«]', question) and not re.search(r'\b\d{4}\b', question):
        negative_score += 0.2
    
    answerability = max(0.0, min(1.0, 0.5 + positive_score - negative_score))
    
    return answerability

def should_attempt_answer(question: str, 
                         answerability_threshold_low: float = 0.3,
                         answerability_threshold_high: float = 0.7) -> Tuple[str, float]:
    is_prov, reason = check_provocation(question)
    if is_prov:
        return "skip", 1.0
    
    answerability = estimate_answerability(question)
    
    if answerability < answerability_threshold_low:
        return "skip", answerability
    elif answerability < answerability_threshold_high:
        return "check", answerability
    else:
        return "answer", answerability
