"""
Модуль для классификации возможности ответа на вопрос (gate/сторож)
Проверяет, можно ли ответить на вопрос и нет ли провокаций
"""
import re
from typing import Tuple, Set

# Паттерны провокаций (логические противоречия)
PROVOCATION_PATTERNS = [
    # Временные противоречия (античность + современные технологии)
    (r"античн[ыйаяое]+\s+\w+\s+(изобрёл|изобрел|создал)", True),
    (r"античн[ыйаяое]+\s+\w+\s+дизель", True),
    (r"древн[ийяе]+\s+\w+\s+двигатель", True),
    
    # Несуществующие сущности (можно расширить)
    (r"кто\s+написал\s+[\"«]([^\"»]+)[\"»]", None),  # Нужна проверка через словарь
    (r"сколько\s+серий\s+в\s+[\"«]([^\"»]+)[\"»]", None),  # Нужна проверка через словарь
    
    # Вопросы с заведомо ложными предпосылками
    (r"в\s+каком\s+году\s+\w+\s+изобрёл\s+[\"«]([^\"»]+)[\"»]", None),
]

# Словарь известных книг/фильмов/произведений (для проверки существования)
# В реальной системе это должен быть большой словарь
KNOWN_ENTITIES: Set[str] = {
    # Добавьте известные сущности из корпуса
}

def check_provocation(question: str) -> Tuple[bool, str]:
    """Проверяет, является ли вопрос провокацией (логическое противоречие)
    
    Returns:
        (is_provocation, reason)
    """
    q_lower = question.lower()
    
    # Проверка явных паттернов противоречий
    for pattern, is_definite in PROVOCATION_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            if is_definite:
                return True, f"Паттерн провокации: {pattern}"
            # Если паттерн требует дополнительной проверки
            if match.groups():
                entity = match.group(1).lower().strip()
                # Проверяем, существует ли сущность (требует словаря)
                # Для упрощения считаем, что если не в известных - возможная провокация
                # В реальной системе здесь должна быть проверка по корпусу
    
    # Проверка на множественные противоречия
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
    """Оценивает вероятность того, что можно дать ответ на вопрос
    
    Returns:
        float от 0.0 до 1.0 (1.0 = точно можно ответить)
    """
    if question_lower is None:
        question_lower = question.lower()
    
    # Положительные индикаторы (хорошие вопросы)
    positive_score = 0.0
    
    # Вопросы с конкретными сущностями в кавычках (книги, фильмы)
    if re.search(r'["«]([^"»]+)["»]', question):
        positive_score += 0.3
    
    # Вопросы про годы (обычно ответить можно)
    if "в каком году" in question_lower or "когда" in question_lower:
        positive_score += 0.2
    
    # Вопросы про компании (обычно есть информация)
    if "компания" in question_lower or "корпорация" in question_lower:
        positive_score += 0.2
    
    # Вопросы про авторов книг
    if "автор" in question_lower and "книг" in question_lower:
        positive_score += 0.2
    
    # Отрицательные индикаторы (плохие вопросы)
    negative_score = 0.0
    
    # Слишком абстрактные вопросы
    if len(question.split()) < 4:
        negative_score += 0.3
    
    # Вопросы без конкретных сущностей
    if not re.search(r'["«]', question) and not re.search(r'\b\d{4}\b', question):
        negative_score += 0.2
    
    # Итоговая оценка
    answerability = max(0.0, min(1.0, 0.5 + positive_score - negative_score))
    
    return answerability

def should_attempt_answer(question: str, 
                         answerability_threshold_low: float = 0.3,
                         answerability_threshold_high: float = 0.7) -> Tuple[str, float]:
    """Определяет, нужно ли пытаться ответить на вопрос
    
    Returns:
        (action, confidence) где action: "skip", "check", "answer"
    """
    # Сначала проверяем провокацию
    is_prov, reason = check_provocation(question)
    if is_prov:
        return "skip", 1.0
    
    # Оцениваем возможность ответа
    answerability = estimate_answerability(question)
    
    if answerability < answerability_threshold_low:
        return "skip", answerability
    elif answerability < answerability_threshold_high:
        return "check", answerability
    else:
        return "answer", answerability

