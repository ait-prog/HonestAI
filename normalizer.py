"""
Модуль для нормализации ответов: канонизация имен, годов, компаний
"""
import re
from typing import Dict, Optional

# Словарь канонических форм для популярных сущностей
CANONICAL_FORMS: Dict[str, str] = {
    # Компании
    "яндекс": "Яндекс",
    "яндекса": "Яндекс",
    "яндексу": "Яндекс",
    "яндексом": "Яндекс",
    "яндексe": "Яндекс",
    "yandex": "Яндекс",
    
    "яндекс лавка": "Яндекс Лавка",
    "яндекс лавки": "Яндекс Лавка",
    
    # Добавьте сюда другие популярные сущности
}

def normalize_year(text: str) -> Optional[str]:
    """Извлекает и нормализует год в формате YYYY"""
    # Ищем 4-значные числа от 1000 до 2099
    years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
    if years:
        return years[0]
    return None

def normalize_answer(answer: str, question: str = "") -> str:
    """Нормализует ответ: канонизирует имена, извлекает годы, убирает лишнее"""
    if not answer or answer.lower() in ["не знаю", "не могу ответить на вопрос"]:
        return answer
    
    answer = answer.strip()
    
    # Удаляем кавычки, если они окружают весь ответ
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith('«') and answer.endswith('»')):
        answer = answer[1:-1].strip()
    
    # Для вопросов про годы - извлекаем только год
    if "год" in question.lower() or "когда" in question.lower():
        year = normalize_year(answer)
        if year:
            return year
    
    # Нормализуем имена/названия компаний к канонической форме
    answer_lower = answer.lower()
    for variant, canonical in CANONICAL_FORMS.items():
        if variant in answer_lower:
            # Заменяем вариант на каноническую форму, сохраняя регистр первого символа
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            answer = pattern.sub(canonical, answer)
            break
    
    # Убираем лишние слова-хвосты
    # "Яндекс имеет доставку" -> "Яндекс"
    if "имеет" in answer.lower() or "имеет доставку" in answer.lower():
        parts = answer.split()
        if len(parts) > 1:
            # Берем только первое слово (обычно это название)
            answer = parts[0]
    
    # Убираем маркеры неопределенности в начале
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
            # Если после удаления осталась только точка или мало текста
            if len(answer) < 3:
                return "не знаю"
    
    # Берем только первую строку (убираем многострочные ответы)
    answer = answer.split('\n')[0].strip()
    
    # Убираем лишние пробелы
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    return answer if answer else "не знаю"

def extract_key_phrase(answer: str, question_type: str = "") -> str:
    """Извлекает ключевую фразу из ответа в зависимости от типа вопроса"""
    if question_type == "who":
        # Для вопросов "кто" - берем первое слово/имя
        parts = answer.split()
        if parts:
            return parts[0]
    elif question_type == "when" or "год" in question_type.lower():
        # Для вопросов "когда" - только год
        year = normalize_year(answer)
        if year:
            return year
    elif question_type == "where":
        # Для вопросов "где" - первое слово (обычно название места)
        parts = answer.split(',')
        if parts:
            return parts[0].strip()
    
    return answer

