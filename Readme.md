# HonestAI
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.44.2+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

Система ответов на вопросы с защитой от галлюцинаций на основе языковых моделей.

## Технологии

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/-FAISS-005C84?style=flat-square&logo=facebook&logoColor=white)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/-SentenceTransformers-FF6B6B?style=flat-square)
![BitsAndBytes](https://img.shields.io/badge/-BitsAndBytes-000000?style=flat-square)

## Описание

HonestAI — это система для ответов на вопросы на русском языке, которая минимизирует галлюцинации и обеспечивает честные ответы. Система определяет логические противоречия в вопросах, оценивает возможность ответа и генерирует краткие фактологические ответы только когда уверена в результате.

## Как работает

Система состоит из четырех основных компонентов, работающих вместе:

### 1. Gate-модуль (gate.py)

Проверяет вопрос перед обработкой:
- **Обнаружение провокаций**: Выявляет логические противоречия (например, "античный математик изобрёл дизельный двигатель")
- **Оценка возможности ответа**: Анализирует структуру вопроса и определяет, можно ли на него ответить
- **Решение**: Возвращает одно из трех действий: "skip" (пропустить), "check" (проверить дополнительно), "answer" (ответить)

### 2. Основная модель (solution.py)

Координирует процесс ответа:
- **Проверка вопроса**: Использует языковую модель для анализа вопроса на провокации и определение наличия знаний
- **Кэширование**: Сохраняет ответы для консистентности при повторных и похожих вопросах
- **Генерация ответа**: Использует детерминированную генерацию (temperature=0.0) для консистентных результатов
- **Оценка уверенности**: Вычисляет лог-вероятность первых токенов для фильтрации неопределенных ответов

### 3. Ретривер (retriever.py)

Обеспечивает контекст из внешних источников:
- **FAISS-индекс**: Быстрый поиск релевантных документов по векторным эмбеддингам
- **Эмбеддинги**: Использует multilingual SentenceTransformer для представления текстов
- **Уверенность ретрива**: Оценивает качество найденных документов на основе схожести

### 4. Нормализатор (normalizer.py)

Приводит ответы к канонической форме:
- **Канонизация сущностей**: Преобразует варианты написания к стандартной форме (например, "яндекс" → "Яндекс")
- **Извлечение годов**: Для вопросов про даты извлекает только год
- **Очистка ответов**: Удаляет маркеры неопределенности и лишние слова

### Процесс работы

1. Вопрос поступает в Gate-модуль для первичной проверки
2. Если вопрос не провокационный, система проверяет наличие знаний через языковую модель
3. При необходимости система ищет контекст в ретривере
4. Генерируется ответ с оценкой уверенности
5. Если уверенность низкая, система отвечает "не знаю"
6. Ответ нормализуется и возвращается

## Что использовано

### Технологии

- **PyTorch**: Фреймворк для работы с нейронными сетями
- **Transformers**: Библиотека HuggingFace для работы с языковыми моделями
- **FAISS**: Векторный поиск для ретрива информации
- **Sentence Transformers**: Мультиязычные эмбеддинги для поиска

### Языковые модели

- **Qwen2.5-1.5B-Instruct**: Основная модель для генерации ответов
- **paraphrase-multilingual-MiniLM-L12-v2**: Модель для эмбеддингов в ретривере

### Основные библиотеки

- `transformers >= 4.44.2`: Работа с языковыми моделями
- `torch`: Тензорные вычисления
- `accelerate >= 0.33.0`: Оптимизация загрузки моделей
- `bitsandbytes`: 4-bit квантизация для экономии памяти
- `faiss-cpu`: Векторный поиск
- `sentence-transformers`: Генерация эмбеддингов
- `numpy`: Численные вычисления

### Методы защиты от галлюцинаций

1. **Детерминированная генерация**: temperature=0.0 для консистентности
2. **Проверка провокаций**: Обнаружение логических противоречий
3. **Оценка уверенности**: Фильтрация по лог-вероятностям токенов
4. **Кэширование**: Обеспечение консистентности для похожих вопросов
5. **Ретрив с контекстом**: Использование внешних источников информации
6. **Нормализация**: Приведение ответов к стандартной форме

## Как пользоваться

### Установка

1. Клонируйте репозиторий:

```bash
git clone <repository-url>
cd HonestAI
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

Или используйте Docker:

```bash
docker build -t honestai .
```

### Конфигурация

Настройте переменные окружения:

```bash
export MODEL_DIR="Qwen/Qwen2.5-1.5B-Instruct"
export USE_4BIT="false"
export HF_HUB_OFFLINE="0"
export TRANSFORMERS_OFFLINE="0"
export FAISS_INDEX_PATH="/path/to/index.faiss"
export FAISS_METADATA_PATH="/path/to/metadata.jsonl"
```

### Подготовка входных данных

Создайте файл `input.json` со списком вопросов:

```json
[
  "Какая компания осуществляет доставку роботами в Москве?",
  "Кто автор книги \"Детство в Соломбале\"?",
  "Какой античный математик изобрёл первый дизельный двигатель?"
]
```

### Запуск

Запустите систему:

```bash
python solution.py
```

Или с Docker:

```bash
docker run -v $(pwd):/workspace honestai
```

### Результаты

Результаты сохраняются в файл `output.json`:

```json
[
  "Яндекс",
  "Борис Шергин",
  "не могу ответить на вопрос"
]
```

### Использование в коде

```python
from solution import HallucinationResistantModel
from retriever import OfflineRetriever

model = HallucinationResistantModel()
retriever = OfflineRetriever()

question = "Какая компания осуществляет доставку роботами в Москве?"
answer = model.get_answer(question, retriever=retriever)
print(answer)
```

## Структура проекта

```
HonestAI/
├── solution.py          # Основная модель и логика ответов
├── gate.py              # Проверка провокаций и возможности ответа
├── retriever.py         # Поиск релевантной информации
├── normalizer.py        # Нормализация ответов
├── Dockerfile           # Конфигурация Docker-образа
├── input.json           # Входные вопросы
└── Readme.md           # Документация
```

## Особенности

- **Защита от галлюцинаций**: Многоуровневая система проверок
- **Честные ответы**: Система признает, когда не знает ответа
- **Консистентность**: Одинаковые вопросы получают одинаковые ответы
- **Производительность**: Кэширование и оптимизация загрузки моделей
- **Гибкость**: Поддержка офлайн-режима и локальных моделей

## Требования

- Python 3.8+
- CUDA (опционально, для GPU)
- 8GB+ RAM (для моделей без квантизации)
- 4GB+ RAM (с 4-bit квантизацией)

---

# HonestAI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.44.2+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

Question-answering system with hallucination resistance based on language models.

## Technologies

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/-FAISS-005C84?style=flat-square&logo=facebook&logoColor=white)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/-SentenceTransformers-FF6B6B?style=flat-square)
![BitsAndBytes](https://img.shields.io/badge/-BitsAndBytes-000000?style=flat-square)

## Description

HonestAI is a system for answering questions in Russian that minimizes hallucinations and ensures honest answers. The system detects logical contradictions in questions, evaluates answerability, and generates brief factual answers only when confident in the result.

## How it works

The system consists of four main components working together:

### 1. Gate Module (gate.py)

Checks the question before processing:
- **Provocation detection**: Identifies logical contradictions (e.g., "ancient mathematician invented diesel engine")
- **Answerability estimation**: Analyzes question structure and determines if it can be answered
- **Decision**: Returns one of three actions: "skip", "check", or "answer"

### 2. Main Model (solution.py)

Coordinates the answering process:
- **Question checking**: Uses language model to analyze questions for provocations and knowledge availability
- **Caching**: Stores answers for consistency on repeated and similar questions
- **Answer generation**: Uses deterministic generation (temperature=0.0) for consistent results
- **Confidence scoring**: Computes log-probabilities of first tokens to filter uncertain answers

### 3. Retriever (retriever.py)

Provides context from external sources:
- **FAISS index**: Fast search for relevant documents using vector embeddings
- **Embeddings**: Uses multilingual SentenceTransformer for text representation
- **Retrieval confidence**: Estimates quality of found documents based on similarity

### 4. Normalizer (normalizer.py)

Normalizes answers to canonical form:
- **Entity canonicalization**: Converts spelling variants to standard form (e.g., "яндекс" → "Яндекс")
- **Year extraction**: For date questions, extracts only the year
- **Answer cleaning**: Removes uncertainty markers and extra words

### Workflow

1. Question enters Gate module for initial check
2. If question is not provocative, system checks knowledge availability via language model
3. If needed, system searches for context in retriever
4. Answer is generated with confidence estimation
5. If confidence is low, system responds "не знаю" (don't know)
6. Answer is normalized and returned

## Technologies Used

### Technologies

- **PyTorch**: Framework for neural network operations
- **Transformers**: HuggingFace library for language models
- **FAISS**: Vector search for information retrieval
- **Sentence Transformers**: Multilingual embeddings for search

### Language Models

- **Qwen2.5-1.5B-Instruct**: Main model for answer generation
- **paraphrase-multilingual-MiniLM-L12-v2**: Embedding model for retriever

### Key Libraries

- `transformers >= 4.44.2`: Language model operations
- `torch`: Tensor computations
- `accelerate >= 0.33.0`: Model loading optimization
- `bitsandbytes`: 4-bit quantization for memory efficiency
- `faiss-cpu`: Vector search
- `sentence-transformers`: Embedding generation
- `numpy`: Numerical computations

### Hallucination Prevention Methods

1. **Deterministic generation**: temperature=0.0 for consistency
2. **Provocation checking**: Detection of logical contradictions
3. **Confidence estimation**: Filtering by token log-probabilities
4. **Caching**: Ensuring consistency for similar questions
5. **Context retrieval**: Using external information sources
6. **Normalization**: Converting answers to standard form

## Usage

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd HonestAI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or use Docker:

```bash
docker build -t honestai .
```

### Configuration

Set environment variables:

```bash
export MODEL_DIR="Qwen/Qwen2.5-1.5B-Instruct"
export USE_4BIT="false"
export HF_HUB_OFFLINE="0"
export TRANSFORMERS_OFFLINE="0"
export FAISS_INDEX_PATH="/path/to/index.faiss"
export FAISS_METADATA_PATH="/path/to/metadata.jsonl"
```

### Prepare Input Data

Create `input.json` file with list of questions:

```json
[
  "Какая компания осуществляет доставку роботами в Москве?",
  "Кто автор книги \"Детство в Соломбале\"?",
  "Какой античный математик изобрёл первый дизельный двигатель?"
]
```

### Run

Run the system:

```bash
python solution.py
```

Or with Docker:

```bash
docker run -v $(pwd):/workspace honestai
```

### Results

Results are saved to `output.json`:

```json
[
  "Яндекс",
  "Борис Шергин",
  "не могу ответить на вопрос"
]
```

### Code Usage

```python
from solution import HallucinationResistantModel
from retriever import OfflineRetriever

model = HallucinationResistantModel()
retriever = OfflineRetriever()

question = "Какая компания осуществляет доставку роботами в Москве?"
answer = model.get_answer(question, retriever=retriever)
print(answer)
```

## Project Structure

```
HonestAI/
├── solution.py          # Main model and answer logic
├── gate.py              # Provocation and answerability checking
├── retriever.py         # Relevant information retrieval
├── normalizer.py        # Answer normalization
├── Dockerfile           # Docker image configuration
├── input.json           # Input questions
└── Readme.md           # Documentation
```

## Features

- **Hallucination resistance**: Multi-level verification system
- **Honest answers**: System admits when it doesn't know the answer
- **Consistency**: Same questions receive same answers
- **Performance**: Caching and model loading optimization
- **Flexibility**: Support for offline mode and local models

## Requirements

- Python 3.8+
- CUDA (optional, for GPU)
- 8GB+ RAM (for models without quantization)
- 4GB+ RAM (with 4-bit quantization)

