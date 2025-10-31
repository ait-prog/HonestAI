"""
Модуль для офлайн-ретрива информации из корпуса
Использует FAISS-индекс и эмбеддинги для поиска релевантных документов
"""
import os
import json
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS не установлен, ретривер будет использовать упрощенную версию")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers не установлен, ретривер будет использовать упрощенную версию")


class OfflineRetriever:
    """Офлайн-ретривер для поиска релевантной информации"""
    
    def __init__(self, 
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            index_path: путь к FAISS-индексу (если None, используется упрощенный режим)
            metadata_path: путь к JSONL с метаданными документов
            embedding_model_name: название модели для эмбеддингов
        """
        self.index = None
        self.metadata = []
        self.embedder = None
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH")
        self.metadata_path = metadata_path or os.getenv("FAISS_METADATA_PATH")
        
        # Попытка загрузить FAISS-индекс
        if FAISS_AVAILABLE and self.index_path and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"Загружен FAISS-индекс из {self.index_path}")
            except Exception as e:
                print(f"Не удалось загрузить FAISS-индекс: {e}")
        
        # Загрузка метаданных
        if self.metadata_path and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.metadata.append(json.loads(line))
                print(f"Загружено {len(self.metadata)} документов из метаданных")
            except Exception as e:
                print(f"Не удалось загрузить метаданные: {e}")
        
        # Инициализация эмбеддера (если доступен)
        if SENTENCE_TRANSFORMERS_AVAILABLE and not self.index:
            try:
                # Используем легкую модель для CPU
                self.embedder = SentenceTransformer(embedding_model_name)
                print(f"Загружен эмбеддер: {embedding_model_name}")
            except Exception as e:
                print(f"Не удалось загрузить эмбеддер: {e}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Ищет релевантные документы для запроса
        
        Returns:
            List[(text, score), ...] отсортированных по релевантности
        """
        if self.index is not None and self.embedder is not None:
            # Полноценный FAISS-поиск
            return self._retrieve_faiss(query, top_k)
        elif self.embedder is not None and self.metadata:
            # Упрощенный поиск с эмбеддингами без FAISS
            return self._retrieve_simple(query, top_k)
        else:
            # Минимальный fallback - возвращаем пустой результат
            return []
    
    def _retrieve_faiss(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Поиск через FAISS-индекс"""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        
        # Нормализуем для косинусного расстояния
        faiss.normalize_L2(query_embedding)
        
        # Поиск
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                doc = self.metadata[idx]
                score = float(1.0 - distances[0][i])  # Преобразуем расстояние в схожесть
                text = doc.get('text', doc.get('title', ''))
                results.append((text, score))
        
        return results
    
    def _retrieve_simple(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Упрощенный поиск без FAISS (вычисление схожести напрямую)"""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Нормализация
        
        similarities = []
        for doc in self.metadata:
            doc_text = doc.get('text', doc.get('title', ''))
            doc_embedding = self.embedder.encode([doc_text], convert_to_numpy=True)[0]
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)
            
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((doc_text, float(similarity)))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Получает контекст для вопроса (объединенные top-k документов)"""
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        
        context_parts = []
        for text, score in results:
            # Берем только первые 200 символов каждого документа
            short_text = text[:200] + "..." if len(text) > 200 else text
            context_parts.append(short_text)
        
        return "\n\n".join(context_parts)
    
    def compute_confidence(self, query: str, top_k: int = 5) -> float:
        """Вычисляет уверенность на основе качества ретрива
        
        Returns:
            float от 0.0 до 1.0
        """
        results = self.retrieve(query, top_k)
        if not results:
            return 0.0
        
        # Уверенность зависит от:
        # 1. Качества top-1 результата
        top1_score = results[0][1] if results else 0.0
        
        # 2. Разницы между top-1 и top-2 (чем больше разница, тем выше уверенность)
        margin = 0.0
        if len(results) > 1:
            margin = results[0][1] - results[1][1]
        
        # Комбинированная уверенность
        confidence = top1_score * 0.7 + min(margin, 0.3) * 0.3
        
        return min(1.0, max(0.0, confidence))


# Заглушка для случаев, когда ретривер недоступен
class DummyRetriever:
    """Заглушка ретривера, когда FAISS/embeddings недоступны"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        return []
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        return ""
    
    def compute_confidence(self, query: str, top_k: int = 5) -> float:
        return 0.0

