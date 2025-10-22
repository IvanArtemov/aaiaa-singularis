"""Sentence model with metadata and embedding"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Sentence:
    """
    Представление одного предложения с метаданными и embedding

    Используется для хранения предложений документа с их векторными
    представлениями для semantic search.

    Attributes:
        text: Текст предложения
        section: Название IMRAD секции (introduction, methods, results, etc.)
        position: Позиция предложения внутри секции (0-indexed)
        embedding: Векторное представление (обычно 1536-dim для text-embedding-3-small)
        char_start: Начальная позиция в тексте секции
        char_end: Конечная позиция в тексте секции
        metadata: Дополнительные метаданные (опционально)
    """

    text: str
    section: str
    position: int
    embedding: Optional[List[float]] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def has_embedding(self) -> bool:
        """
        Проверить, есть ли embedding у предложения

        Returns:
            True если embedding создан
        """
        return self.embedding is not None and len(self.embedding) > 0

    def __len__(self) -> int:
        """
        Длина текста предложения в символах

        Returns:
            Количество символов в тексте
        """
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертировать в словарь (для сериализации)

        Returns:
            Словарь с данными предложения
        """
        return {
            "text": self.text,
            "section": self.section,
            "position": self.position,
            "embedding": self.embedding,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "has_embedding": self.has_embedding(),
            "length": len(self),
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        """String representation"""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        embedding_status = "✓" if self.has_embedding() else "✗"
        return (
            f"Sentence(section='{self.section}', pos={self.position}, "
            f"embedding={embedding_status}, text='{text_preview}')"
        )
