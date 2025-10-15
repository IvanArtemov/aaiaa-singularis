"""Sentence embedder for creating vector representations of document sentences"""

import time
from typing import List, Dict, Any, Optional

from src.models.sentence import Sentence
from src.parsers import ParsedDocument
from src.llm_adapters import get_llm_adapter


class SentenceEmbedder:
    """
    Разбивка документа на предложения и создание embeddings

    Обрабатывает каждую IMRAD секцию отдельно, создавая список
    Sentence объектов с векторными представлениями для semantic search.

    Usage:
        embedder = SentenceEmbedder(llm_provider="openai")
        parsed_doc = embedder.process_document(parsed_doc)
        print(f"Created {len(parsed_doc.sentences)} sentences")
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        batch_size: int = 100,
        min_sentence_length: int = 10,
        cache_size: int = 256
    ):
        """
        Initialize sentence embedder

        Args:
            llm_provider: LLM provider for embeddings ("openai" or "ollama")
            batch_size: Number of sentences to embed in one API call
            min_sentence_length: Minimum sentence length to include (chars)
            cache_size: Size of LRU cache for duplicate sentences
        """
        self.llm_provider = llm_provider
        self.batch_size = batch_size
        self.min_sentence_length = min_sentence_length
        self.cache_size = cache_size

        # Lazy initialization
        self.nlp = None
        self.llm_adapter = None

        # Metrics
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_sentences = 0
        self.cache_hits = 0
        self.embedding_time = 0.0

        # Cache for duplicate sentences
        self._embedding_cache = {}

    def _get_nlp(self):
        """Lazy load spaCy model"""
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
        return self.nlp

    def _get_llm_adapter(self):
        """Lazy load LLM adapter"""
        if self.llm_adapter is None:
            self.llm_adapter = get_llm_adapter(self.llm_provider)
        return self.llm_adapter

    def process_document(
        self,
        parsed_doc: ParsedDocument,
        create_embeddings: bool = True
    ) -> ParsedDocument:
        """
        Разбить документ на предложения и создать embeddings

        Args:
            parsed_doc: ParsedDocument с IMRAD секциями
            create_embeddings: Создавать ли embeddings (False для тестов)

        Returns:
            ParsedDocument с заполненным полем sentences
        """
        if not parsed_doc.imrad_sections:
            print("Warning: No IMRAD sections found. Cannot extract sentences.")
            return parsed_doc

        # Step 1: Разбить все секции на предложения
        all_sentences = []

        for section_name, section_text in parsed_doc.imrad_sections.items():
            sentences = self._extract_sentences_from_section(
                section_text,
                section_name
            )
            all_sentences.extend(sentences)

        self.total_sentences = len(all_sentences)

        # Step 2: Создать embeddings (если нужно)
        if create_embeddings and len(all_sentences) > 0:
            self._create_embeddings(all_sentences)

        # Step 3: Сохранить в ParsedDocument
        parsed_doc.sentences = all_sentences

        return parsed_doc

    def _extract_sentences_from_section(
        self,
        section_text: str,
        section_name: str
    ) -> List[Sentence]:
        """
        Разбить текст секции на Sentence объекты

        Args:
            section_text: Текст секции
            section_name: Название секции (e.g., "introduction")

        Returns:
            List of Sentence objects (без embeddings)
        """
        nlp = self._get_nlp()
        doc = nlp(section_text)

        sentences = []

        for position, sent in enumerate(doc.sents):
            text = sent.text.strip()

            # Фильтр: минимальная длина
            if len(text) < self.min_sentence_length:
                continue

            # Фильтр: только буквы/цифры (исключаем артефакты)
            if not any(c.isalnum() for c in text):
                continue

            sentence = Sentence(
                text=text,
                section=section_name,
                position=position,
                char_start=sent.start_char,
                char_end=sent.end_char,
                embedding=None  # Заполнится в _create_embeddings
            )
            sentences.append(sentence)

        return sentences

    def _create_embeddings(self, sentences: List[Sentence]):
        """
        Создать embeddings для всех предложений (batch processing)

        Args:
            sentences: List of Sentence objects
        """
        llm = self._get_llm_adapter()

        start_time = time.time()

        # Batch processing для экономии времени
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]

            # Проверить кэш для каждого предложения
            texts_to_embed = []
            cached_embeddings = []

            for sentence in batch:
                cached = self._get_from_cache(sentence.text)
                if cached is not None:
                    cached_embeddings.append(cached)
                    self.cache_hits += 1
                else:
                    texts_to_embed.append(sentence.text)
                    cached_embeddings.append(None)

            # Получить embeddings для не-кэшированных предложений
            if texts_to_embed:
                new_embeddings = llm.embed(texts_to_embed)

                # Сохранить в кэш
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    self._save_to_cache(text, embedding)
            else:
                new_embeddings = []

            # Присвоить embeddings предложениям
            new_embedding_idx = 0
            for sentence, cached in zip(batch, cached_embeddings):
                if cached is not None:
                    sentence.embedding = cached
                else:
                    sentence.embedding = new_embeddings[new_embedding_idx]
                    new_embedding_idx += 1

            # Оценка токенов (примерная)
            batch_tokens = sum(len(s.text.split()) * 1.3 for s in batch)
            self.total_tokens += batch_tokens

        # Оценка стоимости (для text-embedding-3-small: $0.02/1M tokens)
        self.total_cost = (self.total_tokens / 1_000_000) * 0.02
        self.embedding_time = time.time() - start_time

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Получить embedding из кэша"""
        return self._embedding_cache.get(text)

    def _save_to_cache(self, text: str, embedding: List[float]):
        """Сохранить embedding в кэш"""
        # Ограничить размер кэша
        if len(self._embedding_cache) >= self.cache_size:
            # Удалить первый элемент (FIFO)
            first_key = next(iter(self._embedding_cache))
            del self._embedding_cache[first_key]

        self._embedding_cache[text] = embedding

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получить метрики создания embeddings

        Returns:
            Dictionary with metrics
        """
        cache_hit_rate = 0.0
        if self.total_sentences > 0:
            cache_hit_rate = self.cache_hits / self.total_sentences

        return {
            "total_sentences": self.total_sentences,
            "total_tokens": int(self.total_tokens),
            "total_cost_usd": round(self.total_cost, 6),
            "embedding_time_seconds": round(self.embedding_time, 2),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "sentences_per_second": round(
                self.total_sentences / self.embedding_time if self.embedding_time > 0 else 0,
                2
            )
        }

    def reset_metrics(self):
        """Reset all metrics counters"""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_sentences = 0
        self.cache_hits = 0
        self.embedding_time = 0.0
