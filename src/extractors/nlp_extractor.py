"""NLP-based extractor using spaCy for facts extraction"""

import uuid
from typing import List, Optional, Set
from collections import Counter

try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.models import Entity, EntityType


class NLPFactExtractor:
    """
    Extract facts using NLP (spaCy)

    Extracts established knowledge and background information
    based on:
    - Named entities (ORG, PERSON, GPE, etc.)
    - Noun phrases with determiners suggesting established knowledge
    - Sentences with past tense verbs indicating prior research
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        confidence_threshold: float = 0.6,
        max_fact_length: int = 200
    ):
        """
        Initialize NLP fact extractor

        Args:
            model_name: spaCy model to use
            confidence_threshold: Minimum confidence for extraction
            max_fact_length: Maximum length for extracted facts (words)
        """
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required for NLP extraction. Install with: pip install spacy")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_fact_length = max_fact_length
        self.nlp = None

    def _load_model(self):
        """Lazy load spaCy model"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Install with: python -m spacy download {self.model_name}"
                )

    def extract(
        self,
        text: str,
        section_name: Optional[str] = None
    ) -> List[Entity]:
        """
        Extract facts from text using NLP

        Args:
            text: Text to extract from
            section_name: Name of section (for metadata)

        Returns:
            List of extracted fact entities
        """
        self._load_model()

        entities = []
        doc = self.nlp(text)

        # Extract facts from sentences
        for sent in doc.sents:
            fact_candidates = []

            # Method 1: Sentences with citations/references
            if self._has_citation(sent):
                fact_candidates.append((sent.text.strip(), 0.8, "citation"))

            # Method 2: Sentences with established knowledge patterns
            if self._has_established_pattern(sent):
                fact_candidates.append((sent.text.strip(), 0.75, "established_pattern"))

            # Method 3: Sentences with key entities and past tense
            if self._has_entities_and_past_tense(sent):
                fact_candidates.append((sent.text.strip(), 0.7, "entity_past_tense"))

            # Method 4: Noun phrases indicating background knowledge
            for chunk in sent.noun_chunks:
                if self._is_background_knowledge(chunk):
                    # Extract sentence containing this chunk
                    fact_candidates.append((sent.text.strip(), 0.65, "background_noun_phrase"))

            # Deduplicate and filter
            for fact_text, confidence, method in fact_candidates:
                # Skip if too long
                if len(fact_text.split()) > self.max_fact_length:
                    continue

                # Skip if below threshold
                if confidence < self.confidence_threshold:
                    continue

                # Create entity
                entity = Entity(
                    id=self._generate_id(),
                    type=EntityType.FACT,
                    text=fact_text,
                    confidence=confidence,
                    source_section=section_name,
                    metadata={
                        "extraction_method": "nlp",
                        "nlp_method": method
                    }
                )

                # Avoid duplicates
                if not any(e.text == entity.text for e in entities):
                    entities.append(entity)
                    break  # Only add once per sentence

        return entities

    def _has_citation(self, sent: Span) -> bool:
        """Check if sentence contains citation pattern"""
        text = sent.text
        # Common citation patterns
        citation_patterns = [
            r'\[\d+\]',  # [1]
            r'\(\d{4}\)',  # (2020)
            r'\b(et al\.|et al)\b',  # et al.
            r'\b\d+\s*[-–]\s*\d+\b',  # page numbers
        ]
        import re
        return any(re.search(p, text) for p in citation_patterns)

    def _has_established_pattern(self, sent: Span) -> bool:
        """Check if sentence has patterns indicating established knowledge"""
        # Phrases indicating established facts
        established_phrases = [
            "previous studies",
            "prior research",
            "it is known",
            "it has been shown",
            "established",
            "well-known",
            "widely accepted",
            "consensus",
            "evidence suggests",
            "literature shows",
            "research indicates"
        ]

        text_lower = sent.text.lower()
        return any(phrase in text_lower for phrase in established_phrases)

    def _has_entities_and_past_tense(self, sent: Span) -> bool:
        """Check if sentence has entities and past tense verbs"""
        # Check for named entities
        has_entities = len(sent.ents) > 0

        # Check for past tense verbs
        has_past_tense = any(
            token.pos_ == "VERB" and token.tag_ in ["VBD", "VBN"]
            for token in sent
        )

        return has_entities and has_past_tense

    def _is_background_knowledge(self, chunk: Span) -> bool:
        """Check if noun phrase indicates background knowledge"""
        text_lower = chunk.text.lower()

        # Indicators of background knowledge
        background_indicators = [
            "the role of",
            "the function of",
            "the mechanism of",
            "the effect of",
            "the importance of",
            "previous work",
            "prior studies",
            "existing research",
            "current understanding"
        ]

        return any(indicator in text_lower for indicator in background_indicators)

    def _generate_id(self) -> str:
        """Generate unique entity ID"""
        short_uuid = str(uuid.uuid4())[:8]
        return f"fact_{short_uuid}"

    def extract_entities_only(self, text: str) -> List[str]:
        """
        Extract only named entities (for statistics)

        Args:
            text: Text to extract from

        Returns:
            List of entity texts
        """
        self._load_model()
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def get_entity_types(self, text: str) -> Counter:
        """
        Get count of entity types in text

        Args:
            text: Text to analyze

        Returns:
            Counter of entity type labels
        """
        self._load_model()
        doc = self.nlp(text)
        return Counter(ent.label_ for ent in doc.ents)
