"""Pattern-based extractors using regex (from extractor.ipynb)"""

import re
import uuid
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.models import Entity, EntityType


class BasePatternExtractor(ABC):
    """Base class for pattern-based extractors"""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize extractor

        Args:
            confidence_threshold: Minimum confidence for extraction
        """
        self.confidence_threshold = confidence_threshold
        self.patterns = self._get_patterns()
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        self.dynamic_keywords = []  # LLM-generated keywords
        self.dynamic_patterns = []  # Compiled patterns from keywords

    @abstractmethod
    def _get_patterns(self) -> List[str]:
        """Get regex patterns for this extractor"""
        pass

    @abstractmethod
    def _get_entity_type(self) -> EntityType:
        """Get entity type for this extractor"""
        pass

    def set_dynamic_keywords(self, keywords: List[str]):
        """
        Set dynamic keywords from LLM for context-specific extraction

        Args:
            keywords: List of keywords/phrases to use for extraction
        """
        self.dynamic_keywords = keywords

        # Generate regex patterns from keywords
        self.dynamic_patterns = []
        for keyword in keywords:
            # Escape special regex characters
            escaped = re.escape(keyword)
            # Create flexible pattern that allows word boundaries
            pattern = rf"\b{escaped}\b"
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.dynamic_patterns.append(compiled)
            except re.error:
                # Skip invalid patterns
                pass

    def get_all_patterns(self):
        """Get combined list of static and dynamic patterns"""
        return self.compiled_patterns + self.dynamic_patterns

    def extract(
        self,
        text: str,
        section_name: Optional[str] = None,
        use_spacy: bool = True
    ) -> List[Entity]:
        """
        Extract entities from text using patterns

        Args:
            text: Text to extract from
            section_name: Name of section (for metadata)
            use_spacy: Whether to use spaCy for sentence splitting

        Returns:
            List of extracted entities
        """
        entities = []

        # Parse text into sentences
        if use_spacy and SPACY_AVAILABLE:
            sentences = self._split_with_spacy(text)
        else:
            sentences = self._split_simple(text)

        # Match patterns (combine static + dynamic)
        all_patterns = self.get_all_patterns()

        for sent_text in sentences:
            sent_text = sent_text.strip().replace("\n", " ")
            if not sent_text:
                continue

            # Check if any pattern matches
            matched = False
            matched_by_dynamic = False
            for pattern in all_patterns:
                if pattern.search(sent_text):
                    matched = True
                    # Check if this was a dynamic pattern
                    if pattern in self.dynamic_patterns:
                        matched_by_dynamic = True
                    break

            if matched:
                # Calculate confidence based on pattern strength
                confidence = self._calculate_confidence(sent_text)

                if confidence >= self.confidence_threshold:
                    entity = Entity(
                        id=self._generate_id(),
                        type=self._get_entity_type(),
                        text=sent_text,
                        confidence=confidence,
                        source_section=section_name,
                        metadata={
                            "extraction_method": "pattern_matching",
                            "used_dynamic_keywords": matched_by_dynamic
                        }
                    )
                    entities.append(entity)

        return entities

    def _split_with_spacy(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        except Exception:
            # Fallback to simple splitting
            return self._split_simple(text)

    def _split_simple(self, text: str) -> List[str]:
        """Simple sentence splitting (fallback)"""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_confidence(self, text: str) -> float:
        """
        Calculate confidence score for extracted entity

        Args:
            text: Extracted text

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence
        confidence = 0.7

        # Boost confidence based on multiple pattern matches (static + dynamic)
        all_patterns = self.get_all_patterns()
        match_count = sum(1 for p in all_patterns if p.search(text))
        if match_count > 1:
            confidence = min(0.95, confidence + 0.1 * (match_count - 1))

        # Reduce confidence for very short sentences
        word_count = len(text.split())
        if word_count < 5:
            confidence *= 0.8
        elif word_count > 50:
            # Very long sentences might be less precise
            confidence *= 0.9

        return round(confidence, 2)

    def _generate_id(self) -> str:
        """Generate unique entity ID"""
        short_uuid = str(uuid.uuid4())[:8]
        entity_type = self._get_entity_type().value
        return f"{entity_type}_{short_uuid}"


class HypothesisExtractor(BasePatternExtractor):
    """Extract hypotheses using patterns (from extractor.ipynb)"""

    def _get_patterns(self) -> List[str]:
        """Hypothesis patterns from extractor.ipynb"""
        return [
            r"(hypothesis|H\d+)[:–-]",
            r"(theor(ies|y)|hypothes[ei]s)\s+(is|was|are|were|that)",
            r"(we|I|authors?|it is|study|theor(ies|y)|hypothes[ei]s)\s+(hypothesi[sz]e|predict|propose|assume|expect|anticipate|foresee|postulate|conjecture|test|suggest)",
            r"(hypothesi[sz]e|predict|propose|assume|expect|anticipate|foresee|postulate|aim|conjecture|test|suggest)(s|d)?\s+that",
            r"(investigate|examine|predict|test)(s|d)?\s+(whether|if)",
            r"(evidence|data)\s+(predict|suggest|propose|assume)",
            r"(aim|objective|goal)\s+(of\s+th(is|e)\s+study)",
            r"if.*then",
            r"it would appear",
            r"it likely that",
            r"According to the (theor(ies|y)|hypothes[ei]s|views?)"
        ]

    def _get_entity_type(self) -> EntityType:
        return EntityType.HYPOTHESIS


class MethodExtractor(BasePatternExtractor):
    """Extract methods/techniques using patterns (from extractor.ipynb)"""

    def _get_patterns(self) -> List[str]:
        """Method patterns from extractor.ipynb"""
        return [
            r"(methods?|methodology|procedures?|techniques?)\s+(is|was|are|were|that|[:–-])",
            r"(we|I|authors?|the study)\s+(used|employed|applied|conducted|performed|carried out|implemented)",
            r"(data\s+(collection|gathering|acquisition)|samples?|participants?|subjects?)\s+(were|was)",
            r"(measurements?|assays?|tests?|analyses?)\s+(were|was)\s+(performed|conducted|carried out|done)",
            r"(statistical|analysis|analytical)\s+(methods?|approach|procedure)",
            r"(in\s+order\s+to\s+(determine|measure|assess|evaluate|test|quantify))",
            r"(equipment|instruments?|software|apparatus|tools?)\s+(was|were|used)",
            r"(protocol|procedure|experimental\s+design)\s+(was|were)",
            r"(collection\s+period|duration|timeframe)\s+(was|of)",
            r"(participants?|subjects?|samples?)\s+(were\s+|was\s+)?(recruited|selected|enrolled|obtained)",
            r"(inclusion|exclusion)\s+(criteria|requirements)",
            r"(we|I|authors?)\s+(measure|perform|conduct|test|quantif(y|ied)|employ|adopt)",
            r"(was|were)\s+(stained|embedded|measured|performed|calculated|obtained|identified|detected|monitored|isolated|stimulated|used|removed|harvested|acqired|gated|tested|considered)"
        ]

    def _get_entity_type(self) -> EntityType:
        return EntityType.TECHNIQUE


class DatasetExtractor(BasePatternExtractor):
    """Extract datasets using patterns (from extractor.ipynb)"""

    def _get_patterns(self) -> List[str]:
        """Dataset patterns from extractor.ipynb"""
        return [
            r"(dataset|data set|database|data source|repository)\s+(was|were|is|are|used|obtained|available|collected|generated|produced)",
            r"(available|accessible|open)\s+(dataset|data set|database|data source|repository)",
            r"(downloaded|retrieved|obtained|accessed)\s+from\s+(the\s+)?(database|repository|server|(web)?site|portal)",
            r"data(\s+were)?\s+(obtained|retrieved|downloaded|sourced)",
            r"(doi|accession\s+number|repository|github|zenodo|figshare|osf|dryad)\s*[:–-]",
            r"(supplementary\s+)?data\s+(available\s+)?(at|in|from|on)",
            r"(gene\s+expression|genomic|proteomic|metabolomic|imaging)\s+dataset",
            r"(data\s+(necessary\s+)?to\s+reproduce|reproducib(le|ility))",
            r"(accession|catalog)\s+(number|code)\s*[:–-]"
        ]

    def _get_entity_type(self) -> EntityType:
        return EntityType.DATASET


class ExperimentExtractor(BasePatternExtractor):
    """Extract experiments using patterns (from extractor.ipynb)"""

    def _get_patterns(self) -> List[str]:
        """Experiment patterns from extractor.ipynb"""
        return [
            r"(experiment|assay|trial|test)\s*(\d+|[A-Z])?(\s*[:–-])",
            r"(test|examine|investigate|evaluate|assess|determine|measure)\s+(the\s+)?(hypothesis|prediction|effect|impact|role|function)",
            r"(performed|conducted|carried out|designed|set up)\s+(an?\s+)?(experiment|assay|trial|test|study)",
            r"experimental\s+(design|setup|procedure|protocol|approach)",
            r"(treatment|condition|group)\s+(was|were)(\s+exposed|\s+treated)",
            r"(control|experimental|treatment)\s+group",
            r"(exposed|subjected|treated|stimulated|challenged)\s+(to|with)",
            r"(intervention|treatment|manipulation)\s+(was|were)",
            r"(validation|confirmatory|confirmation)\s+(experiment|assay|test)",
            r"(parallel|sequential|longitudinal)\s+(experiment|test|measurement)"
        ]

    def _get_entity_type(self) -> EntityType:
        return EntityType.EXPERIMENT


class ResultExtractor(BasePatternExtractor):
    """Extract results using patterns"""

    def _get_patterns(self) -> List[str]:
        """Result patterns"""
        return [
            r"(results?|findings?|observations?)\s+(show|showed|indicate|indicated|reveal|revealed|demonstrate|demonstrated|suggest|suggested)",
            r"(we|I|authors?)\s+(found|observed|discovered|detected|identified|measured)",
            r"(significant|statistically\s+significant)\s+(difference|effect|increase|decrease|correlation|association)",
            r"(p\s*[<>=]\s*0\.\d+|p-value)",
            r"(\d+\.?\d*)\s*%\s+(of|increase|decrease|change)",
            r"(mean|average|median)\s+(\w+\s+)?(was|were|is|are)",
            r"(compared\s+(to|with))\s+(control|baseline)",
            r"(higher|lower|greater|less)\s+(than|in)",
            r"(no\s+significant|significant)\s+(difference|change|effect)"
        ]

    def _get_entity_type(self) -> EntityType:
        return EntityType.RESULT

    def _calculate_confidence(self, text: str) -> float:
        """
        Calculate confidence for results (higher for statistical indicators)

        Args:
            text: Extracted text

        Returns:
            Confidence score
        """
        confidence = super()._calculate_confidence(text)

        # Boost confidence if statistical indicators present
        if re.search(r'p\s*[<>=]\s*0\.\d+', text, re.IGNORECASE):
            confidence = min(0.95, confidence + 0.1)
        if re.search(r'significant', text, re.IGNORECASE):
            confidence = min(0.95, confidence + 0.05)

        return round(confidence, 2)
