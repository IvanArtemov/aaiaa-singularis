"""LLM-based keyword generator for entity extraction"""

import json
from typing import List, Dict, Any, Optional
from functools import lru_cache

from src.models import EntityType
from src.parsers import ParsedDocument
from src.llm_adapters import get_llm_adapter


class EntityKeywordGenerator:
    """
    Generate context-specific keywords for entity extraction using LLM

    Uses title, abstract, and introduction to generate keywords that help
    pattern extractors find relevant entities more accurately.

    Cost: ~$0.001-0.002 per paper (minimal prompt)
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-5-mini",
        cache_size: int = 128
    ):
        """
        Initialize keyword generator

        Args:
            llm_provider: LLM provider ("openai" or "ollama")
            llm_model: Model name (default: gpt-5-mini)
            cache_size: LRU cache size for repeated documents
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_adapter = None  # Lazy initialization

        # Track metrics
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_llm_adapter(self):
        """Lazy initialization of LLM adapter"""
        if self.llm_adapter is None:
            self.llm_adapter = get_llm_adapter(self.llm_provider)
        return self.llm_adapter

    def generate_keywords(
        self,
        parsed_doc: ParsedDocument,
        entity_type: EntityType,
        max_keywords: int = 15
    ) -> List[str]:
        """
        Generate keywords for specific entity type

        Args:
            parsed_doc: Parsed document with sections
            entity_type: Type of entity to generate keywords for
            max_keywords: Maximum number of keywords to generate

        Returns:
            List of keywords/phrases for pattern matching
        """
        # Extract context from document
        title = parsed_doc.title or parsed_doc.metadata.get("title", "Unknown")
        abstract = parsed_doc.get_section("abstract") or ""
        intro = parsed_doc.get_section("introduction") or ""

        # Limit intro to first 500 words to save tokens
        intro_words = intro.split()[:500]
        intro_limited = " ".join(intro_words)

        # Create cache key
        cache_key = self._create_cache_key(title, abstract, intro_limited, entity_type)

        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            return cached_result

        self.cache_misses += 1

        # Generate keywords via LLM
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(
            title=title,
            abstract=abstract,
            intro=intro_limited,
            entity_type=entity_type,
            max_keywords=max_keywords
        )

        try:
            llm = self._get_llm_adapter()
            response = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=300
            )

            # Track metrics
            self.total_tokens += response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            self.total_cost += response["cost"]

            # Parse keywords from response
            keywords = self._parse_keywords(response["content"])

            # Cache result
            self._save_to_cache(cache_key, keywords)

            return keywords

        except Exception as e:
            print(f"Warning: Keyword generation failed: {e}")
            # Return empty list on error - extractors will use default patterns
            return []

    def generate_all_keywords(
        self,
        parsed_doc: ParsedDocument
    ) -> Dict[EntityType, List[str]]:
        """
        Generate keywords for all entity types

        Args:
            parsed_doc: Parsed document

        Returns:
            Dictionary mapping entity types to keywords
        """
        keywords_by_type = {}

        # Generate keywords for each entity type
        entity_types = [
            EntityType.FACT,
            EntityType.HYPOTHESIS,
            EntityType.EXPERIMENT,
            EntityType.METHOD,
            EntityType.RESULT,
            EntityType.DATASET,
            EntityType.ANALYSIS,
            EntityType.CONCLUSION
        ]

        for entity_type in entity_types:
            keywords = self.generate_keywords(parsed_doc, entity_type)
            if keywords:
                keywords_by_type[entity_type] = keywords

        return keywords_by_type

    def _get_system_prompt(self) -> str:
        """Get system prompt for keyword generation"""
        return (
            "You are a keyword extraction expert for scientific papers. "
            "Your task is to generate specific keywords and phrases that help identify "
            "specific types of information in scientific texts. "
            "Return ONLY a JSON array of keywords, no explanations."
        )

    def _build_user_prompt(
        self,
        title: str,
        abstract: str,
        intro: str,
        entity_type: EntityType,
        max_keywords: int
    ) -> str:
        """Build user prompt for keyword generation"""

        # Entity-specific instructions
        entity_instructions = self._get_entity_instructions(entity_type)

        prompt = f"""Given this scientific paper context:

TITLE: {title}

ABSTRACT: {abstract[:500]}

INTRODUCTION (excerpt): {intro[:500]}

Generate {max_keywords} specific keywords or short phrases that would help identify **{entity_type.value.upper()}** in this paper.

{entity_instructions}

Return ONLY a JSON array of strings:
["keyword1", "keyword2", "keyword3", ...]

Keywords should be:
- Specific to this paper's domain and topic
- 3-6 words each
- Useful for pattern matching
- Domain-specific terminology when applicable"""

        return prompt

    def _get_entity_instructions(self, entity_type: EntityType) -> str:
        """Get specific instructions for each entity type"""
        instructions = {
            EntityType.FACT: (
                "Facts are established knowledge, prior findings, or background information. "
                "Look for: citations, known mechanisms, established theories, prior studies."
            ),
            EntityType.HYPOTHESIS: (
                "Hypotheses are testable predictions or proposed explanations. "
                "Look for: predictions, assumptions, theoretical propositions, research questions."
            ),
            EntityType.EXPERIMENT: (
                "Experiments are procedures to test hypotheses. "
                "Look for: experimental designs, test protocols, trial setups, controlled studies."
            ),
            EntityType.METHOD: (
                "Techniques are methods, tools, or procedures used. "
                "Look for: analytical methods, instruments, software, protocols, assays."
            ),
            EntityType.RESULT: (
                "Results are findings, measurements, or observations. "
                "Look for: statistical outcomes, measurements, observed effects, data points."
            ),
            EntityType.DATASET: (
                "Datasets are collections of data used or produced. "
                "Look for: databases, repositories, data sources, sample collections."
            ),
            EntityType.ANALYSIS: (
                "Analyses are statistical or computational processing of results. "
                "Look for: statistical tests, data analysis methods, computational approaches."
            ),
            EntityType.CONCLUSION: (
                "Conclusions are interpretations and implications of findings. "
                "Look for: implications, significance, interpretations, future directions."
            )
        }
        return instructions.get(entity_type, "")

    def _parse_keywords(self, llm_response: str) -> List[str]:
        """
        Parse keywords from LLM response

        Args:
            llm_response: Raw LLM response

        Returns:
            List of extracted keywords
        """
        try:
            # Try to extract JSON array from response
            # Handle cases where LLM adds explanation text
            content = llm_response.strip()

            # Find JSON array in response
            start_idx = content.find('[')
            end_idx = content.rfind(']')

            if start_idx == -1 or end_idx == -1:
                print(f"Warning: No JSON array found in response: {content[:100]}")
                return []

            json_str = content[start_idx:end_idx+1]
            keywords = json.loads(json_str)

            # Validate it's a list of strings
            if not isinstance(keywords, list):
                print(f"Warning: Expected list, got {type(keywords)}")
                return []

            # Filter and clean keywords
            cleaned = []
            for kw in keywords:
                if isinstance(kw, str):
                    kw_cleaned = kw.strip()
                    if kw_cleaned and len(kw_cleaned.split()) <= 5:  # Max 5 words
                        cleaned.append(kw_cleaned)

            return cleaned

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")
            print(f"Response: {llm_response[:200]}")
            return []
        except Exception as e:
            print(f"Warning: Unexpected error parsing keywords: {e}")
            return []

    def _create_cache_key(
        self,
        title: str,
        abstract: str,
        intro: str,
        entity_type: EntityType
    ) -> str:
        """Create cache key from document content"""
        # Use hash of concatenated content + entity type
        content = f"{title}|{abstract[:200]}|{intro[:200]}|{entity_type.value}"
        return str(hash(content))

    def _get_from_cache(self, cache_key: str) -> Optional[List[str]]:
        """Get keywords from cache"""
        # Simple dict-based cache (could use functools.lru_cache or redis)
        if not hasattr(self, '_cache'):
            self._cache = {}
        return self._cache.get(cache_key)

    def _save_to_cache(self, cache_key: str, keywords: List[str]):
        """Save keywords to cache"""
        if not hasattr(self, '_cache'):
            self._cache = {}

        # Limit cache size to prevent memory issues
        if len(self._cache) >= 128:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[cache_key] = keywords

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for keyword generation"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(
                self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                2
            )
        }

    def reset_metrics(self):
        """Reset metrics counters"""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
