"""LLM-guided token extractor for context-aware entity extraction"""

import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple

from src.models import Entity, EntityType
from src.llm_adapters.factory import get_llm_adapter


class LLMGuidedTokenExtractor:
    """
    Three-stage token-based extraction approach:
    1. Extract facts from Abstract + Introduction using LLM
    2. Generate context-aware tokens/terms based on facts using LLM
    3. Apply token matching to extract hypotheses (FREE)

    Token-based matching advantages:
    - Simpler than regex (no validation needed)
    - Faster execution (simple substring matching)
    - More flexible scoring (token combinations)
    - Easier for LLM to generate

    Target cost: ~$0.01-0.015/paper (only Abstract+Intro processed by LLM)
    """

    # Stage 1: Extract facts
    FACT_EXTRACTION_PROMPT = """Extract key scientific FACTS from this text.
A fact is an established piece of knowledge or observation that serves as input to the research.

Text:
---
{text}
---

Return ONLY valid JSON array with facts:
[
  {{"text": "fact text", "confidence": 0.95}},
  {{"text": "another fact", "confidence": 0.90}}
]

Focus on:
- Prior research findings
- Known biological/physical mechanisms
- Established theories
- Observed phenomena

If no facts found, return: []"""

    # Stage 2: Generate key tokens
    TOKEN_GENERATION_PROMPT = """Based on these extracted FACTS, generate key TOKENS/TERMS to find related HYPOTHESES.

Facts found:
{facts_list}

A hypothesis is a testable prediction that builds on these facts.

Generate 10-20 key tokens that are likely to appear in hypothesis statements:

**Required tokens** (domain-specific terms from facts):
- Key concepts, processes, molecules, proteins, genes
- Specific entities mentioned in facts
- Domain terminology

**Optional tokens** (hypothesis indicators):
- Hypothesis verbs: hypothesize, predict, propose, suggest, expect, anticipate, postulate
- Investigation terms: test, investigate, examine, assess, evaluate
- Aim/objective markers: aim, objective, goal, purpose

Return ONLY valid JSON:
{{
  "required_tokens": ["term1", "term2", "term3", ...],
  "optional_tokens": ["hypothesize", "predict", "propose", ...],
  "reasoning": "brief explanation of token selection strategy"
}}

Make tokens specific to the facts but flexible. Use lowercase.
Example:
- If facts mention "cellular senescence" → required: ["senescence", "senescent", "cell"]
- If facts mention "mitochondrial dysfunction" → required: ["mitochondria", "mitochondrial"]

If facts are too generic, focus on general hypothesis indicators in optional_tokens."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        temperature: float = 0.2,
        max_tokens: int = 800,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize LLM-guided token extractor

        Args:
            provider: LLM provider (openai, ollama)
            model: Model name
            temperature: LLM temperature
            max_tokens: Max tokens per request
            confidence_threshold: Minimum confidence for extracted entities
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold

        # Initialize LLM adapter
        self.llm_adapter = get_llm_adapter(provider)

        # Track metrics
        self.total_cost = 0.0
        self.total_tokens = 0
        self.generated_tokens_cache: Dict[str, Dict[str, List[str]]] = {}

    def extract_facts_and_hypotheses(
        self,
        abstract_text: str,
        intro_text: str,
        paper_id: str
    ) -> Tuple[List[Entity], List[Entity], Dict[str, List[str]]]:
        """
        Three-stage extraction: facts + LLM-guided token generation + token matching

        Args:
            abstract_text: Abstract section text
            intro_text: Introduction section text
            paper_id: Paper identifier

        Returns:
            Tuple of (facts, hypotheses, generated_tokens_dict)
        """
        # Combine abstract and intro
        combined_text = f"{abstract_text}\n\n{intro_text}".strip()

        if not combined_text:
            return [], [], {}

        # ========== STAGE 1: Extract Facts ==========
        facts = self._extract_facts_with_llm(combined_text)

        if not facts:
            print("Warning: No facts extracted. Using fallback tokens.")
            # Fallback to generic tokens
            hypothesis_tokens = self._get_fallback_tokens()
        else:
            # ========== STAGE 2: Generate Context-Aware Tokens ==========
            hypothesis_tokens = self._generate_tokens(facts)

        # ========== STAGE 3: Apply Token Matching ==========
        hypotheses = self._extract_hypotheses_with_tokens(
            intro_text,
            hypothesis_tokens
        )

        return facts, hypotheses, hypothesis_tokens

    def _extract_facts_with_llm(self, text: str) -> List[Entity]:
        """
        Extract facts using LLM (Stage 1)

        Args:
            text: Text to extract facts from

        Returns:
            List of fact entities
        """
        # Truncate if too long (cost optimization)
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Text truncated for cost optimization]"

        prompt = self.FACT_EXTRACTION_PROMPT.format(text=text)

        try:
            response = self.llm_adapter.generate(
                prompt=prompt,
                system_prompt="You are a precise scientific information extractor. Return only valid JSON.",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response["content"]
            tokens_used = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            cost = response["cost"]

            # Update metrics
            self.total_tokens += tokens_used
            self.total_cost += cost

            print(f"[Stage 1] Fact extraction cost: ${cost:.4f}, tokens: {tokens_used}")

        except Exception as e:
            print(f"Error in fact extraction: {e}")
            return []

        # Parse JSON
        try:
            fact_data = self._parse_json_response(content)
        except Exception as e:
            print(f"Error parsing fact extraction response: {e}")
            return []

        # Convert to Entity objects
        entities = []
        for item in fact_data:
            entity = Entity(
                id=self._generate_id(EntityType.FACT),
                type=EntityType.FACT,
                text=item["text"],
                confidence=item.get("confidence", 0.85),
                source_section="abstract+introduction",
                metadata={
                    "extraction_method": "llm_guided_stage1",
                    "llm_model": self.model,
                    "llm_cost": cost / max(len(fact_data), 1)
                }
            )
            entities.append(entity)

        return entities

    def _generate_tokens(self, facts: List[Entity]) -> Dict[str, List[str]]:
        """
        Generate context-aware tokens based on extracted facts (Stage 2)

        Args:
            facts: List of extracted fact entities

        Returns:
            Dictionary with "required_tokens" and "optional_tokens" lists
        """
        # Format facts for prompt
        facts_list = "\n".join([f"- {fact.text}" for fact in facts[:10]])  # Limit to 10 facts

        # Check cache
        cache_key = hash(facts_list)
        if cache_key in self.generated_tokens_cache:
            print("[Stage 2] Using cached tokens")
            return self.generated_tokens_cache[cache_key]

        prompt = self.TOKEN_GENERATION_PROMPT.format(facts_list=facts_list)

        try:
            response = self.llm_adapter.generate(
                prompt=prompt,
                system_prompt="You are a scientific terminology expert. Generate context-specific tokens.",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response["content"]
            tokens_used = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            cost = response["cost"]

            # Update metrics
            self.total_tokens += tokens_used
            self.total_cost += cost

            print(f"[Stage 2] Token generation cost: ${cost:.4f}, tokens: {tokens_used}")

        except Exception as e:
            print(f"Error in token generation: {e}")
            return self._get_fallback_tokens()

        # Parse JSON
        try:
            data = self._parse_json_response(content)
            required_tokens = data.get("required_tokens", [])
            optional_tokens = data.get("optional_tokens", [])
            reasoning = data.get("reasoning", "")

            if reasoning:
                print(f"[Stage 2] Token strategy: {reasoning}")

            # Normalize tokens to lowercase
            required_tokens = [t.lower() for t in required_tokens if t]
            optional_tokens = [t.lower() for t in optional_tokens if t]

            if not required_tokens and not optional_tokens:
                print("Warning: No tokens generated. Using fallback.")
                return self._get_fallback_tokens()

            tokens_dict = {
                "required_tokens": required_tokens,
                "optional_tokens": optional_tokens
            }

            # Cache tokens
            self.generated_tokens_cache[cache_key] = tokens_dict

            print(f"[Stage 2] Generated {len(required_tokens)} required + {len(optional_tokens)} optional tokens")

            return tokens_dict

        except Exception as e:
            print(f"Error parsing token generation response: {e}")
            return self._get_fallback_tokens()

    def _extract_hypotheses_with_tokens(
        self,
        text: str,
        tokens_dict: Dict[str, List[str]]
    ) -> List[Entity]:
        """
        Extract hypotheses using token matching (Stage 3 - FREE)

        Matching logic:
        - ≥2 required tokens → extract (confidence 0.75)
        - ≥1 required + ≥2 optional → extract (confidence 0.85)
        - ≥3 optional tokens → extract (confidence 0.70)

        Args:
            text: Text to extract from
            tokens_dict: Dictionary with "required_tokens" and "optional_tokens"

        Returns:
            List of hypothesis entities
        """
        entities = []

        required_tokens = tokens_dict.get("required_tokens", [])
        optional_tokens = tokens_dict.get("optional_tokens", [])

        # Split text into sentences
        sentences = self._split_into_sentences(text)

        # Match tokens
        seen_texts = set()  # Deduplicate
        for sent_text in sentences:
            sent_text = sent_text.strip().replace("\n", " ")
            if not sent_text or len(sent_text) < 10:
                continue

            # Normalize sentence for matching
            sent_lower = sent_text.lower()

            # Count token matches
            matched_required = []
            matched_optional = []

            for token in required_tokens:
                if token in sent_lower:
                    matched_required.append(token)

            for token in optional_tokens:
                if token in sent_lower:
                    matched_optional.append(token)

            # Determine if sentence should be extracted based on token combinations
            should_extract = False
            confidence = 0.0

            # Rule 1: ≥2 required tokens
            if len(matched_required) >= 2:
                should_extract = True
                confidence = 0.75 + min(0.15, (len(matched_required) - 2) * 0.05)

            # Rule 2: ≥1 required + ≥2 optional (BEST match)
            elif len(matched_required) >= 1 and len(matched_optional) >= 2:
                should_extract = True
                confidence = 0.85 + min(0.1, (len(matched_optional) - 2) * 0.03)

            # Rule 3: ≥3 optional tokens (fallback)
            elif len(matched_optional) >= 3:
                should_extract = True
                confidence = 0.70 + min(0.15, (len(matched_optional) - 3) * 0.04)

            if should_extract and confidence >= self.confidence_threshold:
                # Avoid duplicates
                if sent_lower in seen_texts:
                    continue
                seen_texts.add(sent_lower)

                entity = Entity(
                    id=self._generate_id(EntityType.HYPOTHESIS),
                    type=EntityType.HYPOTHESIS,
                    text=sent_text,
                    confidence=round(confidence, 2),
                    source_section="introduction",
                    metadata={
                        "extraction_method": "llm_guided_tokens",
                        "matched_required_tokens": matched_required[:5],  # Store up to 5
                        "matched_optional_tokens": matched_optional[:5],
                        "llm_cost": 0.0  # FREE extraction!
                    }
                )
                entities.append(entity)

        print(f"[Stage 3] Extracted {len(entities)} hypotheses using token matching (FREE)")

        return entities

    def _get_fallback_tokens(self) -> Dict[str, List[str]]:
        """Get fallback tokens if LLM generation fails"""
        return {
            "required_tokens": [],  # No domain-specific tokens without context
            "optional_tokens": [
                # Hypothesis indicators
                "hypothesize", "hypothesis", "predict", "prediction",
                "propose", "proposed", "expect", "expected",
                "anticipate", "postulate", "suggest", "suggested",
                # Investigation terms
                "test", "investigate", "examine", "assess", "evaluate",
                # Aim/objective markers
                "aim", "objective", "goal", "purpose",
                # Question words for research questions
                "whether", "if", "how", "why", "what"
            ]
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _parse_json_response(self, content: str) -> Any:
        """Parse JSON from LLM response"""
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'[\[{].*[\]}]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from response")

        return data

    def _generate_id(self, entity_type: EntityType) -> str:
        """Generate unique entity ID"""
        short_uuid = str(uuid.uuid4())[:8]
        return f"{entity_type.value}_{short_uuid}"

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get extraction metrics

        Returns:
            Dictionary with cost and token metrics
        """
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "token_sets_generated": len(self.generated_tokens_cache)
        }

    def reset_metrics(self):
        """Reset cost tracking metrics"""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.generated_tokens_cache.clear()
