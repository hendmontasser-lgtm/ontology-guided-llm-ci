"""
Prompt templates — v2
Softer Model B prompt with broader class definitions,
direction-explicit develops examples, and FIRE-style sector labels.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ontology import ONTOLOGY_PROMPT_BLOCK

OUTPUT_FORMAT = """
Return ONLY a JSON array. Each element must have exactly these five keys:
  {
    "head":      "<entity text as it appears in the sentence>",
    "head_type": "<one of: Company Product Technology Market Trend Opportunity Threat>",
    "relation":  "<one of: develops operatesIn competesWith createsOpportunity createsThreat>",
    "tail":      "<entity text as it appears in the sentence>",
    "tail_type": "<one of: Company Product Technology Market Trend Opportunity Threat>"
  }

If no relations can be extracted, return: []
Return ONLY the JSON array — no explanation, no markdown fences.
""".strip()


# ── MODEL A: STANDALONE LLM ───────────────────────────────────────────────────
MODEL_A_SYSTEM = """You are an expert information extraction system.
Extract named entities and business relationships from financial and
business text including SEC filings, market intelligence reports, and news.
Only extract relationships explicitly stated or clearly implied in the text.
Do not hallucinate entities or relationships."""

def model_a_user_prompt(sentence: str) -> str:
    return f"""Extract all business entities and relationships from the sentence below.

Use these relationship types where they apply:
  - develops          : a company creates/manufactures a product or technology
  - operatesIn        : a company operates in a market sector or industry
  - competesWith      : a company or product competes with another
  - createsOpportunity: a trend/technology creates a business opportunity
  - createsThreat     : a trend/technology/company poses a competitive threat

For entity types, use: Company, Product, Technology, Market, Trend, Opportunity, Threat

{OUTPUT_FORMAT}

Sentence:
"{sentence}"
"""


# ── MODEL B: ONTOLOGY-GUIDED LLM ─────────────────────────────────────────────
MODEL_B_SYSTEM = """You are an expert information extraction system operating under
strict ontological constraints for competitive intelligence knowledge graph construction.

You extract triples ONLY in the form permitted by the ontology schema below.
Accuracy and ontological validity are more important than recall:
it is better to extract nothing than to extract a wrong triple.
When in doubt about a triple, omit it."""

def model_b_user_prompt(sentence: str) -> str:
    return f"""{ONTOLOGY_PROMPT_BLOCK}

{OUTPUT_FORMAT}

Extract all valid triples from the sentence below that conform to the ontology.
Remember:
  - For develops: Company is ALWAYS the head, Product/Technology is ALWAYS the tail.
  - For operatesIn: brief sector labels like "auto", "pharma", "entertainment" are valid Market entities.
  - Discard any triple whose types or relation violate the ontology constraints above.

Sentence:
"{sentence}"
"""
