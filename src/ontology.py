"""
Competitive Intelligence Ontology — v2
Adds type-alias normalisation so the validation pass handles
FIRE's annotation conventions (Sector → Market, reversed Productof direction).
"""

from typing import Tuple, List, Dict

# Canonical entity classes
CLASSES = {
    "Company", "Product", "Technology",
    "Market", "Opportunity", "Threat", "Trend",
}

# Type aliases: model may return these → map to canonical class
TYPE_ALIASES: Dict[str, str] = {
    # Market aliases (FIRE uses "Sector")
    "Sector":       "Market",
    "Industry":     "Market",
    "Segment":      "Market",
    "sector":       "Market",
    "industry":     "Market",
    # Company aliases
    "Organization": "Company",
    "Organisation": "Company",
    "Corporation":  "Company",
    "Firm":         "Company",
    "organization": "Company",
    # Product aliases
    "Service":      "Product",
    "Platform":     "Product",
    "Software":     "Product",
    "Application":  "Product",
    "service":      "Product",
    # Technology aliases
    "System":       "Technology",
    "Framework":    "Technology",
    "Algorithm":    "Technology",
}

def normalise_type(raw: str) -> str:
    """Map alias → canonical class. Returns raw string if already canonical."""
    return TYPE_ALIASES.get(raw, raw)


RELATIONS = {
    "develops": {
        "head": {"Company"},
        "tail": {"Product", "Technology"},
    },
    "operatesIn": {
        "head": {"Company"},
        "tail": {"Market"},
    },
    "competesWith": {
        "head": {"Company", "Product"},
        "tail": {"Company", "Product"},
    },
    "createsOpportunity": {
        "head": {"Technology", "Market", "Trend", "Company"},
        "tail": {"Company", "Market", "Product"},
    },
    "createsThreat": {
        "head": {"Technology", "Market", "Trend", "Company", "Product"},
        "tail": {"Company", "Market", "Product"},
    },
}

# Prompt block injected into Model B system prompt
ONTOLOGY_PROMPT_BLOCK = """
ONTOLOGY SCHEMA — follow this exactly:

ENTITY CLASSES:
  - Company     : any named organisation, corporation, or firm
                  (e.g. "Apple", "TSMC", "Daimler Benz")
  - Product     : a named product, software, platform, service, or manufactured item
                  (e.g. "iPhone", "Azure", "electric cars", "converge polyols")
  - Technology  : a technology, method, or technical development
                  (e.g. "generative AI", "CRISPR", "5G networks")
  - Market      : an industry sector, market segment, or brief sector label —
                  including informal abbreviations used in financial text
                  (e.g. "auto", "drug-maker", "cloud computing", "renewable energy",
                   "entertainment", "mobile communications", "electric vehicles")
  - Trend       : a social, regulatory, or macroeconomic trend
                  (e.g. "aging demographics", "remote work adoption")
  - Opportunity : an explicitly labelled business opportunity
  - Threat      : an explicitly labelled business threat

RELATIONS (domain → range):
  - develops(Company → Product|Technology)
      The COMPANY is the head; the PRODUCT or TECHNOLOGY is the tail.
      Use when a company created, manufactured, designed, or engineered something.
      Example: head="BYD" head_type="Company" relation="develops"
               tail="electric cars" tail_type="Product"

  - operatesIn(Company → Market)
      Use when a company belongs to, operates in, or is described by a sector label.
      The sector/industry label is the tail — even brief ones like "auto" or "pharma".
      Example: head="Chrysler" head_type="Company" relation="operatesIn"
               tail="auto" tail_type="Market"

  - competesWith(Company|Product → Company|Product)
      Direct competition between two entities in the same market.
      Example: head="Spotify" head_type="Company" relation="competesWith"
               tail="Apple Music" tail_type="Product"

  - createsOpportunity(Technology|Market|Trend|Company → Company|Market|Product)
      A force, trend, or entity that opens a business opportunity for another.

  - createsThreat(Technology|Market|Trend|Company|Product → Company|Market|Product)
      A force, trend, or entity that poses a strategic threat to another.

EXTRACTION RULES:
  1. Extract only relations explicitly stated or clearly implied in the sentence.
  2. Both head and tail must be text spans present in the sentence.
  3. Relation name must be exactly one of the five above.
  4. Entity type must be exactly one of the seven classes above.
  5. For develops: the COMPANY is always the head, the PRODUCT is always the tail.
  6. For operatesIn: brief sector labels ("auto", "pharma") are valid Market entities.
  7. If uncertain about a triple, do NOT include it.
""".strip()


def validate_triple(triple: Dict) -> Tuple[bool, str]:
    """
    Validate a triple against ontology constraints.
    Applies type-alias normalisation before checking.
    Returns (is_valid, reason_if_invalid).
    """
    relation  = str(triple.get("relation",  "")).strip()
    head_type = normalise_type(str(triple.get("head_type", "")).strip())
    tail_type = normalise_type(str(triple.get("tail_type", "")).strip())
    head_text = str(triple.get("head", "")).strip()
    tail_text = str(triple.get("tail", "")).strip()

    if not head_text or not tail_text:
        return False, "Empty head or tail text"

    if relation not in RELATIONS:
        return False, f"Unknown relation '{relation}'"

    if head_type not in CLASSES:
        return False, f"Unrecognised head type '{head_type}'"

    if tail_type not in CLASSES:
        return False, f"Unrecognised tail type '{tail_type}'"

    allowed_heads = RELATIONS[relation]["head"]
    allowed_tails = RELATIONS[relation]["tail"]

    if head_type not in allowed_heads:
        return False, (
            f"'{head_type}' is not a valid head type for '{relation}'. "
            f"Expected: {sorted(allowed_heads)}"
        )

    if tail_type not in allowed_tails:
        return False, (
            f"'{tail_type}' is not a valid tail type for '{relation}'. "
            f"Expected: {sorted(allowed_tails)}"
        )

    return True, ""


def filter_triples(triples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split triples into (valid, rejected) after ontology validation."""
    valid, rejected = [], []
    for t in triples:
        # Normalise types in-place before validation
        t = {**t,
             "head_type": normalise_type(str(t.get("head_type","")).strip()),
             "tail_type": normalise_type(str(t.get("tail_type","")).strip())}
        ok, reason = validate_triple(t)
        if ok:
            valid.append(t)
        else:
            rejected.append({**t, "_rejection_reason": reason})
    return valid, rejected
