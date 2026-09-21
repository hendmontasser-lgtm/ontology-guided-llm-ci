"""
Evaluation metrics for the ontology-guided LLM extraction paper.

Metrics (matching the Variables table in the paper):
  DV1  Precision
  DV2  Recall
  DV3  F1-score
  DV4  Hallucination Rate
  DV5  Knowledge Graph Quality
         → Ontological Consistency Rate
         → Graph Completeness
         → Redundancy Rate
"""

import re
from ontology import validate_triple


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r"['''\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[,.\-–—]$", "", text)
    return text.strip()


def _triple_key(t: dict) -> tuple:
    """Canonical key for a triple used in set comparisons."""
    return (
        _norm(t.get("head", "")),
        _norm(t.get("relation", "")),
        _norm(t.get("tail", "")),
    )


# ── Core matching ─────────────────────────────────────────────────────────────

def match_triples(
    predicted: list[dict],
    gold: list[dict],
) -> tuple[set, set, set]:
    """
    Return (true_positives, false_positives, false_negatives) as sets of keys.
    Matching is on normalised (head, relation, tail) — types are NOT compared
    so that a correct relation extraction is not penalised for minor type errors.
    """
    pred_keys = {_triple_key(t) for t in predicted}
    gold_keys = {_triple_key(t) for t in gold}
    tp = pred_keys & gold_keys
    fp = pred_keys - gold_keys
    fn = gold_keys - pred_keys
    return tp, fp, fn


# ── Per-sentence metrics ──────────────────────────────────────────────────────

def score_sentence(predicted: list[dict], gold: list[dict]) -> dict:
    tp, fp, fn = match_triples(predicted, gold)
    n_pred = len(predicted)
    n_gold = len(gold)

    precision = len(tp) / n_pred if n_pred > 0 else 0.0
    recall    = len(tp) / n_gold if n_gold > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Hallucination: predicted triple whose head OR tail does not appear
    # in the sentence text (surface-level grounding check)
    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "n_pred": n_pred,
        "n_gold": n_gold,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def hallucination_rate(predicted: list[dict], sentence: str) -> float:
    """
    Proportion of predicted triples where head or tail is not found
    in the source sentence (surface grounding check).
    """
    if not predicted:
        return 0.0
    sent_norm = _norm(sentence)
    hallucinated = 0
    for t in predicted:
        head_in = _norm(t.get("head", "")) in sent_norm
        tail_in = _norm(t.get("tail", "")) in sent_norm
        if not head_in or not tail_in:
            hallucinated += 1
    return hallucinated / len(predicted)


# ── KG-level quality metrics ─────────────────────────────────────────────────

def ontological_consistency_rate(triples: list[dict]) -> float:
    """
    DV5a: proportion of triples that satisfy all ontology constraints.
    """
    if not triples:
        return 1.0   # vacuously consistent
    valid = sum(1 for t in triples if validate_triple(t)[0])
    return valid / len(triples)


def redundancy_rate(triples: list[dict]) -> float:
    """
    DV5c: proportion of triples that are duplicates or near-duplicates.
    """
    if not triples:
        return 0.0
    keys = [_triple_key(t) for t in triples]
    unique = len(set(keys))
    duplicates = len(keys) - unique
    return duplicates / len(keys)


# ── Aggregate across all sentences ───────────────────────────────────────────

def aggregate_scores(results: list[dict]) -> dict:
    """
    Compute macro-averaged metrics across all sentence-level results.
    Each entry in results must have keys from score_sentence() plus
    'hallucination_rate', 'ontological_consistency', 'redundancy_rate',
    'sentence', 'triples'.
    """
    n = len(results)
    if n == 0:
        return {}

    # Micro-averaged P/R/F1 (sum TP, FP, FN across all sentences)
    total_tp   = sum(r["tp"]     for r in results)
    total_pred = sum(r["n_pred"] for r in results)
    total_gold = sum(r["n_gold"] for r in results)

    micro_p  = total_tp / total_pred if total_pred > 0 else 0.0
    micro_r  = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    # Macro-averaged (mean of per-sentence scores)
    macro_p  = sum(r["precision"] for r in results) / n
    macro_r  = sum(r["recall"]    for r in results) / n
    macro_f1 = sum(r["f1"]        for r in results) / n

    # DV4: mean hallucination rate
    mean_hall = sum(r["hallucination_rate"]       for r in results) / n

    # DV5 sub-metrics
    mean_consist = sum(r["ontological_consistency"] for r in results) / n
    mean_redund  = sum(r["redundancy_rate"]          for r in results) / n

    # DV5b Graph Completeness = micro recall (proportion of gold triples found)
    completeness = micro_r

    return {
        # Micro
        "micro_precision":  round(micro_p,  4),
        "micro_recall":     round(micro_r,  4),
        "micro_f1":         round(micro_f1, 4),
        # Macro
        "macro_precision":  round(macro_p,  4),
        "macro_recall":     round(macro_r,  4),
        "macro_f1":         round(macro_f1, 4),
        # DV4
        "hallucination_rate": round(mean_hall,    4),
        # DV5
        "ontological_consistency_rate": round(mean_consist,  4),
        "graph_completeness":           round(completeness,  4),
        "redundancy_rate":              round(mean_redund,   4),
        # Counts
        "total_tp":   total_tp,
        "total_pred": total_pred,
        "total_gold": total_gold,
        "n_sentences": n,
    }
