"""
Main evaluation runner.

Usage:
    # Default: Anthropic / Claude Haiku 4.5
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_evaluation.py --track custom

    # OpenAI / GPT-4o-mini
    export OPENAI_API_KEY=sk-...
    python run_evaluation.py --track custom --provider openai

    # Custom model
    python run_evaluation.py --provider openai --model gpt-4o

    # All flags
    python run_evaluation.py [--provider anthropic|openai] [--model NAME]
                             [--dry-run] [--max-sentences N]
                             [--track fire|custom|all] [--delay 0.3]

Outputs (suffix encodes provider/model so different runs don't overwrite):
    results/model_a_raw_<tag>.json      — Model A raw extractions
    results/model_b_raw_<tag>.json      — Model B raw extractions
    results/scores_summary_<tag>.json   — aggregated metrics
"""

import os, sys, json, time, csv, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extractor import extract_model_a, extract_model_b, set_provider, set_model, get_provider, get_model
from scorer   import score_sentence, hallucination_rate, \
                     ontological_consistency_rate, redundancy_rate, \
                     aggregate_scores

# ── Paths ─────────────────────────────────────────────────────────────────────
# Repository layout:
#   <repo>/src/run_evaluation.py   <- this file
#   <repo>/data/gold/*.csv         <- gold-standard datasets
#   <repo>/results/                <- output (created if missing)
REPO_ROOT   = Path(__file__).resolve().parent.parent
GOLD_DIR    = REPO_ROOT / "data" / "gold"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIRE_DEV_CSV     = GOLD_DIR / "FIRE_develops_gold.csv"
FIRE_SECTOR_CSV  = GOLD_DIR / "FIRE_operatesIn_gold.csv"
CUSTOM_CSV       = GOLD_DIR / "CI_supplement_gold.csv"


# ── Gold standard loader ──────────────────────────────────────────────────────

def load_gold(track: str) -> list[dict]:
    """
    Returns list of {sentence, gold_triples: [{head, head_type, relation, tail, tail_type}]}
    grouped by sentence (multiple gold triples per sentence merged).
    """
    raw_rows = []

    if track in ("fire", "all"):
        for path, rel_col in [
            (FIRE_DEV_CSV,    "gold_label"),
            (FIRE_SECTOR_CSV, "label"),
        ]:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lbl = row.get(rel_col, "").strip()
                    if lbl not in ("Development", "operatesIn"):
                        continue
                    relation = "develops" if lbl == "Development" else "operatesIn"
                    raw_rows.append({
                        "sentence":  row["sentence"],
                        "head":      row["head_text"],
                        "head_type": row["head_type"],
                        "relation":  relation,
                        "tail":      row["tail_text"],
                        "tail_type": row["tail_type"],
                    })

    if track in ("custom", "all"):
        with open(CUSTOM_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw_rows.append({
                    "sentence":  row["sentence"],
                    "head":      row["head_text"],
                    "head_type": row["head_type"],
                    "relation":  row["relation"],
                    "tail":      row["tail_text"],
                    "tail_type": row["tail_type"],
                })

    # Group by sentence
    from collections import defaultdict
    groups = defaultdict(list)
    for r in raw_rows:
        sent = r.pop("sentence")
        groups[sent].append(r)

    return [{"sentence": s, "gold_triples": ts} for s, ts in groups.items()]


# ── Per-sentence evaluation ───────────────────────────────────────────────────

def evaluate_sentence(item: dict, model: str, dry_run: bool) -> dict:
    sentence    = item["sentence"]
    gold        = item["gold_triples"]

    if dry_run:
        # Return a zero-result placeholder without calling the API
        predicted = []
    else:
        if model == "A":
            out = extract_model_a(sentence)
        else:
            out = extract_model_b(sentence)
        predicted = out["triples"]

    scores = score_sentence(predicted, gold)
    hall   = hallucination_rate(predicted, sentence)
    consist = ontological_consistency_rate(predicted)
    redund  = redundancy_rate(predicted)

    return {
        "sentence":               sentence,
        "gold_triples":           gold,
        "predicted_triples":      predicted,
        "tp": scores["tp"], "fp": scores["fp"], "fn": scores["fn"],
        "n_pred": scores["n_pred"], "n_gold": scores["n_gold"],
        "precision":              scores["precision"],
        "recall":                 scores["recall"],
        "f1":                     scores["f1"],
        "hallucination_rate":     hall,
        "ontological_consistency": consist,
        "redundancy_rate":        redund,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",      default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model",         default=None,
                        help="Model name (default: claude-haiku-4-5-20251001 for anthropic, "
                             "gpt-4o-mini for openai)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Skip API calls; use empty predictions (sanity check)")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Limit to N sentences (useful for quick dev tests)")
    parser.add_argument("--track",         default="all",
                        choices=["fire", "custom", "all"])
    parser.add_argument("--delay",         type=float, default=0.3,
                        help="Seconds to wait between API calls (rate-limit buffer)")
    args = parser.parse_args()

    # Configure extractor for chosen provider/model BEFORE any API calls
    set_provider(args.provider)
    if args.model:
        set_model(args.model)

    # Build a tag for output file naming: provider + model identifier
    model_tag = get_model().replace("/", "_").replace(":", "_")
    file_tag  = f"{get_provider()}_{model_tag}"

    print(f"\n{'='*60}")
    print(f"  Ontology-Guided LLM Extraction — Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"  Provider:     {get_provider()}")
    print(f"  Model:        {get_model()}")
    print(f"  Track:        {args.track}")
    print(f"  Dry run:      {args.dry_run}")
    print(f"  Max sentences:{args.max_sentences or 'all'}")
    print(f"  API delay:    {args.delay}s")
    print()

    gold_items = load_gold(args.track)
    if args.max_sentences:
        gold_items = gold_items[: args.max_sentences]

    total = len(gold_items)
    print(f"  Loaded {total} unique sentences from gold standard\n")

    results_a, results_b = [], []

    for i, item in enumerate(gold_items, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{total}] Processing...")

        # Model A
        res_a = evaluate_sentence(item, "A", args.dry_run)
        results_a.append(res_a)

        if not args.dry_run:
            time.sleep(args.delay)

        # Model B
        res_b = evaluate_sentence(item, "B", args.dry_run)
        results_b.append(res_b)

        if not args.dry_run:
            time.sleep(args.delay)

    # Save raw results — tagged by provider/model so different runs don't collide
    with open(RESULTS_DIR / f"model_a_raw_{file_tag}.json", "w") as f:
        json.dump(results_a, f, indent=1)
    with open(RESULTS_DIR / f"model_b_raw_{file_tag}.json", "w") as f:
        json.dump(results_b, f, indent=1)

    # Aggregate
    agg_a = aggregate_scores(results_a)
    agg_b = aggregate_scores(results_b)

    summary = {
        "provider":               get_provider(),
        "model":                  get_model(),
        "track":                  args.track,
        "Model_A_Standalone":     agg_a,
        "Model_B_OntologyGuided": agg_b,
    }
    summary_path = RESULTS_DIR / f"scores_summary_{file_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty print comparison
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    metrics = [
        ("micro_precision",            "Precision (micro)"),
        ("micro_recall",               "Recall (micro)"),
        ("micro_f1",                   "F1-score (micro)"),
        ("hallucination_rate",         "Hallucination Rate"),
        ("ontological_consistency_rate","Ontological Consistency"),
        ("graph_completeness",         "Graph Completeness"),
        ("redundancy_rate",            "Redundancy Rate"),
    ]
    print(f"  {'Metric':<35} {'Model A':>10} {'Model B':>10}")
    print(f"  {'-'*55}")
    for key, label in metrics:
        a_val = agg_a.get(key, "—")
        b_val = agg_b.get(key, "—")
        print(f"  {label:<35} {a_val:>10.4f} {b_val:>10.4f}")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Summary file:     scores_summary_{file_tag}.json")


if __name__ == "__main__":
    main()
