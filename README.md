# Ontology-Guided LLM Knowledge Extraction for Competitive Intelligence

Replication package for the paper:

> **When Does Ontology Guidance Improve LLM Knowledge Extraction? A Relation-Level Analysis for Competitive Intelligence Knowledge Graphs**
> Hend Montasser, Helwan University, Cairo, Egypt.
> *Under review at Engineering Applications of Artificial Intelligence (EAAI-26-22294).*

This repository contains the domain ontology, extraction pipeline, prompts, and gold-standard
evaluation datasets used in the study, including a newly released hand-annotated dataset for
competitive-intelligence relation extraction.

---

## What this study does

The paper compares two conditions on an identical corpus, using the same LLM in a zero-shot setting:

| Condition | Description |
|---|---|
| **Model A — Standalone LLM** | General-purpose extraction prompt, no schema. All returned triples accepted. |
| **Model B — Ontology-Guided LLM** | Schema-aware prompt plus a deterministic post-hoc validation pass that discards triples violating the ontology's class and domain/range constraints. |

**Headline results** (Claude Haiku 4.5, 723 gold instances across 476 sentences):

| Metric | Model A | Model B | Change |
|---|---|---|---|
| Precision (micro) | 0.1991 | 0.2663 | **+33.8%** (p < 0.001) |
| Recall (micro) | 0.2960 | 0.2822 | −4.7% (n.s.) |
| F1-score (micro) | 0.2380 | 0.2740 | **+15.1%** (p < 0.001) |
| Ontological consistency | 0.9114 | **1.0000** | +9.7% |

The central finding is a **moderation effect**: on the three novel competitive-intelligence
relations, precision nearly doubles (+98.1%) and F1 improves 77.2%, with precision and recall
improving together; on the two established FIRE relations, precision gains (+25.2%) come at a
recall cost. The ontological consistency guarantee (1.000) replicates across three LLMs from two
providers; the performance gains do not.

---

## Repository structure

```
.
├── src/
│   ├── ontology.py          Ontology schema, type aliases, triple validation
│   ├── prompts.py           Model A and Model B prompt templates
│   ├── extractor.py         LLM API calls (Anthropic + OpenAI), JSON parsing
│   ├── scorer.py            Precision, recall, F1, hallucination, KG-quality metrics
│   └── run_evaluation.py    Main evaluation runner (CLI)
├── data/gold/
│   ├── CI_supplement_gold.csv      74 instances — NEW competitive-intelligence dataset
│   ├── FIRE_develops_gold.csv      167 instances — filtered from FIRE Productof
│   └── FIRE_operatesIn_gold.csv    482 instances — from FIRE Sector
├── docs/
│   └── kappa_validation_sample.csv  20% stratified sample used for κ = 0.904
├── results/                 Output directory (created at runtime)
└── requirements.txt
```

---

## The ontology

Seven entity classes and five relations, grounded in Porter's Five Forces (Company, Product,
Technology) and SWOT analysis (Opportunity, Threat), extended with Market and Trend.

| Relation | Domain (head) | Range (tail) |
|---|---|---|
| `develops` | Company | Product, Technology |
| `operatesIn` | Company | Market |
| `competesWith` | Company, Product | Company, Product |
| `createsOpportunity` | Technology, Market, Trend, Company | Company, Market, Product |
| `createsThreat` | Technology, Market, Trend, Company, Product | Company, Market, Product |

A type-alias normalisation layer maps common LLM output variants (`Sector → Market`,
`Organization → Company`, `Service → Product`) to canonical class names before validation.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set an API key for whichever provider you want to use
export ANTHROPIC_API_KEY=sk-ant-...     # for Claude
export OPENAI_API_KEY=sk-...            # for GPT

# 3. Verify the setup without spending any API credits
python src/run_evaluation.py --dry-run --track all

# 4. Run the small custom track first (74 instances, ~5 min)
python src/run_evaluation.py --track custom

# 5. Run the full evaluation (723 instances, ~45-60 min)
python src/run_evaluation.py --track all
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--provider` | `anthropic` | `anthropic` or `openai` |
| `--model` | provider default | e.g. `gpt-4o`, `claude-haiku-4-5-20251001` |
| `--track` | `all` | `fire`, `custom`, or `all` |
| `--dry-run` | off | Skip API calls; validates data loading and scoring |
| `--max-sentences` | all | Limit to N sentences for quick tests |
| `--delay` | `0.3` | Seconds between API calls (rate-limit buffer) |

Results are written to `results/` with filenames tagged by provider and model, so runs with
different LLMs do not overwrite each other:

```
results/scores_summary_anthropic_claude-haiku-4-5-20251001.json
results/model_a_raw_anthropic_claude-haiku-4-5-20251001.json
results/model_b_raw_anthropic_claude-haiku-4-5-20251001.json
```

### Reproducing the paper's three runs

```bash
python src/run_evaluation.py --track all --provider anthropic --model claude-haiku-4-5-20251001
python src/run_evaluation.py --track all --provider openai    --model gpt-4o
python src/run_evaluation.py --track all --provider openai    --model gpt-4o-mini
```

**Note on reproducibility:** LLM outputs are not fully deterministic, so exact metric values will
vary slightly between runs. The reported effects — the precision and consistency gains, and the
relation-level moderation pattern — are stable across runs.

---

## Datasets

### Competitive Intelligence Supplement (new)

`data/gold/CI_supplement_gold.csv` — 74 hand-annotated relation instances across 48 sentences,
covering `competesWith` (25), `createsOpportunity` (25), and `createsThreat` (24). Sentences were
drawn from real competitive-intelligence discourse (10-K filings, market intelligence reports,
business news) and grounded in publicly documented competitive dynamics. All instances were
manually validated by the author.

Columns: `id`, `sentence`, `head_text`, `head_type`, `tail_text`, `tail_type`, `relation`, `notes`.

To our knowledge this is among the first publicly described datasets designed specifically for
evaluating strategic competitive-intelligence relations.

### FIRE-derived gold labels

Derived from the FIRE financial relation extraction benchmark (Hamad et al., 2024, NAACL Findings),
released under CC-BY 4.0. Two transformations were applied:

1. **`develops`** — FIRE's `Productof` covers any "manufactured, sold, offered, or marketed by"
   relation, which is broader than `develops` (creation only). A rule-based classifier assigned each
   of the 478 instances to Development / SaleOrMarketing / Unspecified; only Development instances
   were retained (167). Classifier reliability was validated against manual labelling of a 20%
   stratified sample (n = 96), yielding **Cohen's κ = 0.904** ("almost perfect agreement"). The
   sample is in `docs/kappa_validation_sample.csv`.

2. **Direction transposition** — FIRE annotates `Productof` as (Product → Company); the ontology
   defines `develops` as (Company → Product). Head and tail were transposed for all 167 instances.

`operatesIn` (482 instances) required no content filtering; FIRE's `Sector` type maps directly to
the ontology's `Market` class.

---

## Metrics

| Metric | Definition |
|---|---|
| Precision | TP / (TP + FP), micro-averaged |
| Recall | TP / (TP + FN), micro-averaged |
| F1-score | Harmonic mean of precision and recall |
| Hallucination rate | Proportion of triples whose head or tail span is not a substring of the source sentence (surface-grounding check) |
| Ontological consistency | Proportion of predicted triples satisfying all domain/range constraints |
| Graph completeness | Equivalent to micro-recall |
| Redundancy rate | Proportion of duplicate triples within a sentence's output |

Triple matching normalises the (head, relation, tail) key — lower-cased, punctuation-stripped,
whitespace-collapsed. Entity types are excluded from the matching key so a correct relation is not
penalised for a minor type-labelling difference.

Statistical significance uses the Wilcoxon signed-rank test on sentence-level paired observations,
with effect size r = Z / √N.

---

## Licence

- **Code** (`src/`): MIT Licence — see `LICENSE`.
- **CI Supplement** (`data/gold/CI_supplement_gold.csv`): CC-BY 4.0.
- **FIRE-derived files**: inherit the CC-BY 4.0 licence of the original FIRE dataset. Please cite
  Hamad et al. (2024) in addition to this work if you use them.

## Citation

```bibtex
@article{montasser2026ontology,
  title   = {When Does Ontology Guidance Improve LLM Knowledge Extraction?
             A Relation-Level Analysis for Competitive Intelligence Knowledge Graphs},
  author  = {Montasser, Hend},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Hend Montasser — Department of Information Systems, Helwan University, Cairo, Egypt.
Please open an issue for questions about the code or data.
