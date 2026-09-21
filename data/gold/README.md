# Gold-standard datasets

| File | Instances | Relations | Source |
|---|---|---|---|
| `CI_supplement_gold.csv` | 74 | competesWith, createsOpportunity, createsThreat | Newly annotated for this study |
| `FIRE_develops_gold.csv` | 167 | develops | Filtered from FIRE `Productof` |
| `FIRE_operatesIn_gold.csv` | 482 | operatesIn | FIRE `Sector` |

**Total: 723 gold-standard relation instances across 476 unique sentences.**

## Column reference

### CI_supplement_gold.csv
`id`, `sentence`, `head_text`, `head_type`, `tail_text`, `tail_type`, `relation`, `notes`

### FIRE_develops_gold.csv
`split`, `head_text`, `head_type`, `tail_text`, `tail_type`, `gold_label`, `sentence`

Only rows where `gold_label == "Development"` are used as gold `develops` instances.
Head and tail have already been transposed to (Company → Product) to match the ontology
convention; FIRE's original annotation direction is (Product → Company).

### FIRE_operatesIn_gold.csv
`split`, `head_text`, `head_type`, `tail_text`, `tail_type`, `sentence`

`tail_type` has been normalised from FIRE's `Sector` to the ontology's `Market` class.

## Licensing

`CI_supplement_gold.csv` is released under CC-BY 4.0.

The two FIRE-derived files inherit the CC-BY 4.0 licence of the original FIRE dataset.
If you use them, please also cite:

> Hamad, R., Kim, K., Mohanty, S., & Reddy, C. K. (2024). FIRE: A dataset for financial
> relation extraction. In *Findings of the ACL: NAACL 2024* (pp. 4060–4078).
