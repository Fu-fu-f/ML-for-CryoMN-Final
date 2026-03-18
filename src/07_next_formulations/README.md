# Step 7: Next Formulations

## Overview

This module builds the next wet-lab batch from the active model state.

It always writes exactly **20 formulations**:

- **8 exploitation** picks from the active iteration's `05_bo_optimization` outputs
- **12 exploration/calibration** picks assembled from local-rank probes, blind-spot probes, and BO fallback only if needed

The script is intentionally strict. It validates required inputs before
generation, validates all 20 outputs again before write, and aborts without
writing partial results if anything is inconsistent.

## Usage

```bash
python src/07_next_formulations/next_formulations.py
```

Optional flags:

```bash
python src/07_next_formulations/next_formulations.py --stage 4
python src/07_next_formulations/next_formulations.py --overwrite
```

## Inputs

- `models/model_metadata.json` to resolve the active target stage
- `models/<iteration_dir>/` for the active model artifacts and observed context
- `models/<previous_iteration_dir>/` or `models/literature_only/` for the last completed stage model
- `data/validation/validation_results.csv` for stage detection and residual learning
- `results/bo_candidates_general_<iteration_tag>.csv`
- `results/bo_candidates_dmso_free_<iteration_tag>.csv`

The stage sequence must be contiguous. If the active stage is iteration `N`,
`validation_results.csv` must contain completed wet-lab results through stage `N-1`.

## Selection Logic

### Exploitation

- source only from `05` BO candidate files
- normalize loaded BO candidates with the same practical concentration floor used by `05`
- drop already tested formulations
- rank by predicted viability with uncertainty and acquisition tie-breaks
- keep a simple chemistry-family diversity cap so the final 8 are not all near-duplicates

### Exploration / Calibration

- compute residuals on stage `N-1` using that stage's frozen model
- aggregate historical positive-residual anchors across all completed wet-lab stages
- try positive-residual anchor thresholds in descending order: `10.0`, `8.0`, `5.0`, `2.0`, `0.0`
- convert positive residuals into feature-level and pair-level blind-spot signals
- generate local-rank probes from the top exploitation anchors by scaling the top two active ingredients down and up
- generate blind-spot probes by midpoint interpolation or local perturbation around underpredicted anchors from historical positive-residual stages
- clip to BO bounds, zero sub-threshold trace ingredients, and enforce the BO-derived ingredient-count limit
- re-score with the active model
- keep local-rank and blind-spot probes ahead of BO fallback, even if a relaxed family cap is needed to fill the exploration bucket
- backfill from BO only if fewer than 12 valid exploration rows survive after the generated-only top-up pass

This is why `07` does not use `03_optimization` as a primary source. The
exploration half is designed directly from model failures, not from a random
candidate pool.

## Outputs

Outputs are written under:

- `results/next_formulations/<iteration_tag>/next_formulations.csv`
- `results/next_formulations/<iteration_tag>/next_formulations_summary.txt`
- `results/next_formulations/<iteration_tag>/next_formulations_metadata.json`
- `results/next_formulations/<iteration_tag>/input_validation.json`
- `results/next_formulations/<iteration_tag>/batch_recommendations.json`
- `results/next_formulations/<iteration_tag>/batch_recommendations.csv`

`next_formulations.csv` includes:

- recommendation type and bucket rank
- origin and source file / rank when applicable
- anchor stage and anchor experiments for generated probes
- predicted viability and uncertainty
- blind-spot and novelty scores
- canonical feature columns in model order
- formulation text and rationale

`batch_recommendations.json` and `batch_recommendations.csv` include one
recommended subset for each wet-lab batch size from 6 through 12. The subset
search is exact over the generated 20-row slate and uses a heuristic utility
score that balances:

- predicted viability
- uncertainty
- blind-spot value
- novelty
- chemistry-family diversity
- the intended exploit / local-rank / blind-spot mix

Displayed formulation identity follows the same floor as BO generation and `06`
matching:

- `_pct` values `<0.1%` are omitted
- `_M` values `<0.001 M` (`<1.0 mM`) are omitted

The metadata and input-validation artifacts also record which residual thresholds
were tried, which threshold was selected, how many exploration rows came
from local-rank probes, blind-spot probes, and BO fallback, which historical
anchor stages fed the generated probes, and how each recommended smaller batch
was scored.

## Failure Mode

This module fails hard on:

- missing or malformed validation columns
- missing model artifacts or BO candidate files
- feature-space mismatches across validation data, models, and candidate files
- non-contiguous stage history
- duplicate or already tested final outputs
- violations of BO bounds, DMSO cap, or effective ingredient-count limits

If the run succeeds, `input_validation.json` records exactly which inputs were
used so future runs can be audited.
