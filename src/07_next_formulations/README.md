# Step 7: Next Formulations

## Overview

This module builds the next wet-lab batch from the active model state.

It always writes exactly **20 formulations**:

- **10 exploitation** picks from the active iteration's `05_bo_optimization` outputs
- **10 exploration/calibration** picks chosen from generated residual probes first, then BO fallback only if needed

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
the latest completed wet-lab stage in `validation_results.csv` must be `N-1`.

## Selection Logic

### Exploitation

- source only from `05` BO candidate files
- normalize loaded BO candidates with the same practical concentration floor used by `05`
- drop already tested formulations
- rank by predicted viability with uncertainty and acquisition tie-breaks
- keep a simple chemistry-family diversity cap so the final 10 are not all near-duplicates

### Exploration / Calibration

- compute residuals on the latest completed wet-lab batch using that batch's frozen model
- aggregate historical positive-residual anchors across all completed wet-lab stages
- try positive-residual anchor thresholds in descending order: `10.0`, `8.0`, `5.0`, `2.0`, `0.0`
- convert positive residuals into feature-level and pair-level blind-spot signals
- generate probes by midpoint interpolation or local perturbation around underpredicted anchors from any historical positive-residual stage
- clip to BO bounds, zero sub-threshold trace ingredients, and enforce the BO-derived ingredient-count limit
- re-score with the active model
- keep generated probes ahead of BO fallback, even if a relaxed family cap is needed to fill the exploration bucket
- backfill from BO only if fewer than 10 valid generated probes survive after the generated-only top-up pass

This is why `07` does not use `03_optimization` as a primary source. The
exploration half is designed directly from model failures, not from a random
candidate pool.

## Outputs

Outputs are written under:

- `results/next_formulations/<iteration_tag>/next_formulations.csv`
- `results/next_formulations/<iteration_tag>/next_formulations_summary.txt`
- `results/next_formulations/<iteration_tag>/next_formulations_metadata.json`
- `results/next_formulations/<iteration_tag>/input_validation.json`

`next_formulations.csv` includes:

- recommendation type and bucket rank
- origin and source file / rank when applicable
- anchor stage and anchor experiments for generated probes
- predicted viability and uncertainty
- blind-spot and novelty scores
- canonical feature columns in model order
- formulation text and rationale

Displayed formulation identity follows the same floor as BO generation and `06`
matching:

- `_pct` values `<0.1%` are omitted
- `_M` values `<0.001 M` (`<1.0 mM`) are omitted

The metadata and input-validation artifacts also record which residual thresholds
were tried, which threshold was selected, how many exploration rows came
from generated probes versus BO fallback, and which historical anchor stages
fed the generated probes.

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
