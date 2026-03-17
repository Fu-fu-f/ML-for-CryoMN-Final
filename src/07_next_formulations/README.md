# Step 7: Next Formulations

## Overview

This module builds the next wet-lab batch from the active model state.

It always writes exactly **10 formulations**:

- **5 exploitation** picks from the active iteration's `05_bo_optimization` outputs
- **5 exploration/calibration** picks generated from the latest completed stage's residual blind spots

The script is intentionally strict. It validates required inputs before
generation, validates all 10 outputs again before write, and aborts without
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
- drop already tested formulations
- rank by predicted viability with uncertainty and acquisition tie-breaks
- keep a simple chemistry-family diversity cap so the final 5 are not all near-duplicates

### Exploration / Calibration

- compute residuals on the latest completed wet-lab batch using that batch's frozen model
- convert positive residuals into feature-level and pair-level blind-spot signals
- generate probes by midpoint interpolation or local perturbation around underpredicted anchors
- clip to BO bounds and enforce the BO-derived ingredient-count limit
- re-score with the active model
- backfill from BO only if fewer than 5 valid generated probes survive

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
