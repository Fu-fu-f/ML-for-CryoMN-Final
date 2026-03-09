#!/usr/bin/env python3
"""
Shared active-model resolver with iteration-aware conflict handling.
"""

import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from iteration_metadata import (
    derive_iteration_dir,
    load_iteration_history,
    method_uses_composite,
    normalize_model_method,
    stamp_model_metadata,
    write_metadata_with_notice,
)
from update_model_weighted_prior import CompositeGP


_main_module = sys.modules.get('__main__')
if _main_module is not None and not hasattr(_main_module, 'CompositeGP'):
    setattr(_main_module, 'CompositeGP', CompositeGP)


class ModelResolutionError(RuntimeError):
    """Raised when the active model cannot be resolved safely."""


@dataclass
class IterationCandidate:
    """One validated iteration that can be loaded safely."""
    iteration: int
    model_method: str
    iteration_dir: str
    is_composite_model: bool
    metadata: Dict
    directory: str


@dataclass
class ActiveModelResolution:
    """Loaded model plus the validated iteration information behind it."""
    gp: object
    scaler: Optional[object]
    metadata: Dict
    is_composite: bool
    directory: str
    iteration: Optional[int]
    iteration_dir: Optional[str]
    model_method: str


def _try_load_json(path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Read JSON and return a user-facing error instead of throwing."""
    if not os.path.exists(path):
        return None, f"{path} is missing."
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return None, f"{path} is malformed JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} does not contain a JSON object."
    return data, None


def _load_history_entries(project_root: str) -> Tuple[List[Dict], Optional[str]]:
    """Load iteration history while preserving parse errors for conflict reporting."""
    history_path = os.path.join(project_root, 'data', 'validation', 'iteration_history.json')
    if not os.path.exists(history_path):
        return [], None
    try:
        return load_iteration_history(project_root), None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return [], f"{history_path} could not be read: {exc}"


def _normalize_history_method(entry: Dict) -> str:
    """Resolve old and new history fields to one normalized method label."""
    return normalize_model_method(
        entry.get('model_method') or entry.get('method'),
        entry.get('is_composite_model'),
    )


def _build_iteration_candidate(model_dir: str, entry: Dict) -> Tuple[Optional[IterationCandidate], Optional[str]]:
    """Validate one history entry against the on-disk iteration artifacts."""
    iteration = entry.get('iteration')
    if not isinstance(iteration, int) or iteration <= 0:
        return None, f"Invalid history entry without a positive integer iteration: {entry!r}"

    model_method = _normalize_history_method(entry)
    iteration_dir = entry.get('iteration_dir') or derive_iteration_dir(iteration, model_method)
    directory = os.path.join(model_dir, iteration_dir)
    if not os.path.isdir(directory):
        return None, f"Iteration {iteration} is recorded, but {directory} does not exist."

    metadata_path = os.path.join(directory, 'model_metadata.json')
    metadata, metadata_error = _try_load_json(metadata_path)
    if metadata_error:
        return None, f"Iteration {iteration} is unusable: {metadata_error}"

    metadata_method = normalize_model_method(
        metadata.get('model_method') or metadata.get('weighting_method'),
        metadata.get('is_composite_model'),
    )
    metadata_iteration_dir = metadata.get('iteration_dir')
    metadata_iteration = metadata.get('iteration')

    if metadata_iteration not in (None, iteration):
        return None, (
            f"Iteration {iteration} metadata points to iteration {metadata_iteration}, "
            "so the record is inconsistent."
        )
    if metadata_iteration_dir not in (None, iteration_dir):
        return None, (
            f"Iteration {iteration} metadata points to {metadata_iteration_dir} instead of "
            f"{iteration_dir}."
        )
    if metadata.get('model_method') is not None and metadata_method != model_method:
        return None, (
            f"Iteration {iteration} metadata says {metadata_method}, but history says "
            f"{model_method}."
        )

    is_composite_model = metadata.get('is_composite_model')
    if is_composite_model is None:
        is_composite_model = method_uses_composite(model_method)

    composite_path = os.path.join(directory, 'composite_model.pkl')
    model_path = os.path.join(directory, 'gp_model.pkl')
    scaler_path = os.path.join(directory, 'scaler.pkl')

    if is_composite_model:
        if not os.path.exists(composite_path):
            return None, (
                f"Iteration {iteration} is marked composite, but {composite_path} is missing."
            )
    else:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, (
                f"Iteration {iteration} is marked standard, but gp_model.pkl/scaler.pkl are missing."
            )

    stamped_metadata = stamp_model_metadata(
        metadata,
        iteration=iteration,
        model_method=model_method,
        iteration_dir=iteration_dir,
        is_composite_model=is_composite_model,
    )
    if 'feature_names' not in stamped_metadata:
        return None, f"Iteration {iteration} metadata is missing feature_names."

    return IterationCandidate(
        iteration=iteration,
        model_method=model_method,
        iteration_dir=iteration_dir,
        is_composite_model=is_composite_model,
        metadata=stamped_metadata,
        directory=directory,
    ), None


def _collect_iteration_candidates(project_root: str, model_dir: str) -> Tuple[List[IterationCandidate], List[str]]:
    """Validate recorded iterations and collect user-facing conflict messages."""
    history_entries, history_error = _load_history_entries(project_root)
    issues: List[str] = []
    if history_error:
        issues.append(history_error)
        return [], issues

    if not history_entries:
        return [], issues

    seen_iterations = set()
    candidates: List[IterationCandidate] = []
    raw_iterations: List[int] = []

    for entry in history_entries:
        iteration = entry.get('iteration')
        if isinstance(iteration, int) and iteration > 0:
            raw_iterations.append(iteration)
        if iteration in seen_iterations:
            issues.append(f"Iteration {iteration} appears multiple times in iteration history.")
            continue
        seen_iterations.add(iteration)

        candidate, issue = _build_iteration_candidate(model_dir, entry)
        if issue:
            issues.append(issue)
            continue
        candidates.append(candidate)

    if raw_iterations:
        highest_recorded = max(raw_iterations)
        highest_valid = max((candidate.iteration for candidate in candidates), default=None)
        if highest_valid == highest_recorded:
            issues = [issue for issue in issues if not issue.startswith('Iteration ')]
        else:
            issues.append(
                f"Latest recorded iteration {highest_recorded} is not loadable, so active "
                "metadata cannot be trusted automatically."
            )

    return candidates, issues


def _describe_candidate(candidate: IterationCandidate) -> str:
    """Human-readable description for prompts and logs."""
    model_kind = 'COMPOSITE' if candidate.is_composite_model else 'STANDARD'
    return (
        f"iteration {candidate.iteration} [{model_kind}; {candidate.model_method}; "
        f"{candidate.iteration_dir}]"
    )


def _prompt_for_iteration_choice(candidates: List[IterationCandidate], issues: List[str]) -> IterationCandidate:
    """Ask the user which recorded iteration should be used after a conflict."""
    print(">>> Active model metadata conflict detected.")
    for issue in issues:
        print(f">>> {issue}")

    if not candidates:
        raise ModelResolutionError(
            "No valid recorded iterations are available. Restore the metadata/history manually."
        )

    print(">>> Available valid iterations:")
    for candidate in sorted(candidates, key=lambda item: item.iteration):
        print(f">>>   {candidate.iteration}: {_describe_candidate(candidate)}")

    try:
        raw_choice = input("Enter the iteration number to use: ").strip()
    except EOFError as exc:
        raise ModelResolutionError(
            "Interactive input is unavailable, so the metadata conflict cannot be resolved safely."
        ) from exc

    if not raw_choice or not raw_choice.lstrip('-').isdigit():
        raise ModelResolutionError(f"Iteration selection '{raw_choice}' is nonsensical.")

    chosen_iteration = int(raw_choice)
    selected = next((candidate for candidate in candidates if candidate.iteration == chosen_iteration), None)
    if selected is None:
        raise ModelResolutionError(
            f"Iteration {chosen_iteration} is nonsensical: no such valid iteration exists."
        )
    return selected


def _load_model_from_candidate(candidate: IterationCandidate) -> ActiveModelResolution:
    """Load one fully validated iteration."""
    if candidate.is_composite_model:
        composite_path = os.path.join(candidate.directory, 'composite_model.pkl')
        with open(composite_path, 'rb') as f:
            gp = pickle.load(f)
        print(f">>> Using COMPOSITE model from {_describe_candidate(candidate)}")
        return ActiveModelResolution(
            gp=gp,
            scaler=None,
            metadata=candidate.metadata,
            is_composite=True,
            directory=candidate.directory,
            iteration=candidate.iteration,
            iteration_dir=candidate.iteration_dir,
            model_method=candidate.model_method,
        )

    model_path = os.path.join(candidate.directory, 'gp_model.pkl')
    scaler_path = os.path.join(candidate.directory, 'scaler.pkl')
    with open(model_path, 'rb') as f:
        gp = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f">>> Using STANDARD GP model from {_describe_candidate(candidate)}")
    return ActiveModelResolution(
        gp=gp,
        scaler=scaler,
        metadata=candidate.metadata,
        is_composite=False,
        directory=candidate.directory,
        iteration=candidate.iteration,
        iteration_dir=candidate.iteration_dir,
        model_method=candidate.model_method,
    )


def _load_root_model_without_history(model_dir: str) -> ActiveModelResolution:
    """Use the active root model only when there is no iteration history to consult."""
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    metadata, metadata_error = _try_load_json(metadata_path)
    if metadata_error:
        raise ModelResolutionError(metadata_error)

    if 'feature_names' not in metadata:
        raise ModelResolutionError(f"{metadata_path} is missing feature_names.")

    model_method = normalize_model_method(
        metadata.get('model_method') or metadata.get('weighting_method'),
        metadata.get('is_composite_model'),
    )
    wants_composite = bool(metadata.get('is_composite_model', False))
    composite_path = os.path.join(model_dir, 'composite_model.pkl')
    model_path = os.path.join(model_dir, 'gp_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    if wants_composite:
        if not os.path.exists(composite_path):
            raise ModelResolutionError(
                "Root metadata selects a composite model, but composite_model.pkl is missing. "
                "No automatic fallback will be used."
            )
        with open(composite_path, 'rb') as f:
            gp = pickle.load(f)
        print(">>> Using COMPOSITE model from root metadata (no iteration history found)")
        return ActiveModelResolution(
            gp=gp,
            scaler=None,
            metadata=metadata,
            is_composite=True,
            directory=model_dir,
            iteration=metadata.get('iteration'),
            iteration_dir=metadata.get('iteration_dir'),
            model_method=model_method,
        )

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise ModelResolutionError(
            "Root metadata selects a standard model, but gp_model.pkl/scaler.pkl are missing."
        )

    with open(model_path, 'rb') as f:
        gp = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(">>> Using STANDARD GP model from root metadata (no iteration history found)")
    return ActiveModelResolution(
        gp=gp,
        scaler=scaler,
        metadata=metadata,
        is_composite=False,
        directory=model_dir,
        iteration=metadata.get('iteration'),
        iteration_dir=metadata.get('iteration_dir'),
        model_method=model_method,
    )


def resolve_active_model(project_root: str) -> ActiveModelResolution:
    """Load the active model with strict iteration-aware conflict resolution."""
    model_dir = os.path.join(project_root, 'models')
    candidates, issues = _collect_iteration_candidates(project_root, model_dir)
    if not candidates:
        if issues:
            raise ModelResolutionError(" ".join(issues))
        return _load_root_model_without_history(model_dir)

    latest_candidate = max(candidates, key=lambda item: item.iteration)
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    root_metadata, metadata_error = _try_load_json(metadata_path)

    conflicts = list(issues)
    if metadata_error:
        conflicts.append(metadata_error)
    else:
        if root_metadata.get('iteration') is None:
            conflicts.append("Root model metadata is missing the iteration field.")
        if root_metadata.get('iteration_dir') is None:
            conflicts.append("Root model metadata is missing the iteration_dir field.")
        if root_metadata.get('model_method') is None:
            conflicts.append("Root model metadata is missing the model_method field.")

        if not conflicts:
            root_method = normalize_model_method(
                root_metadata.get('model_method') or root_metadata.get('weighting_method'),
                root_metadata.get('is_composite_model'),
            )
            if root_metadata.get('iteration') != latest_candidate.iteration:
                conflicts.append(
                    f"Root metadata points to iteration {root_metadata.get('iteration')}, "
                    f"but the latest valid iteration is {latest_candidate.iteration}."
                )
            if root_metadata.get('iteration_dir') != latest_candidate.iteration_dir:
                conflicts.append(
                    f"Root metadata points to {root_metadata.get('iteration_dir')}, but the "
                    f"latest valid directory is {latest_candidate.iteration_dir}."
                )
            if root_method != latest_candidate.model_method:
                conflicts.append(
                    f"Root metadata says {root_method}, but the latest valid iteration says "
                    f"{latest_candidate.model_method}."
                )
            if bool(root_metadata.get('is_composite_model')) != latest_candidate.is_composite_model:
                conflicts.append(
                    "Root metadata disagrees with the latest valid iteration on whether the "
                    "active model is composite."
                )

    if conflicts:
        selected_candidate = _prompt_for_iteration_choice(candidates, conflicts)
        write_metadata_with_notice(
            metadata_path,
            selected_candidate.metadata,
            selected_candidate.iteration,
            selected_candidate.model_method,
            reason='conflict resolution',
        )
        return _load_model_from_candidate(selected_candidate)

    return _load_model_from_candidate(latest_candidate)
