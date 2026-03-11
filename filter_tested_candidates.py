#!/usr/bin/env python3
"""
Filter candidate formulations that have already been tested in wet-lab validation.

The detector compares formulations using the same display precision used in the
candidate summary files so that previously tested formulations still match even
when candidate CSVs store higher-precision floats than the wet-lab records.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd


CANDIDATE_FILE_PATTERN = re.compile(
    r"^(?P<basename>(?:bo_)?candidates_(?:general|dmso_free))_"
    r"(?P<tag>iteration_(?P<iteration>\d+)(?:_[A-Za-z0-9_]+)?)\.csv$"
)


@dataclass(frozen=True)
class CandidateFile:
    """One candidate CSV plus its parsed iteration identity."""

    path: str
    basename: str
    tag: str
    iteration: int


def is_feature_column(name: str) -> bool:
    """Return True when a column is one formulation feature."""
    return name.endswith("_M") or name.endswith("_pct")


def format_feature_value(feature_name: str, value: float) -> str:
    """Format one ingredient exactly like the candidate summary files."""
    clean_name = feature_name.replace("_pct", "").replace("_M", "")
    if feature_name.endswith("_pct"):
        if float(f"{value:.1f}") == 0.0:
            return ""
        return f"{value:.1f}% {clean_name}"
    if value >= 1.0:
        if float(f"{value:.2f}") == 0.0:
            return ""
        return f"{value:.2f}M {clean_name}"
    if value >= 0.001:
        if float(f"{value * 1000:.1f}") == 0.0:
            return ""
        return f"{value * 1000:.1f}mM {clean_name}"
    if float(f"{value * 1e6:.1f}") == 0.0:
        return ""
    return f"{value * 1e6:.1f}µM {clean_name}"


def format_formulation(row: pd.Series, feature_names: Sequence[str]) -> str:
    """Format a formulation in the same style as the candidate summaries."""
    parts: List[str] = []
    for name in feature_names:
        value = row.get(name, 0.0)
        if pd.isna(value) or abs(float(value)) <= 1e-6:
            continue
        formatted = format_feature_value(name, float(value))
        if formatted:
            parts.append(formatted)
    return " + ".join(parts) if parts else "No active ingredients"


def formulation_signature(row: pd.Series, feature_names: Sequence[str]) -> str:
    """
    Build a comparison signature using summary-style formatting.

    This intentionally uses rounded display text rather than raw floats because
    wet-lab entries are typically copied from the human-readable candidate files.
    """
    return format_formulation(row, feature_names)


def project_root_from_script() -> str:
    """Resolve the repository root from this script location."""
    return os.path.dirname(os.path.abspath(__file__))


def find_candidate_files(results_dir: str) -> List[CandidateFile]:
    """Collect iteration-specific candidate CSVs from the results directory."""
    records: List[CandidateFile] = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith("_untested.csv"):
            continue
        match = CANDIDATE_FILE_PATTERN.match(filename)
        if not match:
            continue
        records.append(
            CandidateFile(
                path=os.path.join(results_dir, filename),
                basename=match.group("basename"),
                tag=match.group("tag"),
                iteration=int(match.group("iteration")),
            )
        )
    return records


def prompt_for_iteration(candidate_files: Sequence[CandidateFile]) -> Tuple[List[CandidateFile], int]:
    """Ask the user which iteration to inspect; blank input picks the latest."""
    available_iterations = sorted({record.iteration for record in candidate_files})
    available_tags = sorted({record.tag for record in candidate_files})

    print("Available candidate iterations:")
    for iteration in available_iterations:
        tags = sorted({record.tag for record in candidate_files if record.iteration == iteration})
        print(f"  Iteration {iteration}: {', '.join(tags)}")

    raw_choice = input(
        "\nEnter the iteration number or full iteration tag to inspect "
        "(press Enter for latest available): "
    ).strip()

    if not raw_choice:
        latest_iteration = max(available_iterations)
        print(f"No iteration provided. Using latest candidate results: iteration {latest_iteration}.")
        return (
            [record for record in candidate_files if record.iteration == latest_iteration],
            latest_iteration,
        )

    if raw_choice in available_tags:
        selected = [record for record in candidate_files if record.tag == raw_choice]
        return selected, selected[0].iteration

    normalized_choice = raw_choice
    if normalized_choice.startswith("iteration_"):
        suffix = normalized_choice[len("iteration_"):]
        if suffix.isdigit():
            normalized_choice = suffix

    if normalized_choice.isdigit():
        requested_iteration = int(normalized_choice)
        selected = [record for record in candidate_files if record.iteration == requested_iteration]
        if selected:
            return selected, requested_iteration

    raise ValueError(
        f"Could not find candidate results for '{raw_choice}'. "
        "Use one of the listed iteration numbers or full iteration tags."
    )


def load_validation_results(validation_path: str) -> pd.DataFrame:
    """Load wet-lab validation rows."""
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Validation results file is missing: {validation_path}")
    validation_df = pd.read_csv(validation_path)
    if validation_df.empty:
        raise ValueError("Validation results file is empty.")
    return validation_df


def build_feature_order(validation_df: pd.DataFrame, candidate_df: pd.DataFrame) -> List[str]:
    """Use validation feature order as canonical, then append any candidate-only features."""
    feature_names = [name for name in validation_df.columns if is_feature_column(name)]
    for name in candidate_df.columns:
        if is_feature_column(name) and name not in feature_names:
            feature_names.append(name)
    return feature_names


def build_tested_lookup(
    validation_df: pd.DataFrame, feature_names: Sequence[str]
) -> Dict[str, List[str]]:
    """Map a formulation signature to the wet-lab experiment IDs that tested it."""
    tested_lookup: Dict[str, List[str]] = {}
    for _, row in validation_df.iterrows():
        signature = formulation_signature(row, feature_names)
        experiment_id = str(row.get("experiment_id", "UNKNOWN"))
        tested_lookup.setdefault(signature, []).append(experiment_id)
    return tested_lookup


def build_summary_text(
    candidates_df: pd.DataFrame,
    feature_names: Sequence[str],
    source_filename: str,
    removed_count: int,
    original_count: int,
    is_bo_file: bool,
) -> str:
    """Render a human-readable summary matching the candidate-file style."""
    header = "CryoMN Bayesian Optimization Candidates (Untested Only)"
    if not is_bo_file:
        header = "CryoMN Optimized Formulation Candidates (Untested Only)"

    lines = [
        "=" * 80,
        header,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source candidate file: {source_filename}",
        f"Wet-lab tested formulations removed: {removed_count} of {original_count}",
        "=" * 80,
        "",
    ]

    if candidates_df.empty:
        lines.append("No untested formulations remain in this candidate file.")
        lines.append("")
        return "\n".join(lines)

    for _, row in candidates_df.iterrows():
        lines.append(f"Rank {int(row['rank'])}: {format_formulation(row, feature_names)}")
        if is_bo_file and "acquisition_value" in row.index:
            lines.append(f"  Acquisition Value: {row['acquisition_value']:.4f}")
        lines.append(
            f"  Predicted viability: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%"
        )
        lines.append(f"  DMSO: {row['dmso_percent']:.1f}%")
        lines.append(f"  Ingredients: {int(row['n_ingredients'])}")
        lines.append("")

    return "\n".join(lines)


def filter_candidate_file(candidate_file: CandidateFile, validation_df: pd.DataFrame):
    """Filter one candidate CSV against the wet-lab validation set."""
    candidates_df = pd.read_csv(candidate_file.path)
    if candidates_df.empty:
        print(f"\nSkipping empty candidate file: {os.path.basename(candidate_file.path)}")
        return candidates_df.copy(), 0

    feature_names = build_feature_order(validation_df, candidates_df)
    tested_lookup = build_tested_lookup(validation_df, feature_names)

    signatures = candidates_df.apply(
        lambda row: formulation_signature(row, feature_names), axis=1
    )
    tested_mask = signatures.isin(tested_lookup)
    filtered_df = candidates_df.loc[~tested_mask].copy()
    return filtered_df, int(tested_mask.sum())


def write_filtered_outputs(
    candidate_file: CandidateFile,
    filtered_df: pd.DataFrame,
    removed_count: int,
    validation_df: pd.DataFrame,
    output_dir: str,
):
    """Write the filtered CSV and summary to the dedicated Untested output folder."""
    feature_names = build_feature_order(validation_df, filtered_df)
    original_candidates_df = pd.read_csv(candidate_file.path)
    output_csv_path = os.path.join(
        output_dir,
        os.path.basename(candidate_file.path).replace(".csv", "_untested.csv"),
    )
    output_summary_path = os.path.join(
        output_dir,
        os.path.basename(candidate_file.path).replace(".csv", "_untested_summary.txt"),
    )
    filtered_df.to_csv(output_csv_path, index=False)

    summary_text = build_summary_text(
        candidates_df=filtered_df,
        feature_names=feature_names,
        source_filename=os.path.basename(candidate_file.path),
        removed_count=removed_count,
        original_count=len(original_candidates_df),
        is_bo_file=candidate_file.basename.startswith("bo_"),
    )
    with open(output_summary_path, "w") as handle:
        handle.write(summary_text)
        handle.write("\n")

    print("\n" + summary_text)
    print(f"Filtered CSV saved to: {output_csv_path}")
    print(f"Filtered summary saved to: {output_summary_path}")

    tested_lookup = build_tested_lookup(validation_df, feature_names)
    signatures = original_candidates_df.apply(
        lambda row: formulation_signature(row, feature_names), axis=1
    )
    tested_mask = signatures.isin(tested_lookup)
    if tested_mask.any():
        print("Matched wet-lab experiments:")
        matched_pairs = []
        for signature in signatures[tested_mask]:
            matched_pairs.extend(tested_lookup.get(signature, []))
        print("  " + ", ".join(sorted(set(matched_pairs))))


def main():
    """Run the tested-candidate detector."""
    project_root = project_root_from_script()
    results_dir = os.path.join(project_root, "results")
    untested_root = os.path.join(project_root, "Untested")
    validation_path = os.path.join(project_root, "data", "validation", "validation_results.csv")

    print("=" * 80)
    print("CryoMN Tested Candidate Detector")
    print("=" * 80)

    candidate_files = find_candidate_files(results_dir)
    if not candidate_files:
        print("No iteration-specific candidate CSV files were found in results/.")
        return

    try:
        selected_candidate_files, selected_iteration = prompt_for_iteration(candidate_files)
        validation_df = load_validation_results(validation_path)
    except (EOFError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return

    output_dir = os.path.join(untested_root, f"Iteration {selected_iteration}")
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"\nLoaded {len(validation_df)} wet-lab validation rows from "
        f"{os.path.relpath(validation_path, project_root)}."
    )
    print(f"Saving untested outputs to: {os.path.relpath(output_dir, project_root)}")
    print("Filtering candidate files:")
    for record in selected_candidate_files:
        print(f"  {os.path.basename(record.path)}")

    for candidate_file in selected_candidate_files:
        filtered_df, removed_count = filter_candidate_file(candidate_file, validation_df)
        write_filtered_outputs(
            candidate_file=candidate_file,
            filtered_df=filtered_df,
            removed_count=removed_count,
            validation_df=validation_df,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
