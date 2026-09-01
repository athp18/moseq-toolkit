from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that must always be present in moseq_df, independent of group_cols.
_REQUIRED_COLUMNS = [
    "SubjectName",
    "SessionName",
    "frame index",
    "labels (usage sort)",
    "centroid_x_mm",
    "centroid_y_mm",
    "dist_to_center_px",
]
_UNLABELED_SYLLABLE = -5
_DEFAULT_TIMEPOINT = "single_session"


def _validate_columns(moseq_df: pd.DataFrame, group_cols: list[str]) -> None:
    """Fail loudly if anything the toolkit depends on is missing."""
    missing_required = [c for c in _REQUIRED_COLUMNS if c not in moseq_df.columns]
    missing_group_cols = [c for c in group_cols if c not in moseq_df.columns]

    if missing_required or missing_group_cols:
        problems = []
        if missing_required:
            problems.append(f"missing required columns: {missing_required}")
        if missing_group_cols:
            problems.append(f"missing group_cols columns: {missing_group_cols}")
        raise ValueError(
            "load_moseq_df: invalid moseq_df -> " + "; ".join(problems)
        )


def _compute_frame_displacement(frames: pd.DataFrame) -> pd.Series:
    """Per-frame Euclidean displacement between consecutive centroid positions,
    in mm, zeroed at the start of every session."""
    session_groups = frames.groupby(
        ["SubjectName", "SessionName"], sort=False, group_keys=False
    )

    def _displacement_within_session(g: pd.DataFrame) -> pd.Series:
        dx = g["centroid_x_mm"].diff()
        dy = g["centroid_y_mm"].diff()
        disp = np.sqrt(dx**2 + dy**2)
        disp.iloc[0] = 0.0  # zero at session boundary
        return disp

    return session_groups.apply(_displacement_within_session)


def load_moseq_df(
    moseq_df: pd.DataFrame,
    group_cols: list[str],
    velocity_col: str = "velocity_3d_mm",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate, clean, and reshape a raw per-frame `moseq_df` into the three
    tables every downstream feature module consumes.
    """
    group_cols = list(group_cols)
    _validate_columns(moseq_df, group_cols)
    if velocity_col not in moseq_df.columns:
        raise ValueError(f"load_moseq_df: velocity_col {velocity_col!r} not in moseq_df")

    frames = moseq_df.loc[
        moseq_df["labels (usage sort)"] != _UNLABELED_SYLLABLE
    ].copy()

    # cross-sectional studies have no timepoint column at all
    if "timepoint" not in frames.columns:
        frames["timepoint"] = _DEFAULT_TIMEPOINT

    # StartTime (falling back to SessionName) keeps multi-session subjects in true recording order
    sort_cols = ["SubjectName"]
    if "StartTime" in frames.columns:
        sort_cols.append("StartTime")
    sort_cols += ["SessionName", "frame index"]
    frames = frames.sort_values(sort_cols).reset_index(drop=True)

    frames["frame_displacement_mm"] = _compute_frame_displacement(frames)

    frames = frames.rename(
        columns={"SubjectName": "subject_id", "labels (usage sort)": "syllable"}
    )

    # syllable_kinematics: aggregated globally across all frames
    syllable_kinematics = (
        frames.groupby("syllable")
        .agg(
            mean_velocity=(velocity_col, "mean"),
            mean_displacement=("dist_to_center_px", "mean"),
            n_frames=("syllable", "size"),
        )
        .reset_index()
    )

    # usage: long-format per-subject-per-timepoint-per-syllable
    group_keys = ["subject_id", "timepoint"] + group_cols
    usage_wide = (
        frames.groupby(group_keys)["syllable"]
        .value_counts(normalize=True)
        .rename("value")
        .reset_index()
        .rename(columns={"syllable": "feature_name"})
    )
    usage_wide["feature_name"] = usage_wide["feature_name"].astype(str)
    usage_wide["feature_type"] = "syllable"
    usage = usage_wide[group_keys + ["feature_type", "feature_name", "value"]]

    return frames, syllable_kinematics, usage
