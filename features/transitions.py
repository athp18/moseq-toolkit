from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SUBJECT_COL = "subject_id"
DEFAULT_TIMEPOINT_COL = "timepoint"
DEFAULT_SYLLABLE_COL = "syllable"
DEFAULT_SESSION_COL = "SessionName"
DEFAULT_FRAME_COL = "frame index"


def _single_sequence_matrix(labels: np.ndarray, n_syllables: int) -> np.ndarray:
    """Build one row-normalized transition matrix from a single ordered label
    sequence; rows with zero outgoing transitions stay all-zero, not NaN."""
    counts = np.zeros((n_syllables, n_syllables), dtype=np.float64)

    from_labels = labels[:-1]
    to_labels = labels[1:]

    np.add.at(counts, (from_labels, to_labels), 1)

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero, leaves zero rows as zero
    return counts / row_sums


def compute_matrix(
    frames: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    n_syllables: Optional[int] = None,
    subject_col: str = DEFAULT_SUBJECT_COL,
    timepoint_col: str = DEFAULT_TIMEPOINT_COL,
    syllable_col: str = DEFAULT_SYLLABLE_COL,
    session_col: str = DEFAULT_SESSION_COL,
    frame_col: str = DEFAULT_FRAME_COL,
) -> Tuple[Dict[Tuple, np.ndarray], Dict[Tuple, dict], int]:
    """Compute one row-normalized transition matrix per (subject, timepoint).

    n_syllables sizes the matrix; defaults to the full label range (max + 1)
    if not given. A smaller n_syllables restricts coverage: any pair where
    either syllable is out of range is dropped, not bridged over.
    """
    group_cols = group_cols or []

    missing = [
        col
        for col in [subject_col, timepoint_col, syllable_col, session_col, frame_col]
        + group_cols
        if col not in frames.columns
    ]
    if missing:
        raise ValueError(f"frames is missing required columns: {missing}")

    if n_syllables is None:
        n_syllables = int(frames[syllable_col].max()) + 1
    elif n_syllables < 1:
        raise ValueError(f"compute_matrix: n_syllables must be >= 1, got {n_syllables}")

    matrices, metadata = {}, {}

    group_keys = [subject_col, timepoint_col]
    for key, sub_df in frames.groupby(group_keys, sort=False):
        counts = np.zeros((n_syllables, n_syllables))
        for _, session_df in sub_df.groupby(session_col, sort=False):
            session_df = session_df.sort_values(frame_col)
            labels = session_df[syllable_col].to_numpy(dtype=int)
            if len(labels) >= 2:
                from_labels = labels[:-1]
                to_labels = labels[1:]
                # count a pair only if both sides are within range
                in_range = (from_labels < n_syllables) & (to_labels < n_syllables)
                if in_range.any():
                    np.add.at(counts, (from_labels[in_range], to_labels[in_range]), 1)
        if counts.sum()==0:
            continue
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrices[key] = counts / row_sums
        metadata[key] = {col: sub_df[col].iloc[0] for col in group_cols}

    return matrices, metadata, n_syllables


def to_long(
    matrices: Dict[Tuple, np.ndarray],
    metadata: Dict[Tuple, dict],
    subject_col: str = DEFAULT_SUBJECT_COL,
    timepoint_col: str = DEFAULT_TIMEPOINT_COL,
    feature_type: str = "transition_pairwise",
) -> pd.DataFrame:
    """Flatten per-subject transition matrices into the shared long schema:
    subject_id, timepoint, {group_cols}, feature_type, feature_name ("{i}->{j}"),
    value. Every (i, j) cell is emitted, zeros included -- a missing cell
    would pivot to NaN downstream instead of 0.0."""
    if not matrices:
        return pd.DataFrame(columns=[subject_col, timepoint_col, "feature_type", "feature_name", "value"])

    shapes = {T.shape[0] for T in matrices.values()}
    if len(shapes) != 1:
        raise ValueError(f"to_long: matrices have inconsistent shapes: {shapes}")
    n = shapes.pop()
    i_idx, j_idx = np.indices((n, n))
    feature_names = np.array([f"{i}->{j}" for i, j in zip(i_idx.ravel(), j_idx.ravel())])
    n_cells = n * n

    all_meta_keys = sorted({k for meta in metadata.values() for k in meta})

    subject_chunks, timepoint_chunks, value_chunks = [], [], []
    meta_chunks = defaultdict(list)

    for key, T in matrices.items():
        subject_id, timepoint = key
        meta = metadata.get(key, {})
        subject_chunks.append(np.full(n_cells, subject_id, dtype=object))
        timepoint_chunks.append(np.full(n_cells, timepoint, dtype=object))
        value_chunks.append(T.ravel())
        for mk in all_meta_keys:
            meta_chunks[mk].append(np.full(n_cells, meta.get(mk), dtype=object))

    data = {
        subject_col: np.concatenate(subject_chunks),
        timepoint_col: np.concatenate(timepoint_chunks),
        "feature_type": feature_type,
        "feature_name": np.tile(feature_names, len(matrices)),
        "value": np.concatenate(value_chunks),
    }
    for mk in all_meta_keys:
        data[mk] = np.concatenate(meta_chunks[mk])

    return pd.DataFrame(data)


def collapse_to_clusters(
    matrices: Dict[Tuple, np.ndarray],
    metadata: Dict[Tuple, dict],
    cluster_mapping: Dict[int, str],
    exclude_self_cluster: bool = True,
    subject_col: str = DEFAULT_SUBJECT_COL,
    timepoint_col: str = DEFAULT_TIMEPOINT_COL,
    feature_type: str = "transition_cluster",
) -> pd.DataFrame:
    """Collapse full transition matrices down to "probability of transitioning
    into cluster X from any other cluster", one scalar per subject/timepoint/cluster."""
    cluster_to_ids: Dict[str, List[int]] = defaultdict(list)
    for syl_id, cluster_label in cluster_mapping.items():
        cluster_to_ids[cluster_label].append(syl_id)

    cluster_labels = sorted(cluster_to_ids.keys())

    records = []
    for key, T in matrices.items():
        subject_id, timepoint = key
        meta = metadata.get(key, {})
        n = T.shape[0]
        mapped_ids = np.array(sorted(cluster_mapping.keys()), dtype=int)
        mapped_ids = mapped_ids[mapped_ids < n]

        for cluster_label in cluster_labels:
            target_ids = np.array(cluster_to_ids[cluster_label], dtype=int)
            target_ids = target_ids[target_ids < n]
            if len(target_ids) == 0:
                continue

            target_set = set(target_ids.tolist())

            if exclude_self_cluster:
                source_ids = np.array(
                    [i for i in mapped_ids if i not in target_set], dtype=int
                )
            else:
                source_ids = mapped_ids

            if len(source_ids) == 0:
                continue

            sub_matrix = T[np.ix_(source_ids, target_ids)]
            p_into_cluster = sub_matrix.sum(axis=1).mean()

            record = {
                subject_col: subject_id,
                timepoint_col: timepoint,
                "feature_type": feature_type,
                "feature_name": f"into_{cluster_label}",
                "value": p_into_cluster,
            }
            record.update(meta)
            records.append(record)

    return pd.DataFrame.from_records(records)


def average_matrices(matrices, metadata, group_cols):
    """Average per-subject transition matrices across subjects into one
    matrix per unique combination of (timepoint, *group_cols)."""
    grouped = defaultdict(list)
    group_meta = {}

    for key, T in matrices.items():
        _, timepoint = key
        meta = metadata.get(key, {})
        missing = [c for c in group_cols if c not in meta]
        if missing:
            raise ValueError(f"average_matrices: metadata for {key} is missing group_cols: {missing}")

        group_key = (timepoint,) + tuple(meta[c] for c in group_cols)
        grouped[group_key].append(T)
        group_meta[group_key] = {
            "timepoint": timepoint,
            **{c: meta[c] for c in group_cols},
        }

    averaged = {}
    for group_key, mats in grouped.items():
        averaged[group_key] = np.mean(mats, axis=0)
        group_meta[group_key]["n_subjects"] = len(mats)

    return averaged, group_meta
