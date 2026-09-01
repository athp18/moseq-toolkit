from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ALR_EPSILON = 1e-6


def _balanced_accuracy(y_true, y_pred):
    """Mean per-class recall, same as sklearn's balanced_accuracy_score but without its warning."""
    C = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.diag(C) / C.sum(axis=1)
    per_class_recall = per_class_recall[~np.isnan(per_class_recall)]
    return float(np.mean(per_class_recall))


def build_matrix(
    long_df: pd.DataFrame,
    feature_types: Sequence[str],
    subject_col: str = "subject_id",
    timepoint_col: str = "timepoint",
    feature_names: Sequence | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    """Reshape the shared long schema into one wide feature matrix.

    Returns (X, column_feature_types, metadata): X is one row per
    (subject_col, timepoint_col) with one column per (feature_type,
    feature_name) pair (missing combinations filled with 0.0); column
    names are prefixed "{feature_type}::{feature_name}" only when
    multiple feature_types are requested. column_feature_types maps each
    X column back to its source feature_type. metadata is
    [subject_col, timepoint_col], row-aligned with X.
    """
    missing = {subject_col, timepoint_col, "feature_type", "feature_name", "value"} - set(long_df.columns)
    if missing:
        raise ValueError(f"build_matrix: long_df missing columns {missing}")

    sub = long_df[long_df["feature_type"].isin(feature_types)]
    if feature_names is not None:
        wanted = {str(f) for f in feature_names}
        sub = sub[sub["feature_name"].astype(str).isin(wanted)]
    if sub.empty:
        raise ValueError(
            f"build_matrix: no rows found for feature_types={list(feature_types)}"
            + (f", feature_names={list(feature_names)}" if feature_names is not None else "")
        )

    multiple_types = sub["feature_type"].nunique() > 1

    pivoted = sub.pivot_table(
        index=[subject_col, timepoint_col],
        columns=["feature_type", "feature_name"],
        values="value",
        aggfunc="first",
    )
    pivoted = pivoted.fillna(0.0)

    column_feature_types: Dict[str, str] = {}
    new_columns = []
    for ftype, fname in pivoted.columns:
        col_name = f"{ftype}::{fname}" if multiple_types else str(fname)
        new_columns.append(col_name)
        column_feature_types[col_name] = ftype
    pivoted.columns = new_columns

    metadata = pivoted.index.to_frame(index=False)[[subject_col, timepoint_col]]
    X = pivoted.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    return X, column_feature_types, metadata


def complement_feature_names(
    long_df: pd.DataFrame,
    feature_type: str,
    exclude: Sequence,
) -> List[str]:
    """Every feature_name present under `feature_type` in long_df except those in `exclude`.
    Returns a sorted list of feature_name strings."""
    available = set(long_df["feature_type"].unique())
    if feature_type not in available:
        raise ValueError(f"complement_feature_names: feature_type {feature_type!r} not found in long_df")

    exclude_set = {str(x) for x in exclude}
    all_names = long_df.loc[long_df["feature_type"] == feature_type, "feature_name"].astype(str).unique()
    return sorted(n for n in all_names if n not in exclude_set)


def _alr_block(block: pd.DataFrame, epsilon: float = ALR_EPSILON) -> pd.DataFrame:
    """Additive log-ratio transform of one compositional block (e.g. syllable
    usage fractions). Reference column is the lexicographically last column
    name; epsilon avoids -inf/divide-by-zero on exact zeros."""
    if block.shape[1] <= 1:
        return block.copy()

    reference_col = sorted(block.columns)[-1]
    numerator_cols = [c for c in block.columns if c != reference_col]

    shifted = block + epsilon
    ref = shifted[reference_col]

    out = pd.DataFrame(index=block.index)
    for c in numerator_cols:
        out[f"{c}__alr_vs_{reference_col}"] = np.log(shifted[c] / ref)

    return out


_HISTOGRAM_BIN_RE = re.compile(r"^bin_\d+$")


def _bare_column_name(col: str) -> str:
    """Strip an optional '{feature_type}::' prefix build_matrix adds when multiple feature_types are combined."""
    return col.split("::", 1)[1] if "::" in col else col


def _is_histogram_block(cols: List[str]) -> bool:
    """True if every column matches features.scalars.compute_histogram's
    "bin_NNN" naming, i.e. this feature_type is a histogram channel."""
    return len(cols) > 0 and all(_HISTOGRAM_BIN_RE.match(_bare_column_name(c)) for c in cols)


def transform_matrix(
    X: pd.DataFrame, column_feature_types: Dict[str, str]
) -> pd.DataFrame:
    """Apply the ALR transform to compositional feature blocks (syllable/cluster
    columns as one block; transition_pairwise columns grouped and ALR'd per
    source syllable; histogram channels as one block each); everything else
    passes through unchanged. Column order in the output is not guaranteed
    to match the input.
    """
    by_type: Dict[str, List[str]] = {}
    for col, ftype in column_feature_types.items():
        by_type.setdefault(ftype, []).append(col)

    out_blocks = []
    for ftype, cols in by_type.items():
        block = X[cols]
        if ftype in ("syllable", "cluster"):
            out_blocks.append(_alr_block(block))
        elif ftype == "transition_pairwise":
            by_source: Dict[str, List[str]] = {}
            for col in cols:
                bare = _bare_column_name(col)
                if "->" not in bare:
                    raise ValueError(
                        f"transform_matrix: transition_pairwise column {col!r} "
                        "is not in 'i->j' format"
                    )
                source, _, _dest = bare.partition("->")
                by_source.setdefault(source, []).append(col)
            for source, source_cols in by_source.items():
                out_blocks.append(_alr_block(block[source_cols]))
        elif _is_histogram_block(cols):
            out_blocks.append(_alr_block(block))
        else:
            out_blocks.append(block)

    return pd.concat(out_blocks, axis=1)


def _cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    cv_folds: int,
    regularization: str,
    random_state: int,
) -> Dict:
    """Run one StratifiedGroupKFold CV pass, pooling out-of-fold predictions
    for macro-AUROC/confusion matrix, and averaging per-fold scores otherwise."""
    skf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_scores = {"macro_f1": [], "balanced_accuracy": [], "precision_macro": [], "recall_macro": []}
    oof_true, oof_pred, oof_proba = [], [], []

    for train_idx, test_idx in skf.split(X, y, groups):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty=regularization, solver="saga",
                class_weight="balanced", max_iter=5000, random_state=random_state,
            )),
        ])
        pipeline.fit(X[train_idx], y[train_idx])

        y_pred = pipeline.predict(X[test_idx])
        y_true = y[test_idx]

        fold_scores["macro_f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        fold_scores["balanced_accuracy"].append(_balanced_accuracy(y_true, y_pred))
        fold_scores["precision_macro"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        fold_scores["recall_macro"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))

        proba_fold = pipeline.predict_proba(X[test_idx])
        # remap fold-local classes_ order onto full 0..n_classes-1 columns
        proba_full = np.zeros((proba_fold.shape[0], n_classes))
        for local_idx, cls in enumerate(pipeline.named_steps["clf"].classes_):
            proba_full[:, cls] = proba_fold[:, local_idx]

        oof_true.append(y_true)
        oof_pred.append(y_pred)
        oof_proba.append(proba_full)

    oof_true = np.concatenate(oof_true)
    oof_pred = np.concatenate(oof_pred)
    oof_proba = np.concatenate(oof_proba, axis=0)

    try:
        if n_classes == 2:
            auroc_macro = roc_auc_score(oof_true, oof_proba[:, 1])
        else:
            auroc_macro = roc_auc_score(oof_true, oof_proba, multi_class="ovr", average="macro")
    except ValueError:
        auroc_macro = np.nan

    cm = confusion_matrix(oof_true, oof_pred, normalize="true")

    return {
        "fold_scores": fold_scores,
        "mean_scores": {k: float(np.mean(v)) for k, v in fold_scores.items()} | {"auroc_macro": float(auroc_macro) if not np.isnan(auroc_macro) else np.nan},
        "confusion_matrix": cm,
        "oof_true": oof_true,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
    }


def run(
    X: pd.DataFrame,
    y: Sequence,
    subject_ids: Sequence,
    cv_folds: int = 10,
    regularization: str = "l2",
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Dict:
    """Grouped cross-validated logistic regression with a label-permutation
    significance test. subject_ids groups folds so no subject appears in
    both train and test.

    Returns a dict with fold_scores, mean_scores, confusion_matrix, classes,
    permutation_p_value, null_distribution, and the pooled out-of-fold
    oof_true/oof_proba (for plotting an ROC curve without re-running CV).
    """
    if regularization not in ("l1", "l2"):
        raise ValueError(f"run: regularization must be 'l1' or 'l2', got {regularization!r}")

    X_arr = np.asarray(X)
    subject_ids = np.asarray(subject_ids)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y))
    n_classes = len(encoder.classes_)

    observed = _cv_scores(X_arr, y_encoded, subject_ids, n_classes, cv_folds, regularization, random_state)

    rng = np.random.default_rng(random_state)
    null_distribution = np.empty(n_permutations)
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y_encoded)
        permuted = _cv_scores(X_arr, y_shuffled, subject_ids, n_classes, cv_folds, regularization, random_state)
        null_distribution[i] = permuted["mean_scores"]["macro_f1"]

    observed_score = observed["mean_scores"]["macro_f1"]
    p_value = (np.sum(null_distribution >= observed_score) + 1) / (n_permutations + 1)

    return {
        "fold_scores": observed["fold_scores"],
        "mean_scores": observed["mean_scores"],
        "confusion_matrix": observed["confusion_matrix"],
        "classes": list(encoder.classes_),
        "permutation_p_value": float(p_value),
        "null_distribution": null_distribution,
        "oof_true": observed["oof_true"],
        "oof_proba": observed["oof_proba"],
    }


def compare_feature_types(
    long_df: pd.DataFrame,
    feature_type_sets: Dict[str, object],
    y_col: str,
    subject_col: str = "subject_id",
    timepoint_col: str = "timepoint",
    metric: str = "macro_f1",
    **run_kwargs,
) -> Dict:
    """Run build_matrix -> transform_matrix -> run() once per named feature
    set in feature_type_sets (each a list of feature_types, or a dict with
    an optional feature_names restriction), then pairwise-compare their
    per-fold `metric` scores with Mann-Whitney U.

    Returns a dict with `results` ({name: run() output}) and
    `pairwise_comparisons` (DataFrame [group_a, group_b, statistic, p_value]).
    """
    # y_col may itself be timepoint_col (e.g. classifying by timepoint) --
    # dedupe the column list so that doesn't select "timepoint" twice.
    label_cols = [subject_col, timepoint_col] if y_col == timepoint_col else [subject_col, timepoint_col, y_col]
    labels = long_df[label_cols].drop_duplicates()

    results = {}
    for name, spec in feature_type_sets.items():
        if isinstance(spec, dict):
            ftypes = spec["feature_types"]
            fnames = spec.get("feature_names")
        else:
            ftypes = spec
            fnames = None

        X, column_feature_types, metadata = build_matrix(
            long_df, ftypes, subject_col, timepoint_col, feature_names=fnames
        )
        X_transformed = transform_matrix(X, column_feature_types)

        merged = metadata.merge(labels, on=[subject_col, timepoint_col], how="left")
        if merged[y_col].isna().any():
            raise ValueError(f"compare_feature_types: {y_col} missing for some rows in feature set {name!r}")

        results[name] = run(
            X_transformed, merged[y_col].to_numpy(), merged[subject_col].to_numpy(), **run_kwargs
        )

    records = []
    for name_a, name_b in combinations(results.keys(), 2):
        scores_a = results[name_a]["fold_scores"][metric]
        scores_b = results[name_b]["fold_scores"][metric]
        stat, p = mannwhitneyu(scores_a, scores_b)
        records.append(dict(group_a=name_a, group_b=name_b, statistic=stat, p_value=p))

    return {"results": results, "pairwise_comparisons": pd.DataFrame.from_records(records)}
