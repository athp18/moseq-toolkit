import numpy as np
import pandas as pd


def _iqr_mask_outliers(frames, metric, group_cols):
    grouped = frames.groupby(group_cols)[metric]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    values = frames[metric]
    is_outlier = (values < lower) | (values > upper)
    return values.where(~is_outlier)

def compute(frames, metrics, group_cols, outlier_method=None):
    missing_metrics = [m for m in metrics if m not in frames.columns]
    if missing_metrics:
        raise ValueError(f"scalars.compute: missing metric columns: {missing_metrics}")
    if outlier_method not in (None, "iqr"):
        raise ValueError(f"scalars.compute: unsupported outlier_method {outlier_method!r}")

    working = frames.copy()
    if outlier_method == "iqr":
        for metric in metrics:
            working[metric] = _iqr_mask_outliers(working, metric, group_cols)

    subject_keys = ["subject_id", "timepoint"] + group_cols
    aggregated = working.groupby(subject_keys)[metrics].mean().reset_index()
    long = aggregated.melt(id_vars=subject_keys, value_vars=metrics,
                            var_name="feature_name", value_name="value")
    long["feature_type"] = "scalar"
    return long[subject_keys + ["feature_type", "feature_name", "value"]]


def compute_histogram(frames, metric, n_bins, group_cols, feature_type=None, bin_range=None):
    """Per-subject-per-timepoint histogram of one raw per-frame scalar, as a
    proportion vector rather than compute()'s single mean. bin_range, if not
    given, is computed globally so bins are comparable across subjects.

    Returns a long-format DataFrame: subject_id, timepoint, {group_cols},
    feature_type, feature_name ("bin_000", ...), value (that bin's fraction
    of the subject's frames, summing to 1, or all-zero if no valid frames).
    """
    if metric not in frames.columns:
        raise ValueError(f"scalars.compute_histogram: missing metric column {metric!r}")
    if n_bins < 1:
        raise ValueError(f"scalars.compute_histogram: n_bins must be >= 1, got {n_bins}")

    feature_type = feature_type or f"{metric}_histogram"

    if bin_range is None:
        all_values = frames[metric].to_numpy(dtype=float)
        all_values = all_values[~np.isnan(all_values)]
        if all_values.size == 0:
            raise ValueError(f"scalars.compute_histogram: no non-NaN values for metric {metric!r}")
        bin_range = (float(all_values.min()), float(all_values.max()))
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

    subject_keys = ["subject_id", "timepoint"] + group_cols
    records = []
    for group_key, sub in frames.groupby(subject_keys, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = sub[metric].to_numpy(dtype=float)
        values = values[~np.isnan(values)]

        counts, _ = np.histogram(values, bins=bin_edges)
        total = counts.sum()
        # zero valid frames -> all-zero histogram, not NaN/uniform
        proportions = counts / total if total > 0 else np.zeros(n_bins)

        for bin_idx, proportion in enumerate(proportions):
            record = dict(zip(subject_keys, group_key))
            record["feature_type"] = feature_type
            record["feature_name"] = f"bin_{bin_idx:03d}"
            record["value"] = float(proportion)
            records.append(record)

    return pd.DataFrame.from_records(records, columns=subject_keys + ["feature_type", "feature_name", "value"])


def compute_histograms(frames, bin_defs, group_cols):
    """Build several histogram-based scalar channels at once (via
    compute_histogram) and concatenate them into one long table.
    bin_defs: dict[channel_name -> (metric_column, n_bins)]."""
    parts = [
        compute_histogram(frames, metric, n_bins, group_cols, feature_type=channel_name)
        for channel_name, (metric, n_bins) in bin_defs.items()
    ]
    return pd.concat(parts, ignore_index=True)
