"""Group comparison over the shared long schema. Branches on normality
(Shapiro-Wilk, per group) and group count, then applies a single
Benjamini-Hochberg FDR pass across every p-value produced by the call."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


def infer_paired(long_df: pd.DataFrame, group_col: str) -> str:
    """"paired" if every subject_id appears at every distinct value of
    group_col; "unpaired" otherwise."""
    groups = long_df[group_col].unique()
    if len(groups) < 2:
        return "unpaired"

    subject_sets = [set(long_df.loc[long_df[group_col] == g, "subject_id"]) for g in groups]
    common = set.intersection(*subject_sets)
    union = set.union(*subject_sets)

    return "paired" if (len(union) > 0 and common == union) else "unpaired"


def _shapiro_all_normal(group_values: dict, alpha: float) -> bool:
    """True only if every group's Shapiro-Wilk test passes (p > alpha)."""
    for values in group_values.values():
        _, p = sps.shapiro(values)
        if p <= alpha:
            return False
    return True


def compare_groups(
    long_df: pd.DataFrame,
    group_col: str,
    feature_types: list[str] | None = None,
    paired: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare `value` across group_col's categories, once per distinct
    (feature_type, feature_name) in long_df. `paired` is only honored
    per-feature if that feature's subject overlap across group_col is
    actually complete (see infer_paired), else falls back to unpaired.

    Returns a DataFrame with columns: feature_type, feature_name, test,
    group_a, group_b, statistic, p_value, p_adj, omnibus_p, n_a, n_b --
    one row per pairwise comparison (post-hoc pairs for >2 groups, with
    the omnibus test's p-value repeated in omnibus_p). p_adj is one
    BH-FDR pass across every row this call produces.
    """
    df = long_df if feature_types is None else long_df[long_df["feature_type"].isin(feature_types)]

    records = []
    for (ftype, fname), sub in df.groupby(["feature_type", "feature_name"]):
        groups = sorted(sub[group_col].unique())
        if len(groups) < 2:
            continue

        use_paired = paired and infer_paired(sub, group_col) == "paired"

        group_rows = {g: sub.loc[sub[group_col] == g, ["subject_id", "value"]] for g in groups}
        group_values = {g: rows["value"].to_numpy() for g, rows in group_rows.items()}
        is_normal = _shapiro_all_normal(group_values, alpha)

        if len(groups) == 2:
            g_a, g_b = groups
            if use_paired:
                merged = group_rows[g_a].merge(group_rows[g_b], on="subject_id", suffixes=("_a", "_b"))
                x_a, x_b = merged["value_a"].to_numpy(), merged["value_b"].to_numpy()
                if is_normal:
                    stat, p = sps.ttest_rel(x_a, x_b)
                    test_name = "paired t-test"
                else:
                    stat, p = sps.wilcoxon(x_a, x_b)
                    test_name = "Wilcoxon signed-rank"
            else:
                x_a, x_b = group_values[g_a], group_values[g_b]
                if is_normal:
                    stat, p = sps.ttest_ind(x_a, x_b)
                    test_name = "independent t-test"
                else:
                    stat, p = sps.mannwhitneyu(x_a, x_b)
                    test_name = "Mann-Whitney U"

            records.append(dict(
                feature_type=ftype, feature_name=fname, test=test_name,
                group_a=g_a, group_b=g_b, statistic=stat, p_value=p,
                omnibus_p=np.nan, n_a=len(x_a), n_b=len(x_b),
            ))

        else:
            if use_paired:
                wide = sub.pivot(index="subject_id", columns=group_col, values="value").dropna()
                if is_normal:
                    long_for_rm = wide.reset_index().melt(
                        id_vars="subject_id", var_name=group_col, value_name="value"
                    )
                    aov = AnovaRM(long_for_rm, "value", "subject_id", within=[group_col]).fit()
                    omnibus_p = float(aov.anova_table["Pr > F"].iloc[0])
                    posthoc = lambda a, b: sps.ttest_rel(a, b)
                    posthoc_name = "paired t-test (post-hoc)"
                else:
                    stat, omnibus_p = sps.friedmanchisquare(*[wide[g].to_numpy() for g in groups])
                    posthoc = lambda a, b: sps.wilcoxon(a, b)
                    posthoc_name = "Wilcoxon signed-rank (post-hoc)"

                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        a, b = wide[groups[i]].to_numpy(), wide[groups[j]].to_numpy()
                        stat_p, p_p = posthoc(a, b)
                        records.append(dict(
                            feature_type=ftype, feature_name=fname, test=posthoc_name,
                            group_a=groups[i], group_b=groups[j], statistic=stat_p, p_value=p_p,
                            omnibus_p=omnibus_p, n_a=len(a), n_b=len(b),
                        ))
            else:
                arrs = [group_values[g] for g in groups]
                if is_normal:
                    stat, omnibus_p = sps.f_oneway(*arrs)
                    posthoc = lambda a, b: sps.ttest_ind(a, b)
                    posthoc_name = "independent t-test (post-hoc)"
                else:
                    stat, omnibus_p = sps.kruskal(*arrs)
                    posthoc = lambda a, b: sps.mannwhitneyu(a, b)
                    posthoc_name = "Mann-Whitney U (post-hoc)"

                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        a, b = group_values[groups[i]], group_values[groups[j]]
                        stat_p, p_p = posthoc(a, b)
                        records.append(dict(
                            feature_type=ftype, feature_name=fname, test=posthoc_name,
                            group_a=groups[i], group_b=groups[j], statistic=stat_p, p_value=p_p,
                            omnibus_p=omnibus_p, n_a=len(a), n_b=len(b),
                        ))

    result = pd.DataFrame.from_records(records)
    if not result.empty:
        _, p_adj, _, _ = multipletests(result["p_value"], alpha=alpha, method="fdr_bh")
        result["p_adj"] = p_adj

    return result
