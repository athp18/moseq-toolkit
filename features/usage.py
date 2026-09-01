
def attach_clusters(usage_df, cluster_mapping):
    if not (usage_df["feature_type"] == "syllable").all():
        raise ValueError('attach_clusters: usage_df must contain only feature_type=="syllable" rows')

    usage_df = usage_df.copy()
    usage_df["cluster"] = usage_df["feature_name"].astype(int).map(cluster_mapping)

    if usage_df.empty:
        raise ValueError("attach_clusters: usage_df is empty")
    if usage_df["cluster"].isna().all():
        raise ValueError("attach_clusters: cluster_mapping did not match any feature_name (check dtypes)")

    return usage_df.dropna(subset=["cluster"])

def aggregate_clusters(usage_with_clusters, renormalize=False):
    fixed_cols = {"subject_id", "timepoint", "feature_type", "feature_name", "value", "cluster"}
    group_cols = [c for c in usage_with_clusters.columns if c not in fixed_cols]

    subject_keys = ["subject_id", "timepoint"] + group_cols

    cluster_usage = (
        usage_with_clusters
        .groupby(subject_keys + ["cluster"], as_index=False)["value"]
        .sum()
    )

    if renormalize:
        totals = cluster_usage.groupby(subject_keys)["value"].transform("sum")
        cluster_usage["value"] = cluster_usage["value"] / totals

    cluster_usage = cluster_usage.rename(columns={"cluster": "feature_name"})
    cluster_usage["feature_type"] = "cluster"

    return cluster_usage[subject_keys + ["feature_type", "feature_name", "value"]]