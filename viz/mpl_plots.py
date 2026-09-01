"""Static matplotlib/seaborn renderings of this toolkit's plots, mirroring
the interactive Plotly versions 1:1. Selected via the app's "Plot style"
control ("Notebook (matplotlib)")."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.metrics import roc_curve, auc
from statannotations.Annotator import Annotator

from viz.palette import (
    DEFAULT_CLUSTER_COLORS, DEFAULT_QUALITATIVE_PALETTE, DEFAULT_SEQUENTIAL_COLORSCALE,
    colors_for_categories,
)

# Plotly colorscale names mapped to their matplotlib cmap equivalents.
_COLORSCALE_TO_CMAP = {
    "Blues": "Blues",
    "Viridis": "viridis",
    "Magma": "magma",
    "Greens": "Greens",
    "RdBu": "RdBu",
    "YlOrRd": "YlOrRd",
}


def _dynamic_figsize(n_groups, height=6.0, base=1.8, per_group=1.2):
    """Widen a bar/strip figure as more categories are added, to avoid label overlap."""
    width = base + per_group * max(n_groups, 1)
    return (max(4.0, width), height)


def mpl_group_comparison(
    long_df, feature_name, group_col, feature_type=None, stats_result=None,
    order=None, palette_name=DEFAULT_QUALITATIVE_PALETTE, color_mapping=None,
    title=None, ylim=None, chance_level=None,
):
    """Static matplotlib/seaborn rendering of plot_group_comparison: bar
    (mean, 68% CI) + jittered strip, with significance brackets from
    already-computed p-values."""
    data = long_df[long_df["feature_name"] == feature_name]
    if feature_type is not None:
        data = data[data["feature_type"] == feature_type]
    if data.empty:
        raise ValueError(
            f"mpl_group_comparison: no rows for feature_name={feature_name!r}"
            + (f", feature_type={feature_type!r}" if feature_type else "")
        )

    categories = list(order) if order is not None else sorted(data[group_col].dropna().unique(), key=str)
    if not categories:
        raise ValueError(f"mpl_group_comparison: no non-null {group_col!r} values to compare")

    colors = colors_for_categories(categories, palette_name)
    if color_mapping:
        colors.update(color_mapping)
    palette = [colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=_dynamic_figsize(len(categories)), dpi=300)

    sns.barplot(
        data=data, x=group_col, y="value", order=categories,
        hue=group_col, palette=palette, legend=False,
        errorbar=("ci", 68), capsize=0, dodge=False, alpha=0.85, ax=ax,
    )
    sns.stripplot(
        data=data, x=group_col, y="value", order=categories,
        color="black", size=8, alpha=0.6, jitter=True, zorder=3, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(feature_name)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if chance_level is not None:
        ax.axhline(chance_level, color="gray", linestyle="--", linewidth=1, label="chance")
        ax.annotate("chance", xy=(0.01, chance_level), xycoords=("axes fraction", "data"),
                    va="bottom", ha="left", fontsize=8, color="gray")

    if stats_result is not None:
        sub_stats = stats_result[stats_result["feature_name"] == feature_name]
        if not sub_stats.empty:
            p_col = "p_adj" if "p_adj" in sub_stats.columns else "p_value"
            pairs = [
                (a, b) for a, b in zip(sub_stats["group_a"], sub_stats["group_b"])
                if a in categories and b in categories
            ]
            pvalues = [
                p for a, b, p in zip(sub_stats["group_a"], sub_stats["group_b"], sub_stats[p_col])
                if a in categories and b in categories
            ]
            if pairs:
                annotator = Annotator(ax, pairs, data=data, x=group_col, y="value", order=categories, plot="barplot")
                annotator.configure(test=None, text_format="star", loc="outside", verbose=0, line_width=0.8, fontsize=10)
                annotator.set_pvalues_and_annotate(pvalues)

    if title:
        ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def mpl_cluster_locations(
    syllable_kinematics, cluster_col="cluster", id_col="syllable",
    velocity_col="mean_velocity", displacement_col="mean_displacement",
    palette_name=DEFAULT_QUALITATIVE_PALETTE, title=None,
):
    """Static matplotlib rendering of plot_cluster_locations: scatter of
    every syllable at (velocity, displacement), colored by cluster, with a
    convex hull and star centroid per cluster."""
    missing = {cluster_col, id_col, velocity_col, displacement_col} - set(syllable_kinematics.columns)
    if missing:
        raise ValueError(f"mpl_cluster_locations: syllable_kinematics missing columns {missing}")

    data = syllable_kinematics.dropna(subset=[cluster_col])
    if data.empty:
        raise ValueError("mpl_cluster_locations: no rows with a non-null cluster assignment")

    clusters = sorted(data[cluster_col].unique(), key=str)
    colors = colors_for_categories(clusters, palette_name, DEFAULT_CLUSTER_COLORS)

    fig, ax = plt.subplots(figsize=(6, 7.5), dpi=300)

    # hulls drawn first so they sit behind the points
    for cluster in clusters:
        group = data[data[cluster_col] == cluster]
        points = group[[velocity_col, displacement_col]].to_numpy()
        if len(points) < 3:
            continue
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        poly = mpatches.Polygon(hull_points, closed=True, facecolor=colors[cluster], edgecolor=colors[cluster],
                                 alpha=0.15, linewidth=1, zorder=1)
        ax.add_patch(poly)

    for cluster in clusters:
        group = data[data[cluster_col] == cluster]
        label = str(cluster).replace("_", " ")
        ax.scatter(
            group[velocity_col], group[displacement_col],
            s=60, color=colors[cluster], alpha=0.8, edgecolor="white", linewidth=0.5,
            label=label, zorder=2,
        )
        for _, row in group.iterrows():
            ax.annotate(str(row[id_col]), (row[velocity_col], row[displacement_col]),
                        textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center", zorder=3)

    for cluster in clusters:
        group = data[data[cluster_col] == cluster]
        points = group[[velocity_col, displacement_col]].to_numpy()
        if len(points) < 3:
            continue
        centroid = points.mean(axis=0)
        ax.scatter(*centroid, marker="*", s=220, color=colors[cluster], edgecolor="black", linewidth=0.8, zorder=4)

    ax.set_xlabel(f"Mean velocity ({velocity_col})")
    ax.set_ylabel(f"Mean displacement ({displacement_col})")
    if title:
        ax.set_title(title)
    ax.legend(title="Cluster", frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def mpl_transition_heatmap(matrix, labels=None, colorscale=DEFAULT_SEQUENTIAL_COLORSCALE, title=None, **kwargs):
    """Static matplotlib/seaborn rendering of plot_transition_heatmap.
    **kwargs are forwarded to sns.heatmap."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"mpl_transition_heatmap: expected a square 2D matrix, got shape {matrix.shape}")

    n = matrix.shape[0]
    if labels is None:
        labels = list(range(n))
    elif len(labels) != n:
        raise ValueError(f"mpl_transition_heatmap: labels has length {len(labels)}, expected {n}")
    labels = [str(l) for l in labels]

    heatmap_kwargs = dict(vmin=0, vmax=float(np.nanmax(matrix)) if matrix.size else 1.0)
    heatmap_kwargs.update(kwargs)

    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    sns.heatmap(
        matrix, xticklabels=labels, yticklabels=labels,
        cmap=_COLORSCALE_TO_CMAP.get(colorscale, "Blues"),
        square=True, linewidths=0, cbar_kws=dict(label="transition probability"),
        ax=ax, **heatmap_kwargs,
    )
    ax.set_xlabel("to syllable")
    ax.set_ylabel("from syllable")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def mpl_confusion_matrix(run_result, colorscale=DEFAULT_SEQUENTIAL_COLORSCALE, title=None, **kwargs):
    """Static matplotlib/seaborn rendering of plot_confusion_matrix."""
    missing = {"confusion_matrix", "classes"} - set(run_result.keys())
    if missing:
        raise ValueError(f"mpl_confusion_matrix: run_result missing keys {missing}")

    cm = np.asarray(run_result["confusion_matrix"])
    classes = [str(c) for c in run_result["classes"]]
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(
            f"mpl_confusion_matrix: confusion_matrix shape {cm.shape} doesn't match "
            f"{len(classes)} classes"
        )

    heatmap_kwargs = dict(vmin=0, vmax=1)
    heatmap_kwargs.update(kwargs)

    fig, ax = plt.subplots(figsize=(5.5, 6.5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes,
        cmap=_COLORSCALE_TO_CMAP.get(colorscale, "Blues"),
        square=True, linewidths=0, cbar_kws=dict(label="fraction of true class"),
        ax=ax, **heatmap_kwargs,
    )
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def mpl_roc(run_result, palette_name=DEFAULT_QUALITATIVE_PALETTE, title=None):
    """Static matplotlib rendering of plot_roc: binary (one curve) or
    multiclass (one-vs-rest per class), each labeled with its own AUC."""
    missing = {"oof_true", "oof_proba", "classes"} - set(run_result.keys())
    if missing:
        raise ValueError(f"mpl_roc: run_result missing keys {missing}")

    oof_true = np.asarray(run_result["oof_true"])
    oof_proba = np.asarray(run_result["oof_proba"])
    classes = run_result["classes"]
    n_classes = len(classes)

    if oof_proba.shape != (len(oof_true), n_classes):
        raise ValueError(
            f"mpl_roc: oof_proba shape {oof_proba.shape} doesn't match "
            f"({len(oof_true)}, {n_classes})"
        )

    class_labels = [str(c) for c in classes]
    colors = colors_for_categories(class_labels, palette_name)

    fig, ax = plt.subplots(figsize=(5.5, 6.5), dpi=300)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(oof_true, oof_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[class_labels[1]], label=f"{class_labels[1]} (AUC = {roc_auc:.2f})")
    else:
        for i, cls in enumerate(class_labels):
            y_true_binary = (oof_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, oof_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[cls], label=f"{cls} vs rest (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def mpl_feature_type_comparison(
    compare_result, metric="macro_f1", style="bar", order=None,
    palette_name=DEFAULT_QUALITATIVE_PALETTE, title=None, ylim=None, chance_level=None,
):
    """Static matplotlib/seaborn rendering of plot_feature_type_comparison:
    bar+strip or box+strip comparison across named feature sets, with
    brackets from compare_result["pairwise_comparisons"]."""
    missing = {"results", "pairwise_comparisons"} - set(compare_result.keys())
    if missing:
        raise ValueError(f"mpl_feature_type_comparison: compare_result missing keys {missing}")
    if style not in ("bar", "box"):
        raise ValueError(f"mpl_feature_type_comparison: style must be 'bar' or 'box', got {style!r}")

    results = compare_result["results"]
    names = list(order) if order is not None else list(results.keys())
    unknown = [n for n in names if n not in results]
    if unknown:
        raise ValueError(f"mpl_feature_type_comparison: order includes unknown names {unknown}")

    colors = colors_for_categories(names, palette_name)
    palette = [colors[n] for n in names]

    rows = []
    for name in names:
        fold_scores = results[name]["fold_scores"]
        if metric not in fold_scores:
            raise ValueError(f"mpl_feature_type_comparison: metric {metric!r} not in fold_scores for {name!r}")
        for score in fold_scores[metric]:
            rows.append({"feature_set": name, "score": score})
    import pandas as pd
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=_dynamic_figsize(len(names)), dpi=300)

    if style == "bar":
        sns.barplot(
            data=plot_df, x="feature_set", y="score", order=names,
            hue="feature_set", palette=palette, legend=False,
            errorbar=("ci", 68), capsize=0, dodge=False, alpha=0.85, ax=ax,
        )
    else:
        sns.boxplot(
            data=plot_df, x="feature_set", y="score", order=names,
            hue="feature_set", palette=palette, legend=False, dodge=False, ax=ax,
        )
    sns.stripplot(
        data=plot_df, x="feature_set", y="score", order=names,
        color="black", size=8, alpha=0.6, jitter=True, zorder=3, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if chance_level is not None:
        ax.axhline(chance_level, color="gray", linestyle="--", linewidth=1)
        ax.annotate("chance", xy=(0.01, chance_level), xycoords=("axes fraction", "data"),
                    va="bottom", ha="left", fontsize=8, color="gray")

    pairwise = compare_result["pairwise_comparisons"]
    if not pairwise.empty:
        relevant = pairwise[pairwise["group_a"].isin(names) & pairwise["group_b"].isin(names)]
        if not relevant.empty:
            pairs = list(zip(relevant["group_a"], relevant["group_b"]))
            pvalues = relevant["p_value"].tolist()
            annotator = Annotator(ax, pairs, data=plot_df, x="feature_set", y="score", order=names,
                                   plot="barplot" if style == "bar" else "boxplot")
            annotator.configure(test=None, text_format="star", loc="inside", verbose=0, line_width=0.8, fontsize=10)
            annotator.set_pvalues_and_annotate(pvalues)

    if title:
        ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
