import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from viz.palette import (
    DEFAULT_QUALITATIVE_PALETTE, DEFAULT_SEQUENTIAL_COLORSCALE, DEFAULT_TEMPLATE,
    colors_for_categories, foreground_for_template,
)
from viz.stats_brackets import add_significance_brackets
from viz.mpl_style import add_chance_level


def plot_confusion_matrix(
    run_result, colorscale=DEFAULT_SEQUENTIAL_COLORSCALE, title=None, template=DEFAULT_TEMPLATE,
    **kwargs,
):
    """Render run()'s pooled out-of-fold confusion matrix (row-normalized to
    "true") as an interactive Plotly heatmap."""
    missing = {"confusion_matrix", "classes"} - set(run_result.keys())
    if missing:
        raise ValueError(f"plot_confusion_matrix: run_result missing keys {missing}")

    cm = np.asarray(run_result["confusion_matrix"])
    classes = [str(c) for c in run_result["classes"]]
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(
            f"plot_confusion_matrix: confusion_matrix shape {cm.shape} doesn't match "
            f"{len(classes)} classes"
        )

    heatmap_kwargs = dict(zmin=0, zmax=1)
    heatmap_kwargs.update(kwargs)

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=classes, y=classes, colorscale=colorscale,
        text=np.round(cm, 2), texttemplate="%{text}",
        colorbar=dict(title="fraction of<br>true class"),
        hovertemplate="true: %{y}<br>predicted: %{x}<br>fraction: %{z:.3f}<extra></extra>",
        **heatmap_kwargs,
    ))
    fig.update_layout(
        xaxis=dict(title="predicted class", side="bottom"),
        yaxis=dict(title="true class", autorange="reversed"),
        template=template,
        title=title,
        height=500,
        width=520,
    )
    return fig


def plot_roc(
    run_result, palette_name=DEFAULT_QUALITATIVE_PALETTE, title=None, template=DEFAULT_TEMPLATE,
):
    """Interactive ROC curve from run()'s pooled out-of-fold predictions:
    one curve for binary classification, one-vs-rest per class for
    multiclass (each labeled with its own AUC)."""
    missing = {"oof_true", "oof_proba", "classes"} - set(run_result.keys())
    if missing:
        raise ValueError(f"plot_roc: run_result missing keys {missing}")

    oof_true = np.asarray(run_result["oof_true"])
    oof_proba = np.asarray(run_result["oof_proba"])
    classes = run_result["classes"]
    n_classes = len(classes)

    if oof_proba.shape != (len(oof_true), n_classes):
        raise ValueError(
            f"plot_roc: oof_proba shape {oof_proba.shape} doesn't match "
            f"({len(oof_true)}, {n_classes})"
        )

    class_labels = [str(c) for c in classes]
    colors = colors_for_categories(class_labels, palette_name)

    fig = go.Figure()

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(oof_true, oof_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", line=dict(color=colors[class_labels[1]]),
            name=f"{class_labels[1]} (AUC = {roc_auc:.2f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
    else:
        for i, cls in enumerate(class_labels):
            y_true_binary = (oof_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, oof_proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", line=dict(color=colors[cls]),
                name=f"{cls} vs rest (AUC = {roc_auc:.2f})",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        xaxis=dict(title="false positive rate", range=[-0.02, 1.02]),
        yaxis=dict(title="true positive rate", range=[-0.02, 1.02]),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom", font=dict(size=10)),
        template=template,
        title=title,
        height=500,
        width=520,
    )
    return fig


def plot_feature_type_comparison(
    compare_result, metric="macro_f1", style="bar",
    order=None, palette_name=DEFAULT_QUALITATIVE_PALETTE, title=None, template=DEFAULT_TEMPLATE,
    ylim=None, chance_level=None,
):
    """Render classify.classifier.compare_feature_types()'s output as an
    interactive fold-level score comparison across named feature sets, with
    significance brackets from its pairwise_comparisons.

    style: "bar" (mean + 68% bootstrap CI, for a couple of feature sets) or
    "box" (reads better for 4-5+ sets). ylim/chance_level optionally
    override the y-axis range / add a dashed reference line.
    """
    missing = {"results", "pairwise_comparisons"} - set(compare_result.keys())
    if missing:
        raise ValueError(f"plot_feature_type_comparison: compare_result missing keys {missing}")
    if style not in ("bar", "box"):
        raise ValueError(f"plot_feature_type_comparison: style must be 'bar' or 'box', got {style!r}")

    results = compare_result["results"]
    names = list(order) if order is not None else list(results.keys())
    unknown = [n for n in names if n not in results]
    if unknown:
        raise ValueError(f"plot_feature_type_comparison: order includes unknown names {unknown}")

    position = {name: i for i, name in enumerate(names)}
    colors = colors_for_categories(names, palette_name)
    fg_color = foreground_for_template(template)

    scores_by_name = {}
    for name in names:
        fold_scores = results[name]["fold_scores"]
        if metric not in fold_scores:
            raise ValueError(
                f"plot_feature_type_comparison: metric {metric!r} not in fold_scores for {name!r}"
            )
        scores_by_name[name] = np.asarray(fold_scores[metric], dtype=float)

    fig = go.Figure()
    y_top_per_category = {}
    all_values = []

    if style == "bar":
        from viz.usage_plots import _bootstrap_ci

        bar_x, bar_y, err_lo, err_hi, bar_colors = [], [], [], [], []
        for name in names:
            values = scores_by_name[name]
            mean, lo, hi = _bootstrap_ci(values)
            bar_x.append(position[name])
            bar_y.append(mean)
            err_lo.append(0.0 if np.isnan(mean) else mean - lo)
            err_hi.append(0.0 if np.isnan(mean) else hi - mean)
            bar_colors.append(colors[name])
            finite_top = [v for v in list(values) + [hi] if not np.isnan(v)]
            y_top_per_category[name] = max(finite_top) if finite_top else 0.0
            all_values.extend(v for v in values if not np.isnan(v))

        fig.add_trace(go.Bar(
            x=bar_x, y=bar_y,
            error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, color=fg_color, thickness=3.5, width=0),
            marker=dict(color=bar_colors, opacity=0.85),
            showlegend=False,
            hovertemplate="mean: %{y:.3f}<extra></extra>",
        ))
    else:
        for name in names:
            values = scores_by_name[name]
            fig.add_trace(go.Box(
                x=[position[name]] * len(values), y=values,
                marker=dict(color=colors[name]), line=dict(color=colors[name]),
                boxpoints=False, showlegend=False,
                hovertemplate="%{y:.3f}<extra></extra>",
            ))
            y_top_per_category[name] = float(np.max(values)) if len(values) else 0.0
            all_values.extend(values.tolist())

    jitter_rng = np.random.default_rng(0)
    for name in names:
        values = scores_by_name[name]
        if values.size == 0:
            continue
        jitter = (jitter_rng.random(values.size) - 0.5) * 0.3
        fig.add_trace(go.Scatter(
            x=position[name] + jitter, y=values,
            mode="markers", marker=dict(color=fg_color, size=9, opacity=0.65, line=dict(width=0.5, color=fg_color)),
            showlegend=False, hovertemplate="%{y:.3f}<extra></extra>",
        ))

    y_min = min(all_values) if all_values else 0.0
    y_max = max(all_values) if all_values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    axis_top = y_max + (y_max - y_min) * 0.1

    pairwise = compare_result["pairwise_comparisons"]
    if not pairwise.empty:
        relevant = pairwise[pairwise["group_a"].isin(names) & pairwise["group_b"].isin(names)]
        if not relevant.empty:
            pairs = list(zip(relevant["group_a"], relevant["group_b"]))
            pvalues = relevant["p_value"].tolist()
            axis_top = add_significance_brackets(
                fig, pairs, pvalues, names, y_top_per_category, (y_min, y_max), line_color=fg_color,
            )
            axis_top += (y_max - y_min) * 0.08

    add_chance_level(fig, chance_level)

    yaxis_range = list(ylim) if ylim is not None else [y_min - (y_max - y_min) * 0.08, axis_top]

    fig.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=list(range(len(names))), ticktext=names,
            title="", tickangle=-20,
        ),
        yaxis=dict(title=metric, range=yaxis_range),
        template=template,
        title=title,
        height=480,
        showlegend=False,
    )
    return fig
