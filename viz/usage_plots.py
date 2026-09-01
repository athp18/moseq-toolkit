from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from viz.palette import DEFAULT_QUALITATIVE_PALETTE, DEFAULT_TEMPLATE, colors_for_categories, foreground_for_template
from viz.stats_brackets import add_significance_brackets
from viz.mpl_style import add_chance_level
from ordering import natural_sort


def _bootstrap_ci(values, ci=68, n_boot=1000, seed=0):
    """Bootstrap mean + `ci`% confidence interval, with a fixed seed for
    reproducibility."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(values.mean()), float(lower), float(upper)


def plot_group_comparison(
    long_df,
    feature_name,
    group_col,
    feature_type=None,
    stats_result=None,
    order=None,
    palette_name=DEFAULT_QUALITATIVE_PALETTE,
    color_mapping=None,
    title=None,
    save_path=None,
    template=DEFAULT_TEMPLATE,
    ylim=None,
    chance_level=None,
):
    """Interactive bar (mean, 68% bootstrap CI) + jittered strip comparing
    one feature_name across group_col's categories. `stats_result`, if
    given (stats.compare_groups()'s output shape), draws significance
    brackets from its already-computed p-values (p_adj preferred over
    p_value) rather than re-running a test. `ylim`/`chance_level`
    optionally override the y-axis range / add a dashed reference line.
    """
    data = long_df[long_df["feature_name"] == feature_name]
    if feature_type is not None:
        data = data[data["feature_type"] == feature_type]

    if data.empty:
        raise ValueError(
            f"plot_group_comparison: no rows for feature_name={feature_name!r}"
            + (f", feature_type={feature_type!r}" if feature_type else "")
        )

    categories = list(order) if order is not None else natural_sort(data[group_col].dropna().unique())
    if not categories:
        raise ValueError(f"plot_group_comparison: no non-null {group_col!r} values to compare")
    position = {cat: i for i, cat in enumerate(categories)}

    colors = colors_for_categories(categories, palette_name)
    if color_mapping:
        colors.update(color_mapping)

    has_subject_col = "subject_id" in data.columns
    fg_color = foreground_for_template(template)

    fig = go.Figure()

    bar_x, bar_y, err_lo, err_hi, bar_colors = [], [], [], [], []
    y_top_per_category = {}
    all_values = []

    for cat in categories:
        sub = data[data[group_col] == cat]
        values = sub["value"].to_numpy()
        mean, lo, hi = _bootstrap_ci(values)
        bar_x.append(position[cat])
        bar_y.append(mean)
        err_lo.append(0.0 if np.isnan(mean) else mean - lo)
        err_hi.append(0.0 if np.isnan(mean) else hi - mean)
        bar_colors.append(colors[cat])
        finite_top = [v for v in list(values) + [hi] if not np.isnan(v)]
        y_top_per_category[cat] = max(finite_top) if finite_top else 0.0
        all_values.extend(v for v in values if not np.isnan(v))

    fig.add_trace(go.Bar(
        x=bar_x, y=bar_y,
        error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, color=fg_color, thickness=3.5, width=0),
        marker=dict(color=bar_colors, opacity=0.85),
        showlegend=False,
        hovertemplate="mean: %{y:.3f}<extra></extra>",
    ))

    jitter_rng = np.random.default_rng(0)
    for cat in categories:
        sub = data[data[group_col] == cat]
        values = sub["value"].to_numpy()
        if values.size == 0:
            continue
        jitter = (jitter_rng.random(values.size) - 0.5) * 0.3
        hover = (
            [f"subject: {s}<br>value: {v:.3f}" for s, v in zip(sub["subject_id"], values)]
            if has_subject_col else [f"value: {v:.3f}" for v in values]
        )
        fig.add_trace(go.Scatter(
            x=position[cat] + jitter, y=values,
            mode="markers", marker=dict(color=fg_color, size=9, opacity=0.65, line=dict(width=0.5, color=fg_color)),
            showlegend=False, hovertext=hover, hoverinfo="text",
        ))

    y_min = min(all_values) if all_values else 0.0
    y_max = max(all_values + [v for v in bar_y if not np.isnan(v)]) if all_values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    axis_top = y_max + (y_max - y_min) * 0.1

    if stats_result is not None:
        sub_stats = stats_result[stats_result["feature_name"] == feature_name]
        if not sub_stats.empty:
            pairs = list(zip(sub_stats["group_a"], sub_stats["group_b"]))
            p_col = "p_adj" if "p_adj" in sub_stats.columns else "p_value"
            pvalues = sub_stats[p_col].tolist()
            axis_top = add_significance_brackets(
                fig, pairs, pvalues, categories, y_top_per_category, (y_min, y_max), line_color=fg_color,
            )
            axis_top += (y_max - y_min) * 0.08

    add_chance_level(fig, chance_level)

    yaxis_range = list(ylim) if ylim is not None else [y_min - (y_max - y_min) * 0.08, axis_top]

    fig.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=list(range(len(categories))),
            ticktext=[str(c) for c in categories], title=group_col,
        ),
        yaxis=dict(title=feature_name, range=yaxis_range),
        title=title or feature_name,
        template=template,
        height=450,
        showlegend=False,
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path), format="pdf")  # requires kaleido

    return fig
