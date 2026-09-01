"""Hand-built significance brackets for Plotly bar/box plots (Plotly has no
statannotations equivalent), shared by plot_group_comparison and
plot_feature_type_comparison."""


def p_to_stars(p):
    """Match statannotations' default star thresholds."""
    if p <= 1e-4:
        return "****"
    if p <= 1e-3:
        return "***"
    if p <= 1e-2:
        return "**"
    if p <= 5e-2:
        return "*"
    return "ns"


def add_significance_brackets(fig, pairs, pvalues, order, y_top_per_category, y_axis_range, line_color="black"):
    """Draw stacked horizontal significance brackets above a categorical
    bar/box plot, one per (pair, p-value), starred via p_to_stars. Categories
    must already be plotted at integer x positions matching `order`. Any
    pair with a category not in `order` is skipped. Returns the new plot
    top (existing data top + all bracket rows) so the caller can extend
    the y-axis range, since Plotly won't auto-expand for shapes/annotations.
    """
    position = {cat: i for i, cat in enumerate(order)}
    span = y_axis_range[1] - y_axis_range[0]
    if span <= 0:
        span = max(abs(y_axis_range[1]), 1.0)
    step = span * 0.10
    tick_height = step * 0.15

    base = (max(y_top_per_category.values()) if y_top_per_category else y_axis_range[1]) + span * 0.06

    row = 0
    for (a, b), p in zip(pairs, pvalues):
        if a not in position or b not in position:
            continue
        x0, x1 = position[a], position[b]
        if x0 == x1:
            continue
        if x0 > x1:
            x0, x1 = x1, x0
        y = base + row * step
        stars = p_to_stars(p)

        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=line_color, width=3))
        fig.add_shape(type="line", x0=x0, x1=x0, y0=y - tick_height, y1=y, line=dict(color=line_color, width=3))
        fig.add_shape(type="line", x0=x1, x1=x1, y0=y - tick_height, y1=y, line=dict(color=line_color, width=3))
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y, yshift=6, text=stars, showarrow=False,
            font=dict(size=20, color=line_color), align="center",
        )
        row += 1

    return base + row * step
