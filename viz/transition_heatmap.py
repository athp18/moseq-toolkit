import numpy as np
import plotly.graph_objects as go

from viz.palette import DEFAULT_SEQUENTIAL_COLORSCALE, DEFAULT_TEMPLATE


def plot_transition_heatmap(
    matrix, labels=None, colorscale=DEFAULT_SEQUENTIAL_COLORSCALE, title=None,
    template=DEFAULT_TEMPLATE, **kwargs,
):
    """Render one N x N transition matrix (a single subject's or group's; not
    averaged here) as an interactive Plotly heatmap. `labels` defaults to
    integer indices 0..n-1. **kwargs are forwarded to go.Heatmap."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"plot_transition_heatmap: expected a square 2D matrix, got shape {matrix.shape}")

    n = matrix.shape[0]
    if labels is None:
        labels = list(range(n))
    elif len(labels) != n:
        raise ValueError(f"plot_transition_heatmap: labels has length {len(labels)}, expected {n}")
    labels = [str(l) for l in labels]

    heatmap_kwargs = dict(zmin=0, zmax=float(np.nanmax(matrix)) if matrix.size else 1.0)
    heatmap_kwargs.update(kwargs)

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale=colorscale,
        colorbar=dict(title="transition<br>probability"),
        hovertemplate="from: %{y}<br>to: %{x}<br>p: %{z:.3f}<extra></extra>",
        **heatmap_kwargs,
    ))
    fig.update_layout(
        xaxis=dict(title="to syllable", side="bottom"),
        yaxis=dict(title="from syllable", autorange="reversed"),
        template=template,
        title=title,
        height=550,
        width=600,
    )
    return fig
