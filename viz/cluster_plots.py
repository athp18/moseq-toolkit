import plotly.graph_objects as go
from scipy.spatial import ConvexHull

from viz.palette import (
    DEFAULT_CLUSTER_COLORS, DEFAULT_QUALITATIVE_PALETTE, DEFAULT_TEMPLATE,
    colors_for_categories, foreground_for_template, hex_to_rgba,
)


def plot_cluster_locations(
    syllable_kinematics,
    cluster_col="cluster",
    id_col="syllable",
    velocity_col="mean_velocity",
    displacement_col="mean_displacement",
    palette_name=DEFAULT_QUALITATIVE_PALETTE,
    title=None,
    template=DEFAULT_TEMPLATE,
):
    """Interactive scatter of every syllable at its (mean_velocity,
    mean_displacement) location, colored by cluster, with a convex hull
    and star centroid per cluster. syllable_kinematics must already carry
    a `cluster_col` column; rows with a null cluster are dropped.
    """
    missing = {cluster_col, id_col, velocity_col, displacement_col} - set(syllable_kinematics.columns)
    if missing:
        raise ValueError(f"plot_cluster_locations: syllable_kinematics missing columns {missing}")

    data = syllable_kinematics.dropna(subset=[cluster_col])
    if data.empty:
        raise ValueError("plot_cluster_locations: no rows with a non-null cluster assignment")

    clusters = sorted(data[cluster_col].unique(), key=str)
    colors = colors_for_categories(clusters, palette_name, DEFAULT_CLUSTER_COLORS)
    fg_color = foreground_for_template(template)

    fig = go.Figure()

    # hulls added first so they render behind each cluster's points
    for cluster in clusters:
        group = data[data[cluster_col] == cluster]
        c = colors[cluster]
        label = str(cluster).replace("_", " ")
        points = group[[velocity_col, displacement_col]].to_numpy()

        # a cluster with fewer than 3 members has no well-defined hull
        if len(points) < 3:
            continue

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hx = list(hull_points[:, 0]) + [hull_points[0, 0]]
        hy = list(hull_points[:, 1]) + [hull_points[0, 1]]
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="lines", fill="toself",
            fillcolor=hex_to_rgba(c, 0.15),
            line=dict(color=hex_to_rgba(c, 0.4), width=1),
            legendgroup=label, showlegend=False, hoverinfo="skip",
        ))

    for cluster in clusters:
        group = data[data[cluster_col] == cluster]
        c = colors[cluster]
        label = str(cluster).replace("_", " ")
        fig.add_trace(go.Scatter(
            x=group[velocity_col], y=group[displacement_col],
            mode="markers+text",
            text=group[id_col].astype(str),
            textposition="top center",
            textfont=dict(size=9, color=fg_color),
            marker=dict(size=11, color=c, opacity=0.8, line=dict(width=0.5, color="white")),
            name=label,
            legendgroup=label,
            customdata=group[id_col].astype(str).to_numpy().reshape(-1, 1),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "id: %{customdata[0]}<br>"
                f"{velocity_col}: %{{x:.3f}}<br>"
                f"{displacement_col}: %{{y:.3f}}"
                "<extra></extra>"
            ),
        ))

    for cluster in clusters:
        group = data[data[cluster_col] == cluster]
        c = colors[cluster]
        label = str(cluster).replace("_", " ")
        points = group[[velocity_col, displacement_col]].to_numpy()
        if len(points) < 3:
            continue

        centroid = points.mean(axis=0)
        fig.add_trace(go.Scatter(
            x=[centroid[0]], y=[centroid[1]], mode="markers",
            marker=dict(size=16, symbol="star", color=c, line=dict(width=1.5, color=fg_color)),
            legendgroup=label, showlegend=False,
            hovertemplate=(
                f"<b>{label} centroid</b><br>"
                f"{velocity_col}: %{{x:.3f}}<br>"
                f"{displacement_col}: %{{y:.3f}}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title=f"Mean velocity ({velocity_col})",
        yaxis_title=f"Mean displacement ({displacement_col})",
        title=title,
        template=template,
        legend=dict(title="Cluster"),
        height=550,
    )
    return fig
