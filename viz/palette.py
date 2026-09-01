"""Shared color options for every Plotly plot in this toolkit."""

import plotly.express as px

# Qualitative (categorical) palettes for discrete series: clusters, group_col
# values, ROC classes, feature-set comparison bars.
QUALITATIVE_PALETTES = {
    "Default": [
        "#DC143C",  # Crimson
        "#4169E1",  # Royal Blue
        "#FA8072",  # Salmon
        "#87CEEB",  # Sky Blue
        "#2E8B57",  # Sea Green
        "#DAA520",  # Goldenrod
        "#9370DB",  # Medium Purple
        "#20B2AA",  # Light Sea Green
    ],
    "Plotly": list(px.colors.qualitative.Plotly),
    "Colorblind-safe": list(px.colors.qualitative.Safe),
    "Pastel": list(px.colors.qualitative.Pastel),
}

# Fixed mapping for this toolkit's canonical 4-cluster labels under the
# "Default" palette (high displacement = red family, low = blue family).
DEFAULT_CLUSTER_COLORS = {
    "motor improvement": "#DC143C",
    "motor deficit": "#FA8072",
    "cognitive improvement": "#4169E1",
    "cognitive deficit": "#87CEEB",
}

# Sequential colorscales for heatmaps (transition matrices, confusion
# matrices). Valid Plotly `colorscale=` names as-is.
SEQUENTIAL_COLORSCALES = ["Blues", "Viridis", "Magma", "Greens", "RdBu", "YlOrRd"]

DEFAULT_QUALITATIVE_PALETTE = "Default"
DEFAULT_SEQUENTIAL_COLORSCALE = "Blues"

# Plotly template used by every plot; the app derives this from Streamlit's
# own light/dark setting.
DEFAULT_TEMPLATE = "plotly_white"


def foreground_for_template(template):
    """Color for plot elements hardcoded outside the template's own color
    scale (bracket lines/text, jittered points, cluster labels)."""
    return "#EEEEEE" if template == "plotly_dark" else "black"


def colors_for_categories(categories, palette_name=DEFAULT_QUALITATIVE_PALETTE, use_default_mapping=None):
    """Assign a color to each of `categories`. use_default_mapping (consulted
    first, only when palette_name == "Default") lets a fixed mapping like
    DEFAULT_CLUSTER_COLORS override the cycling palette. Returns dict
    {category: hex color string}."""
    palette = QUALITATIVE_PALETTES.get(palette_name, QUALITATIVE_PALETTES[DEFAULT_QUALITATIVE_PALETTE])
    categories = list(categories)

    # lock in default-mapping colors first, then cycle the palette for the rest
    mapping = {}
    used_colors = set()
    if palette_name == "Default" and use_default_mapping:
        for cat in categories:
            if cat in use_default_mapping:
                mapping[cat] = use_default_mapping[cat]
                used_colors.add(mapping[cat])

    fallback_idx = 0
    for cat in categories:
        if cat in mapping:
            continue
        while palette[fallback_idx % len(palette)] in used_colors and fallback_idx < len(palette):
            fallback_idx += 1
        color = palette[fallback_idx % len(palette)]
        mapping[cat] = color
        used_colors.add(color)
        fallback_idx += 1

    return mapping


def hex_to_rgba(hex_color, alpha):
    """'#RRGGBB' -> 'rgba(r, g, b, alpha)', for Plotly fillcolor/etc that need transparency."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color  # not a plain hex string; return as-is
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"
