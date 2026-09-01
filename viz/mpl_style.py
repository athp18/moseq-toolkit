"""Small helper shared across the Plotly plot functions."""


def add_chance_level(fig, chance_level, label="chance"):
    """Add a dashed horizontal reference line at `chance_level`."""
    if chance_level is None:
        return
    fig.add_hline(
        y=chance_level, line=dict(color="gray", dash="dash", width=1),
        annotation_text=label, annotation_position="top left",
        annotation_font=dict(size=10, color="gray"),
    )
