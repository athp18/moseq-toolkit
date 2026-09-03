"""Streamlit front-end over the moseq_toolkit pipeline: wires widgets to the
library functions (ingest, clusters, usage, scalars, transitions, stats,
classify) and renders their output."""

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl

from ingest.from_moseq_df import load_moseq_df
from features.clusters import assign_clusters
from features.usage import attach_clusters, aggregate_clusters
from features.scalars import compute as compute_scalars, compute_histograms
from features.transitions import (
    compute_matrix,
    to_long,
    collapse_to_clusters,
    average_matrices,
)
from stats.compare_groups import compare_groups
from classify.classifier import (
    build_matrix, transform_matrix, run, compare_feature_types, complement_feature_names,
)
from viz.palette import QUALITATIVE_PALETTES, SEQUENTIAL_COLORSCALES
from viz.usage_plots import plot_group_comparison
from viz.cluster_plots import plot_cluster_locations
from viz.transition_heatmap import plot_transition_heatmap
from viz.classifier_plots import plot_confusion_matrix, plot_roc, plot_feature_type_comparison
from viz import mpl_plots
from ordering import natural_sort


st.set_page_config(page_title="MoSeq Toolkit", layout="wide")


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def _current_plotly_template():
    """Pick the Plotly template matching Streamlit's light/dark theme setting."""
    try:
        return "plotly_dark" if st.context.theme.type == "dark" else "plotly_white"
    except Exception:
        return "plotly_white"


def _df_download_button(df, label, filename):
    st.download_button(label, data=df.to_csv(index=False), file_name=filename, mime="text/csv")


def _show_plot(fig, filename):
    """Render a Plotly or matplotlib figure with a matching download button."""
    if isinstance(fig, mpl.Figure):
        st.pyplot(fig, use_container_width=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight")
        st.download_button(
            "Download plot (PDF)", data=buf.getvalue(), file_name=filename, mime="application/pdf",
        )
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def _cached_result(cache_key, context, build_fn):
    """Reuse a previously-built result (a figure, a stats table, ...) from
    session_state 
    """
    cached = st.session_state.get(cache_key)
    if cached is not None and cached["context"] == context:
        return cached["result"]
    result = build_fn()
    st.session_state[cache_key] = {"context": context, "result": result}
    return result


def _section_header(title):
    """A consistently-styled bold header for a workflow step's card. The
    global pipeline status bar already shows overall progress, so steps
    are identified by title only, not renumbered per tab."""
    st.markdown(f"##### {title}")


def _render_pipeline_status():
    data_loaded = st.session_state["frames"] is not None
    steps = [
        ("Data", data_loaded),
        ("Clusters", st.session_state["cluster_mapping"] is not None),
        ("Scalars", st.session_state["scalars_long"] is not None),
        ("Transitions", (
            st.session_state["transitions_long"] is not None
            or st.session_state["transition_cluster_long"] is not None
        )),
        ("Classifier", st.session_state.get("classify_result") is not None),
    ]

    completed_count = sum(1 for _, ready in steps if ready)
    progress_pct = completed_count / len(steps)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        completed_names = [label for label, ready in steps if ready]
        if completed_names:
            st.caption(f"**Pipeline Progress:** Active stages — {', '.join(completed_names)}")
        else:
            st.caption("**Pipeline Progress:** No stages completed")
        st.progress(progress_pct)
    if data_loaded:
        n_subj = st.session_state["usage"]["subject_id"].nunique()
        n_syl = st.session_state["syllable_kinematics"].shape[0]
        with col2:
            st.metric("Subjects", n_subj)
        with col3:
            st.metric("Syllables", n_syl)


_CLUSTER_LABEL_ORDER = ["motor improvement", "motor deficit", "cognitive improvement", "cognitive deficit"]


def _sorted_feature_names(names, feature_types):
    """Order candidate feature_names for a picker: fixed cluster order for the
    "cluster" feature_type, alphabetical otherwise."""
    names = list(names)
    if list(feature_types) == ["cluster"]:
        rank = {name: i for i, name in enumerate(_CLUSTER_LABEL_ORDER)}
        return sorted(names, key=lambda n: (rank.get(n, len(rank)), str(n)))
    return sorted(names, key=str)


def _plot_comparison_with_stats(data_df, feature_name, group_col, paired, key_prefix, timepoint=None, filters=None):
    """Plot a group comparison for one feature_name, with stats annotated on
    the bars; the full multi-feature stats table loads on demand. 
    The full stats table below is built from the
    same filtered data but still covers every timepoint (compare_groups
    stratifies it automatically)."""
    if filters:
        for col, val in filters.items():
            data_df = data_df[data_df[col] == val]

    sub = data_df[data_df["feature_name"] == feature_name]
    if timepoint is not None:
        sub = sub[sub["timepoint"] == timepoint]
    if sub.empty:
        st.error(f"No rows for feature_name={feature_name!r}")
        return

    plot_style = st.session_state["plot_style"]
    palette = st.session_state["plot_palette"]
    plot_context = (
        feature_name, group_col, paired, timepoint, tuple(sorted((filters or {}).items())),
        plot_style, palette, PLOT_TEMPLATE, len(sub), tuple(sub.columns),
        int(pd.util.hash_pandas_object(sub["value"]).sum()),
    )

    def _build():
        stats_result = compare_groups(sub, group_col, paired=paired)
        if plot_style == "notebook":
            built_fig = mpl_plots.mpl_group_comparison(
                sub, feature_name, group_col,
                stats_result=stats_result if not stats_result.empty else None,
                palette_name=palette,
            )
        else:
            built_fig = plot_group_comparison(
                sub, feature_name, group_col,
                stats_result=stats_result if not stats_result.empty else None,
                palette_name=palette,
                template=PLOT_TEMPLATE,
            )
        return stats_result, built_fig

    annotation_stats, fig = _cached_result(f"{key_prefix}_plot_cache", plot_context, _build)
    _show_plot(fig, f"{key_prefix}_{feature_name}_{group_col}_{timepoint or 'all'}.pdf")

    table_key = f"{key_prefix}_stats_table"
    context_key = f"{key_prefix}_stats_table_context"
    current_context = (
        group_col, paired, timepoint, tuple(sorted((filters or {}).items())),
        tuple(sorted(data_df["feature_name"].astype(str).unique())),
    )

    if st.button("See stats for all features", key=f"{table_key}_button"):
        try:
            st.session_state[table_key] = compare_groups(data_df, group_col, paired=paired)
            st.session_state[context_key] = current_context
        except Exception as e:
            st.error(str(e))

    stale = st.session_state.get(context_key) != current_context
    table = st.session_state.get(table_key)
    if table is not None and not stale:
        with st.expander("View data: full comparison stats", expanded=True):
            if table.empty:
                st.caption("No comparisons found.")
            else:
                st.dataframe(table, use_container_width=True)
                _df_download_button(table, "Download stats (CSV)", f"{key_prefix}_{group_col}_stats.csv")
    elif table is not None and stale:
        st.warning(
            "**Group, pairing, filter, or timepoint changed since this stats table was loaded.** "
            "Click **See stats for all features** again to refresh it -- the table below is out of date."
        )


def render_comparison_section(source_options, group_cols, key_prefix, empty_message=None):
    """Build the full "plot a group comparison" UI (source selector, controls,
    Plot button, plot + stats table); shared by the Usage, Scalars, and
    Transitions tabs, namespaced by key_prefix."""
    if not source_options:
        if empty_message:
            st.info(empty_message)
        return

    source_state_key = f"_{key_prefix}_plot_source"
    feature_name_key = f"{key_prefix}_feature_name"
    group_col_key = f"{key_prefix}_group_col"
    paired_key = f"{key_prefix}_paired"
    plot_button_key = f"{key_prefix}_plot_button"
    selection_key = f"{key_prefix}_plot_selection"

    multi_source = len(source_options) > 1
    if multi_source:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            source_label = st.selectbox(
                "Feature source", options=list(source_options.keys()), key=f"{key_prefix}_source",
            )
    else:
        c2, c3, c4 = st.columns([2, 2, 1])
        source_label = next(iter(source_options))
    source_df = source_options[source_label]

    if st.session_state.get(source_state_key) != source_label:
        st.session_state.pop(feature_name_key, None)
        st.session_state[source_state_key] = source_label

    with c2:
        feature_name = st.selectbox(
            "Feature", options=sorted(source_df["feature_name"].unique(), key=str), key=feature_name_key,
        )
    with c3:
        grouping_options = ["timepoint"] + [c for c in group_cols if c != "timepoint"]
        group_col = st.selectbox("Grouping variable", options=grouping_options, key=group_col_key)
    with c4:
        paired = st.checkbox("Paired", key=paired_key, help="Treat this as a paired-samples comparison.")

    filter_dims = [c for c in group_cols if c not in (group_col, "timepoint") and c in source_df.columns]
    filters = {}
    if filter_dims:
        filter_cols = st.columns(len(filter_dims))
        for col, fcol in zip(filter_dims, filter_cols):
            with fcol:
                filter_options = ["All"] + natural_sort(source_df[col].dropna().unique())
                chosen = st.selectbox(f"Filter: {col}", options=filter_options, key=f"{key_prefix}_filter_{col}")
                if chosen != "All":
                    filters[col] = chosen

    timepoint_key = f"{key_prefix}_timepoint"
    timepoint = None
    if group_col != "timepoint" and "timepoint" in source_df.columns and source_df["timepoint"].nunique() > 1:
        timepoint_options = natural_sort(source_df["timepoint"].dropna().unique())
        timepoint = st.selectbox(
            "Timepoint", options=timepoint_options, key=timepoint_key,
            help="This dataset has multiple timepoints. The plot below shows one "
                 "timepoint at a time so repeated visits from the same subject "
                 "aren't pooled into one bar.",
        )

    if st.button("Plot", key=plot_button_key, disabled=group_col is None):
        st.session_state[selection_key] = dict(
            source_label=source_label, feature_name=feature_name, group_col=group_col, paired=paired,
            timepoint=timepoint, filters=filters,
        )

    # Read from session_state, not the button's own `if`, so it survives reruns.
    sel = st.session_state.get(selection_key)
    if sel is not None and sel["source_label"] in source_options:
        try:
            sel_source_df = source_options[sel["source_label"]]
            stats_key_prefix = key_prefix
            if multi_source:
                source_key = sel["source_label"].replace(" ", "_").replace("-", "_")
                stats_key_prefix = f"{key_prefix}_{source_key}"
            _plot_comparison_with_stats(
                sel_source_df, sel["feature_name"], sel["group_col"], sel["paired"], stats_key_prefix,
                timepoint=sel.get("timepoint"), filters=sel.get("filters"),
            )
        except ValueError as e:
            st.error(str(e))


def _combined_long():
    """Concatenate every long-format table computed so far into one frame."""
    parts = []
    for key in ["usage", "cluster_usage", "scalars_long", "histograms_long",
                "transitions_long", "transition_cluster_long"]:
        df = st.session_state.get(key)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True, sort=False)


def _compute_and_store_histograms(frames, bin_defs, group_cols):
    """Compute histogram channels and merge into st.session_state["histograms_long"],
    replacing any previously-computed rows for the same channel names."""
    new_df = compute_histograms(frames, bin_defs, group_cols)
    channel_names = set(bin_defs.keys())
    existing = st.session_state["histograms_long"]
    if existing is not None and not existing.empty:
        existing = existing[~existing["feature_type"].isin(channel_names)]
        st.session_state["histograms_long"] = pd.concat([existing, new_df], ignore_index=True)
    else:
        st.session_state["histograms_long"] = new_df


def _init_state():
    defaults = dict(
        frames=None, syllable_kinematics=None, usage=None, group_cols=[],
        cluster_mapping=None, usage_with_clusters=None, cluster_usage=None,
        scalars_long=None, histograms_long=None,
        matrices=None, transitions_metadata=None, n_syllables=None,
        averaged_matrices=None, averaged_group_meta=None,
        transitions_long=None, transition_cluster_long=None,
        feature_sets=[],
        usage_plot_selection=None, scalars_plot_selection=None, transitions_plot_selection=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# Computed once per rerun; used by every plot call below.
PLOT_TEMPLATE = _current_plotly_template()


# ---------------------------------------------------------------------------
# sidebar: dataset + appearance
# ---------------------------------------------------------------------------

st.sidebar.title("MoSeq Toolkit")

st.sidebar.caption("Dataset")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Parquet)", type=["csv", "parquet"])

raw_df = None
if uploaded is not None:
    try:
        if uploaded.name.endswith(".parquet"):
            raw_df = pd.read_parquet(uploaded)
        else:
            raw_df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

if raw_df is not None:
    _structural_cols = {"timepoint", "SubjectName"}
    candidate_group_cols = [
        c for c in raw_df.columns
        if c not in _structural_cols and (raw_df[c].dtype == object or raw_df[c].nunique() < 20)
    ]
    group_cols = st.sidebar.multiselect(
        "Grouping variables",
        options=[c for c in raw_df.columns if c not in _structural_cols],
        default=[c for c in candidate_group_cols if c in ("treatment", "group", "sex", "genotype")],
        help="Experimental grouping columns already present in your dataset (e.g. treatment, sex, genotype). "
             "\"timepoint\" doesn't need to be added here -- it's already available as a Grouping variable "
             "everywhere below.",
    )
    velocity_candidates = [c for c in raw_df.columns if "velocity" in c.lower()]
    velocity_col = st.sidebar.selectbox(
        "Velocity column",
        options=velocity_candidates or list(raw_df.columns),
        help="Which column to use as velocity when computing syllable kinematics.",
    )

    if st.sidebar.button("Load data", type="primary"):
        try:
            frames, syllable_kinematics, usage = load_moseq_df(raw_df, group_cols, velocity_col)
            st.session_state["frames"] = frames
            st.session_state["syllable_kinematics"] = syllable_kinematics
            st.session_state["usage"] = usage
            st.session_state["group_cols"] = group_cols
            # loading fresh data invalidates everything downstream
            for k in ["cluster_mapping", "usage_with_clusters", "cluster_usage", "scalars_long",
                      "histograms_long",
                      "matrices", "transitions_metadata", "n_syllables", "averaged_matrices",
                      "averaged_group_meta", "transitions_long", "transition_cluster_long"]:
                st.session_state[k] = None
            # avoid stale min_value conflicts from the previous dataset
            st.session_state.pop("use_custom_n", None)
            st.session_state.pop("n_syllables_override_value", None)
            st.sidebar.success(
                f"Ingested: {frames.shape[0]:,} frames, "
                f"{syllable_kinematics.shape[0]} syllables, "
                f"{usage['subject_id'].nunique()} subjects"
            )
        except ValueError as e:
            st.sidebar.error(str(e))

st.sidebar.divider()
with st.sidebar.expander("Appearance", expanded=False):
    st.selectbox(
        "Color palette", options=list(QUALITATIVE_PALETTES.keys()), key="plot_palette",
        help="Controls every categorical plot's colors (clusters, groups, ROC classes, "
             "feature-set comparisons). 'Default' keeps this toolkit's fixed cluster color scheme "
             "where applicable; any other palette cycles through its own colors instead.",
    )
    st.selectbox(
        "Heatmap colorscale", options=SEQUENTIAL_COLORSCALES, key="plot_colorscale",
        help="Controls the transition-matrix and confusion-matrix heatmaps.",
    )
    st.selectbox(
        "Plot style", options=["toolkit", "notebook"], key="plot_style",
        format_func=lambda s: "Toolkit" if s == "toolkit" else "Notebook (matplotlib)",
        help="'Toolkit' renders every plot as an interactive Plotly figure (hover/zoom/click-to-toggle). "
             "'Notebook (matplotlib)' renders the exact same data with real matplotlib/seaborn/statannotations "
             "instead -- a static figure matching the original analysis notebooks' conventions (despine, "
             "tight_layout, dpi=300), downloadable as a PDF. This switches the rendering engine, not just a "
             "visual theme -- both draw from the same underlying data and stats. Colors and colorscale are "
             "controlled by the pickers above regardless of this setting.",
    )


# ---------------------------------------------------------------------------
# main canvas: header + pipeline status
# ---------------------------------------------------------------------------

data_loaded = st.session_state["frames"] is not None

st.title("MoSeq Toolkit")
st.caption("Behavioral sequence analysis & visualization")
_render_pipeline_status()
st.divider()

if not data_loaded:
    st.info(
        "Upload a MoSeq per-frame dataset (CSV or Parquet) in the sidebar, "
        "pick your grouping variables and velocity column, then click **Load data** to get started."
    )
    st.markdown(
        "Once data is loaded, this walks you through: assigning behavioral clusters, "
        "comparing feature usage across groups, computing scalar and histogram summaries, "
        "building transition matrices, and training classifiers across feature sets."
    )
    st.stop()

frames = st.session_state["frames"]
syllable_kinematics = st.session_state["syllable_kinematics"]
usage = st.session_state["usage"]
group_cols = st.session_state["group_cols"]

tab_usage, tab_scalars, tab_transitions, tab_classify = st.tabs(
    ["Usage & Clusters", "Scalars", "Transitions", "Classifier"]
)


# ---------------------------------------------------------------------------
# Tab 1: Usage & Clusters
# ---------------------------------------------------------------------------

with tab_usage:
    st.header("Usage & Clusters")

    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            _section_header("Assign behavioral clusters")
            threshold_mode = st.radio(
                "Syllable inclusion threshold",
                options=["min_frames (absolute count)", "coverage (% of frames)"],
                horizontal=True,
            )
            if threshold_mode == "min_frames (absolute count)":
                min_frames = st.number_input("Minimum frames", min_value=0, value=5, step=1)
                coverage_pct = None
            else:
                coverage_pct = st.number_input(
                    "Coverage (%)", min_value=1, max_value=100, value=99, step=1,
                    help="Keep the most-used syllables covering this % of all frames; "
                         "drop the rare tail beyond it (e.g. 99% excludes the bottom ~1%).",
                )
                min_frames = None

            if st.session_state["cluster_mapping"] is not None:
                st.warning(
                    "Reassigning will clear the cluster usage computed on the right, and any "
                    "cluster-collapsed transitions exported in the Transitions tab."
                )
            if st.button("Assign clusters"):
                try:
                    cluster_mapping = assign_clusters(
                        syllable_kinematics,
                        min_frames=min_frames if min_frames is not None else 5,
                        coverage=(coverage_pct / 100.0) if coverage_pct is not None else None,
                    )
                    st.session_state["cluster_mapping"] = cluster_mapping
                    st.session_state["usage_with_clusters"] = None
                    st.session_state["cluster_usage"] = None
                    st.session_state["transition_cluster_long"] = None
                except ValueError as e:
                    st.error(str(e))

            if st.session_state["cluster_mapping"] is not None:
                mapping = st.session_state["cluster_mapping"]
                n_clusters = len(set(mapping.values()))
                mc1, mc2 = st.columns(2)
                mc1.metric("Clusters assigned", n_clusters)
                mc2.metric("Syllables included", len(mapping))
                with st.expander("View data: cluster mapping"):
                    mapping_df = pd.DataFrame(sorted(mapping.items()), columns=["syllable", "cluster"])
                    st.dataframe(mapping_df, use_container_width=True, height=200)
                    _df_download_button(mapping_df, "Download cluster mapping (CSV)", "cluster_mapping.csv")

    with col_b:
        with st.container(border=True):
            _section_header("Cluster-level usage")
            st.caption("↳ requires **Assign behavioral clusters** (left)")
            renormalize = st.checkbox("Renormalize cluster fractions to sum to 1", value=True)
            cluster_ready = st.session_state["cluster_mapping"] is not None
            if st.button("Compute cluster usage", disabled=not cluster_ready):
                try:
                    usage_with_clusters = attach_clusters(usage, st.session_state["cluster_mapping"])
                    cluster_usage = aggregate_clusters(usage_with_clusters, renormalize=renormalize)
                    st.session_state["usage_with_clusters"] = usage_with_clusters
                    st.session_state["cluster_usage"] = cluster_usage
                except ValueError as e:
                    st.error(str(e))
            if not cluster_ready:
                st.info("Assign clusters first (left).")
            if st.session_state["cluster_usage"] is not None:
                cu = st.session_state["cluster_usage"]
                n_subj_cu = cu["subject_id"].nunique() if "subject_id" in cu.columns else None
                st.success("Cluster usage computed")
                if n_subj_cu is not None:
                    st.metric("Subjects", n_subj_cu)
                with st.expander("View data: cluster usage"):
                    st.dataframe(cu.head(20), use_container_width=True, height=200)
                    _df_download_button(cu, "Download cluster usage (CSV)", "cluster_usage.csv")

    with st.container(border=True):
        _section_header("Cluster locations")
        if st.session_state["cluster_mapping"] is None:
            st.info("Assign clusters first (above).")
        else:
            try:
                plot_style = st.session_state["plot_style"]
                palette = st.session_state["plot_palette"]
                location_context = (
                    tuple(sorted(st.session_state["cluster_mapping"].items())), plot_style, palette, PLOT_TEMPLATE,
                )

                def _build():
                    location_df = syllable_kinematics.copy()
                    # syllables filtered out by assign_clusters map to NaN and get dropped by the plot
                    location_df["cluster"] = location_df["syllable"].map(st.session_state["cluster_mapping"])
                    if plot_style == "notebook":
                        return mpl_plots.mpl_cluster_locations(location_df, palette_name=palette)
                    return plot_cluster_locations(location_df, palette_name=palette, template=PLOT_TEMPLATE)

                fig = _cached_result("usage_cluster_locations_cache", location_context, _build)
                _show_plot(fig, "cluster_locations.pdf")
            except ValueError as e:
                st.error(str(e))

    with st.container(border=True):
        _section_header("Group comparison")
        plot_source_options = {"syllable usage": usage}
        if st.session_state["cluster_usage"] is not None:
            plot_source_options["cluster usage"] = st.session_state["cluster_usage"]
        render_comparison_section(plot_source_options, group_cols, "usage")


# ---------------------------------------------------------------------------
# Tab 2: Scalars
# ---------------------------------------------------------------------------

with tab_scalars:
    st.header("Scalars")

    scalars = ('velocity', 'area', 'length', 'height', 'dist_to_center', 'angle', 'displacement')
    scalar_cols = [c for c in frames.select_dtypes(include=np.number).columns
                    if any(k in c.lower() for k in scalars)]

    with st.container(border=True):
        _section_header("Compute scalar summaries")
        metrics = st.multiselect("Metrics to summarize", options=scalar_cols)
        outlier_method = st.radio("Outlier handling", options=["none", "iqr"], horizontal=True)

        if st.button("Compute scalars", disabled=not metrics, type="primary"):
            try:
                scalars_long = compute_scalars(
                    frames, metrics, group_cols,
                    outlier_method=None if outlier_method == "none" else outlier_method,
                )
                st.session_state["scalars_long"] = scalars_long
            except ValueError as e:
                st.error(str(e))

        if st.session_state["scalars_long"] is not None:
            n_metrics = st.session_state["scalars_long"]["feature_name"].nunique()
            st.success("Scalars computed")
            st.metric("Scalar metrics", n_metrics)
            with st.expander("View data: scalars"):
                st.dataframe(st.session_state["scalars_long"].head(20), use_container_width=True, height=200)
                _df_download_button(st.session_state["scalars_long"], "Download scalars (CSV)", "scalars_long.csv")

    with st.container(border=True):
        _section_header("Compute histogram features")
        with st.expander("What is this?"):
            st.write(
                "A distributional feature -- the full per-subject shape of a metric's frame-level values "
                "(as a proportion-per-bin vector), rather than the single per-subject mean computed above. "
                "Bin edges are shared across every subject so histograms stay comparable. Select as many "
                "metrics as you want below and compute all of their histograms in one click."
            )
        hist_metrics = st.multiselect("Metrics to histogram", options=scalar_cols, key="hist_metrics")

        bin_defs = {}
        if hist_metrics:
            same_settings = st.checkbox(
                "Use the same number of bins for all of these, named after their metric column",
                value=True, key="hist_same_settings",
            )
            if same_settings:
                shared_n_bins = st.number_input(
                    "Number of bins (applied to every selected metric)", min_value=1, value=50, step=1,
                    key="hist_shared_n_bins",
                )
                bin_defs = {metric: (metric, int(shared_n_bins)) for metric in hist_metrics}
            else:
                st.caption("Customize each channel's name and bin count:")
                for metric in hist_metrics:
                    rc1, rc2 = st.columns([2, 1])
                    with rc1:
                        channel_name = st.text_input(
                            f"Channel name for {metric}", value=metric, key=f"hist_channel_name_{metric}",
                        )
                    with rc2:
                        n_bins = st.number_input(
                            f"Number of bins for {metric}", min_value=1, value=50, step=1,
                            key=f"hist_n_bins_{metric}",
                        )
                    bin_defs[channel_name] = (metric, int(n_bins))

        if st.button("Compute histograms", disabled=not bin_defs, key="compute_histogram_button"):
            try:
                _compute_and_store_histograms(frames, bin_defs, group_cols)
            except ValueError as e:
                st.error(str(e))

        if st.session_state["histograms_long"] is not None and not st.session_state["histograms_long"].empty:
            hist_channels = sorted(st.session_state["histograms_long"]["feature_type"].unique())
            st.success(f"Histogram channels computed: {', '.join(hist_channels)}")
            with st.expander("View data: histograms"):
                st.dataframe(st.session_state["histograms_long"].head(20), use_container_width=True, height=200)
                _df_download_button(
                    st.session_state["histograms_long"], "Download histograms (CSV)", "histograms_long.csv",
                )

    if st.session_state["scalars_long"] is not None:
        with st.container(border=True):
            _section_header("Group comparison")
            render_comparison_section({"scalars": st.session_state["scalars_long"]}, group_cols, "scalars")


# ---------------------------------------------------------------------------
# Tab 3: Transitions
# ---------------------------------------------------------------------------

with tab_transitions:
    st.header("Transitions")

    with st.container(border=True):
        _section_header("Compute per-subject transition matrices")

        inferred_n_syllables = int(frames["syllable"].max()) + 1 if not frames.empty else 1

        use_custom_n = st.checkbox("Override number of syllables (default: infer from data)", key="use_custom_n")
        if use_custom_n:
            n_syllables_input = st.number_input(
                "Number of syllables", min_value=1, value=inferred_n_syllables, step=1,
                key="n_syllables_override_value",
                help=(
                    f"Full coverage for the currently loaded data is {inferred_n_syllables} "
                    f"(labels 0..{inferred_n_syllables - 1}). A smaller value restricts coverage: "
                    "any transition touching an excluded syllable (label >= this value) is dropped, "
                    "not bridged over -- so a syllable's row only reflects transitions that were "
                    "genuinely, immediately consecutive to another included syllable. A larger value "
                    "only makes sense if your MoSeq model was fit with more syllables than actually "
                    "appear in this dataset."
                ),
            )
        else:
            n_syllables_input = None
            st.caption(
                f"Inferred from current data: {inferred_n_syllables} syllables "
                f"(labels 0...{inferred_n_syllables - 1})."
            )

        if st.button("Compute transition matrices", type="primary"):
            try:
                matrices, metadata, n_syllables = compute_matrix(
                    frames, group_cols=group_cols, n_syllables=n_syllables_input
                )
                st.session_state["matrices"] = matrices
                st.session_state["transitions_metadata"] = metadata
                st.session_state["n_syllables"] = n_syllables
                st.session_state["averaged_matrices"] = None
                st.session_state["averaged_group_meta"] = None
                st.session_state["transitions_long"] = None
                st.session_state["transition_cluster_long"] = None
            except ValueError as e:
                st.error(str(e))

        matrices_ready = st.session_state["matrices"] is not None
        if matrices_ready:
            st.success("Transition matrices computed")
            tm1, tm2 = st.columns(2)
            tm1.metric("Subject/timepoint matrices", len(st.session_state["matrices"]))
            tm2.metric("Syllables", st.session_state["n_syllables"])

    with st.container(border=True):
        _section_header("Average by group")
        avg_group_cols = st.multiselect(
            "Average across subjects within:", options=group_cols, default=group_cols,
            disabled=not matrices_ready,
        )
        if st.button("Average matrices", disabled=not matrices_ready):
            try:
                averaged, group_meta = average_matrices(
                    st.session_state["matrices"], st.session_state["transitions_metadata"], avg_group_cols
                )
                st.session_state["averaged_matrices"] = averaged
                st.session_state["averaged_group_meta"] = group_meta
            except ValueError as e:
                st.error(str(e))

        if st.session_state["averaged_matrices"] is not None:
            st.success(f"{len(st.session_state['averaged_matrices'])} group-averaged matrices")
            with st.expander("View data: groups averaged"):
                meta_df = pd.DataFrame(list(st.session_state["averaged_group_meta"].values()))
                st.dataframe(meta_df, use_container_width=True, height=150)

    with st.container(border=True):
        _section_header("View a heatmap")
        if not matrices_ready:
            st.info("Compute transition matrices first (above).")
        else:
            view_mode = st.radio("Matrix", options=["per-subject", "group-averaged"], horizontal=True,
                                  disabled=st.session_state["averaged_matrices"] is None)
            if view_mode == "per-subject" or st.session_state["averaged_matrices"] is None:
                keys = list(st.session_state["matrices"].keys())
                key_labels = [f"{s} / {t}" for s, t in keys]
                chosen = st.selectbox("Subject / timepoint", options=range(len(keys)),
                                       format_func=lambda i: key_labels[i])
                matrix_to_plot = st.session_state["matrices"][keys[chosen]]
                plot_title = key_labels[chosen]
            else:
                keys = list(st.session_state["averaged_matrices"].keys())
                key_labels = [
                    ", ".join(f"{k}={v}" for k, v in st.session_state["averaged_group_meta"][key].items())
                    for key in keys
                ]
                chosen = st.selectbox("Group", options=range(len(keys)), format_func=lambda i: key_labels[i])
                matrix_to_plot = st.session_state["averaged_matrices"][keys[chosen]]
                plot_title = key_labels[chosen]

            if st.button("Show heatmap"):
                if st.session_state["plot_style"] == "notebook":
                    fig = mpl_plots.mpl_transition_heatmap(
                        matrix_to_plot, title=plot_title, colorscale=st.session_state["plot_colorscale"],
                    )
                else:
                    fig = plot_transition_heatmap(
                        matrix_to_plot, title=plot_title, colorscale=st.session_state["plot_colorscale"],
                        template=PLOT_TEMPLATE,
                    )
                _show_plot(fig, "transition_heatmap.pdf")

    with st.container(border=True):
        _section_header("Export to long format (for Stats / Classifier)")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export pairwise transitions", disabled=not matrices_ready):
                st.session_state["transitions_long"] = to_long(
                    st.session_state["matrices"], st.session_state["transitions_metadata"]
                )
                st.success(f"{len(st.session_state['transitions_long']):,} rows exported")
        with c2:
            collapse_ready = matrices_ready and st.session_state["cluster_mapping"] is not None
            if st.button("Export cluster-collapsed transitions", disabled=not collapse_ready):
                st.session_state["transition_cluster_long"] = collapse_to_clusters(
                    st.session_state["matrices"], st.session_state["transitions_metadata"],
                    st.session_state["cluster_mapping"],
                )
                st.success(f"{len(st.session_state['transition_cluster_long']):,} rows exported")
            if not collapse_ready and matrices_ready:
                st.info("Assign clusters in the Usage & Clusters tab first.")

    with st.container(border=True):
        _section_header("Group comparison")
        transitions_source_options = {}
        if st.session_state["transitions_long"] is not None:
            transitions_source_options["pairwise transitions"] = st.session_state["transitions_long"]
        if st.session_state["transition_cluster_long"] is not None:
            transitions_source_options["cluster-collapsed transitions"] = st.session_state["transition_cluster_long"]

        render_comparison_section(
            transitions_source_options, group_cols, "transitions",
            empty_message="Export a long-format table above first.",
        )


# ---------------------------------------------------------------------------
# Tab 4: Classifier
# ---------------------------------------------------------------------------

with tab_classify:
    st.header("Classifier")

    # Draws only from tables already computed in the Usage/Scalars/Transitions tabs.
    readiness = [
        ("cluster usage", "Usage & Clusters tab", st.session_state["cluster_usage"] is not None),
        ("scalars", "Scalars tab", st.session_state["scalars_long"] is not None),
        ("histograms", "Scalars tab", st.session_state["histograms_long"] is not None),
        (
            "transitions", "Transitions tab",
            st.session_state["transitions_long"] is not None
            or st.session_state["transition_cluster_long"] is not None,
        ),
    ]
    missing = [f"{name} ({tab})" for name, tab, ready in readiness if not ready]
    if missing:
        st.caption(
            "Syllable usage is already available (computed as soon as data is loaded). "
            "Not yet computed: " + "; ".join(missing) + " -- compute any of those in their own "
            "tab, if you want them available as feature types here too."
        )

    combined = _combined_long()
    if combined is None:
        st.info("Compute at least one feature table (usage, scalars, or transitions) first.")
    else:
        available_types = sorted(combined["feature_type"].unique())

        with st.expander(f"Available feature types ({len(available_types)})", expanded=False):
            info_df = (
                combined.groupby("feature_type")
                .agg(n_features=("feature_name", "nunique"), n_rows=("value", "size"))
                .reset_index()
                .sort_values("feature_type")
            )
            st.dataframe(info_df, use_container_width=True)
            st.caption(
                "n_features = distinct feature values under that feature type; "
                "n_rows = total (subject, timepoint) x feature rows."
            )

        with st.container(border=True):
            _section_header("Define feature sets to compare")

            st.caption(
                "Quick start: one click adds every currently-computed feature type as its own "
                "comparable set (syllable usage, cluster usage, scalars, and any histogram "
                "channel you've computed all show up here automatically) -- or build a custom "
                "set by hand below."
            )
            if st.button("Quick add: one set per available feature type", key="quick_compare_button"):
                existing_names = {name for name, _ in st.session_state["feature_sets"]}
                added = [ftype for ftype in available_types if ftype not in existing_names]
                for ftype in added:
                    st.session_state["feature_sets"].append((ftype, {"feature_types": [ftype]}))
                if added:
                    st.success(f"Added: {', '.join(added)}")
                else:
                    st.info("Every available feature type already has a set defined below.")

            with st.expander("Build a custom set", expanded=False):
                c1, c2 = st.columns([2, 3])
                with c1:
                    new_set_name = st.text_input("Set name", key="new_set_name")
                with c2:
                    new_set_types = st.multiselect("Feature types", options=available_types, key="new_set_types")

                restrict = st.checkbox(
                    "Restrict to specific features within these feature types "
                    "(e.g. a hand-picked syllable subset, not the whole feature type)",
                    key="new_set_restrict",
                )

                new_set_feature_names = None
                add_complement = False
                if restrict and new_set_types:
                    candidate_names = _sorted_feature_names(
                        combined.loc[
                            combined["feature_type"].isin(new_set_types), "feature_name"
                        ].astype(str).unique(),
                        new_set_types,
                    )
                    new_set_feature_names = st.multiselect(
                        f"Features ({len(candidate_names)} available across the selected feature types)",
                        options=candidate_names, key="new_set_feature_names",
                    )
                    if new_set_feature_names:
                        n_matching = combined[
                            combined["feature_type"].isin(new_set_types)
                            & combined["feature_name"].astype(str).isin(new_set_feature_names)
                        ]["feature_name"].nunique()
                        st.caption(f"This set uses {n_matching} feature(s).")
                        add_complement = st.checkbox(
                            "Also add the complement as its own set (\"everything else\" in these feature types) "
                            "-- a built-in negative control, no need to hand-build it",
                            key="new_set_add_complement",
                        )

                add_disabled = not new_set_name or not new_set_types or (restrict and not new_set_feature_names)
                if st.button("Add set", disabled=add_disabled):
                    spec = {"feature_types": new_set_types}
                    if restrict and new_set_feature_names:
                        spec["feature_names"] = new_set_feature_names
                    st.session_state["feature_sets"].append((new_set_name, spec))

                    if add_complement:
                        complement_names = []
                        for ftype in new_set_types:
                            complement_names.extend(
                                complement_feature_names(combined, ftype, new_set_feature_names)
                            )
                        st.session_state["feature_sets"].append(
                            (f"{new_set_name} (complement)",
                             {"feature_types": new_set_types, "feature_names": complement_names})
                        )
                    st.rerun()

            if st.session_state["feature_sets"]:
                st.caption(f"{len(st.session_state['feature_sets'])} feature set(s) defined:")
                rows = []
                for name, spec in st.session_state["feature_sets"]:
                    if isinstance(spec, dict):
                        ftypes = spec["feature_types"]
                        fnames = spec.get("feature_names")
                    else:
                        ftypes = spec
                        fnames = None
                    rows.append({
                        "Selected": False,
                        "Name": name,
                        "Feature types": ", ".join(ftypes),
                        "Restricted features": len(fnames) if fnames else None,
                    })
                sets_df = pd.DataFrame(rows)
                edited_sets = st.data_editor(
                    sets_df,
                    column_config={
                        "Selected": st.column_config.CheckboxColumn("Select"),
                        "Name": st.column_config.TextColumn(disabled=True),
                        "Feature types": st.column_config.TextColumn(disabled=True),
                        "Restricted features": st.column_config.NumberColumn(
                            disabled=True, help="Count of hand-picked features, if this set is restricted",
                        ),
                    },
                    hide_index=True, use_container_width=True, key="feature_sets_editor",
                )
                n_selected = int(edited_sets["Selected"].sum())
                bulk_cols = st.columns([1, 1, 3])
                if bulk_cols[0].button("Remove selected", disabled=n_selected == 0):
                    keep = edited_sets.index[~edited_sets["Selected"]].tolist()
                    st.session_state["feature_sets"] = [st.session_state["feature_sets"][i] for i in keep]
                    st.rerun()
                if bulk_cols[1].button("Clear all", disabled=not st.session_state["feature_sets"]):
                    st.session_state["feature_sets"] = []
                    st.rerun()
                if n_selected:
                    bulk_cols[2].caption(f"{n_selected} selected")

        with st.container(border=True):
            _section_header("Run")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                y_col_options = ["timepoint"] + [c for c in group_cols if c != "timepoint"]
                y_col = st.selectbox("Classification target", options=y_col_options)
            with c2:
                regularization = st.selectbox("Regularization", options=["l2", "l1"], format_func=str.upper)
            with c3:
                cv_folds = st.number_input("Cross-validation folds", min_value=2, value=5, step=1)
            with c4:
                n_permutations = st.number_input("Permutation tests", min_value=10, value=200, step=10)

            run_disabled = not st.session_state["feature_sets"] or y_col is None
            if st.button("Run classifier comparison", disabled=run_disabled, type="primary"):
                with st.spinner("Running cross-validation and permutation tests..."):
                    try:
                        feature_type_sets = dict(st.session_state["feature_sets"])
                        result = compare_feature_types(
                            combined, feature_type_sets, y_col,
                            cv_folds=cv_folds, regularization=regularization, n_permutations=n_permutations,
                        )
                        st.session_state["classify_result"] = result
                    except Exception as e:
                        st.error(str(e))

        classify_result = st.session_state.get("classify_result")
        if classify_result is not None:
            with st.container(border=True):
                _section_header("Results")

                summary_rows = []
                for name, run_result in classify_result["results"].items():
                    auroc = run_result["mean_scores"].get("auroc_macro")
                    summary_rows.append({
                        "Feature set": name,
                        "Macro F1": run_result["mean_scores"]["macro_f1"],
                        "Balanced accuracy": run_result["mean_scores"]["balanced_accuracy"],
                        "AUROC (macro)": np.nan if auroc is None else auroc,
                        "Permutation p": run_result["permutation_p_value"],
                    })
                summary_df = pd.DataFrame(summary_rows).set_index("Feature set")
                st.dataframe(
                    summary_df.style.format({
                        "Macro F1": "{:.3f}",
                        "Balanced accuracy": "{:.3f}",
                        "AUROC (macro)": "{:.3f}",
                        "Permutation p": "{:.4f}",
                    }),
                    use_container_width=True,
                )

                st.markdown("**Compare feature sets**")
                cc1, cc2 = st.columns(2)
                with cc1:
                    comparison_metric = st.selectbox(
                        "Metric", options=["macro_f1", "balanced_accuracy", "precision_macro", "recall_macro"],
                        key="comparison_metric",
                    )
                with cc2:
                    comparison_style = st.radio(
                        "Style", options=["bar", "box"], horizontal=True, key="comparison_style",
                        help="bar: bar + 68% CI, best for a couple of sets. box: box plot, reads better for 4-5+ sets.",
                    )
                plot_style = st.session_state["plot_style"]
                palette = st.session_state["plot_palette"]
                colorscale = st.session_state["plot_colorscale"]
                # classify_result is only ever replaced (a new dict object) when the
                # classifier is actually re-run, so its id is a cheap, reliable
                # fingerprint for "did the underlying results change" -- far cheaper
                # than hashing the nested dict every rerun.
                result_id = id(classify_result)

                compare_context = (result_id, comparison_metric, comparison_style, plot_style, palette, PLOT_TEMPLATE)

                def _build_compare():
                    if plot_style == "notebook":
                        return mpl_plots.mpl_feature_type_comparison(
                            classify_result, metric=comparison_metric, style=comparison_style,
                            title="Feature set comparison", palette_name=palette,
                        )
                    return plot_feature_type_comparison(
                        classify_result, metric=comparison_metric, style=comparison_style,
                        title="Feature set comparison", palette_name=palette, template=PLOT_TEMPLATE,
                    )

                fig_compare = _cached_result("classify_compare_plot_cache", compare_context, _build_compare)
                _show_plot(fig_compare, "feature_set_comparison.pdf")

                st.markdown("**Per feature-set detail**")
                detail_names = list(classify_result["results"].keys())
                detail_tabs = st.tabs(detail_names)
                for tab, name in zip(detail_tabs, detail_names):
                    run_result = classify_result["results"][name]
                    with tab:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Macro F1", f"{run_result['mean_scores']['macro_f1']:.3f}")
                        m2.metric("Balanced accuracy", f"{run_result['mean_scores']['balanced_accuracy']:.3f}")
                        auroc = run_result["mean_scores"].get("auroc_macro")
                        m3.metric("AUROC (macro)", f"{auroc:.3f}" if auroc is not None and not pd.isna(auroc) else "n/a")
                        m4.metric("Permutation p", f"{run_result['permutation_p_value']:.4f}")
                        st.caption(f"classes: {', '.join(str(c) for c in run_result['classes'])}")

                        fc1, fc2 = st.columns(2)
                        with fc1:
                            cm_context = (result_id, name, "cm", plot_style, colorscale, PLOT_TEMPLATE)

                            def _build_cm(run_result=run_result, name=name):
                                if plot_style == "notebook":
                                    return mpl_plots.mpl_confusion_matrix(
                                        run_result, title=f"{name}: confusion matrix", colorscale=colorscale,
                                    )
                                return plot_confusion_matrix(
                                    run_result, title=f"{name}: confusion matrix",
                                    colorscale=colorscale, template=PLOT_TEMPLATE,
                                )

                            fig_cm = _cached_result(f"classify_{name}_cm_cache", cm_context, _build_cm)
                            _show_plot(fig_cm, f"{name}_confusion_matrix.pdf")
                        with fc2:
                            roc_context = (result_id, name, "roc", plot_style, palette, PLOT_TEMPLATE)

                            def _build_roc(run_result=run_result, name=name):
                                if plot_style == "notebook":
                                    return mpl_plots.mpl_roc(run_result, title=f"{name}: ROC", palette_name=palette)
                                return plot_roc(
                                    run_result, title=f"{name}: ROC", palette_name=palette, template=PLOT_TEMPLATE,
                                )

                            fig_roc = _cached_result(f"classify_{name}_roc_cache", roc_context, _build_roc)
                            _show_plot(fig_roc, f"{name}_roc.pdf")

                with st.expander("View data: pairwise comparisons"):
                    st.dataframe(classify_result["pairwise_comparisons"], use_container_width=True)
                    _df_download_button(
                        classify_result["pairwise_comparisons"], "Download comparisons (CSV)",
                        "classifier_comparisons.csv",
                    )
