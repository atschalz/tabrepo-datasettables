from tabarena.icml2026.plotting.per_dataset_results import plot_model_performance_across_datasets
from tabarena.icml2026.plotting.new_single_prep_boxplots import compare_methods_via_boxplots
from tabarena.icml2026.plotting.two_figures_boxplots import _compute_scores_generic, boxplot_two_dataframes_pubready, boxplot_models_combined_vs_tabprep, boxplot_dataframe_pubready
from tabarena.icml2026.plotting.single_preprocessor_boxplots import ablation_boxplot_colored_by_best
from tabarena.plot.plot_pareto_frontier import get_pareto_frontier, plot_pareto

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
datasets_metadata = load_task_metadata()

from tabarena.nips2025_utils.per_dataset_tables import get_per_dataset_tables

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def _resolve_first_existing_column(df: pd.DataFrame, candidates: list[str], *, label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not find any {label} column in dataframe. Tried: {candidates}. Available: {list(df.columns)}")


def _base_method_name(name: object) -> str:
    text = str(name)
    if " (" in text:
        text = text.split(" (", 1)[0]
    return text.strip()


def plot_pareto_frontier(
    leaderboard,
    metric_col="improvability",
    time_col="time_train_s",
    method_col="method",
    highlight_methods=None,
    agg_func="median",
    minimize_metric=True,
    xscale="log",
    figsize=(10, 6),
    annotate=True,
    annotate_main=True,
    annotate_highlighted=True,
    annotate_additional=True,
    additional_frontiers=None,
    legend=True,
    show_method_labels=True,
    save_path=None,
):
    if highlight_methods is None:
        highlight_methods = []
    if additional_frontiers is None:
        additional_frontiers = []

    def compute_pareto(df_in, minimize_metric=True):
        pareto_front, pareto_names = get_pareto_frontier(
            Xs=df_in["time_value"].tolist(),
            Ys=df_in["metric_value"].tolist(),
            names=df_in[method_col].tolist(),
            max_X=False,
            max_Y=not minimize_metric,
            include_boundary_edges=False,
        )

        frontier_rows = [
            {
                "time_value": x,
                "metric_value": y,
                method_col: name,
            }
            for (x, y), name in zip(pareto_front, pareto_names)
            if name is not None
        ]

        return pd.DataFrame(frontier_rows).sort_values("time_value").reset_index(drop=True)

    df = leaderboard.copy()
    metric_col = _resolve_first_existing_column(
        df,
        [metric_col, "improvability", "bestdiff", "best_diff"],
        label="metric",
    )
    time_col = _resolve_first_existing_column(
        df,
        [
            time_col,
            "time_infer_s_per_1K",
            "median_time_infer_s_per_1K",
            "time_infer_s",
            "median_time_infer_s",
            "time_train_s_per_1K",
            "median_time_train_s_per_1K",
            "time_train_s",
            "median_time_train_s",
        ],
        label="time",
    )
    df = df[[method_col, metric_col, time_col]].dropna()

    method_df = (
        df.groupby(method_col, as_index=False)
          .agg(
              metric_value=(metric_col, agg_func),
              time_value=(time_col, agg_func)
          )
    ).sort_values("time_value").reset_index(drop=True)

    # Main Pareto frontier
    pareto_df = compute_pareto(method_df, minimize_metric=minimize_metric)
    pareto_methods = set(pareto_df[method_col])

    # Highlighted methods not on main frontier
    extra_highlight_df = method_df[
        method_df[method_col].isin(highlight_methods) &
        ~method_df[method_col].isin(pareto_methods)
    ].copy()

    # Additional frontiers
    additional_pareto_dfs = []
    for i, frontier_cfg in enumerate(additional_frontiers):
        methods = frontier_cfg.get("methods", [])
        subset_df = method_df[method_df[method_col].isin(methods)].copy()

        if subset_df.empty:
            frontier_df = pd.DataFrame(columns=method_df.columns)
        else:
            frontier_df = compute_pareto(subset_df, minimize_metric=minimize_metric)

        additional_pareto_dfs.append({
            "label": frontier_cfg.get("label", f"Additional frontier {i + 1}"),
            "df": frontier_df,
            "linestyle": frontier_cfg.get("linestyle", "--"),
            "marker": frontier_cfg.get("marker", "D"),
            "color": frontier_cfg.get("color", None),
            "annotate": frontier_cfg.get("annotate", True),
            "point_size": frontier_cfg.get("point_size", 110),
        })

    # Frontier line colors: one per frontier
    frontier_color_cycle = plt.get_cmap("tab10")
    for i, frontier in enumerate(additional_pareto_dfs):
        if frontier["color"] is None:
            frontier["color"] = frontier_color_cycle(i % 10)

    # Stable colors for any manually annotated frontier points
    all_frontier_methods = list(pareto_df[method_col])
    for frontier in additional_pareto_dfs:
        all_frontier_methods.extend(frontier["df"][method_col].tolist())
    all_frontier_methods = list(dict.fromkeys(all_frontier_methods))
    point_cmap = plt.get_cmap("tab20", max(len(all_frontier_methods), 1))
    method_to_point_color = {
        method: point_cmap(i) for i, method in enumerate(all_frontier_methods)
    }

    plot_df = method_df.rename(columns={method_col: "Method"}).copy() if method_col != "Method" else method_df.copy()
    fig_aspect = figsize[0] / 4.5 if figsize else 2.0

    fig, ax = plot_pareto(
        data=plot_df,
        x_name="time_value",
        y_name="metric_value",
        title=f"Pareto Frontier: {metric_col} vs {time_col}",
        hue="Method",
        max_X=False,
        max_Y=not minimize_metric,
        sort_y=True,
        save_path=None,
        show=False,
        legend_in_plot=True,
        annotate_frontier=annotate and show_method_labels and annotate_main,
        close=False,
        return_fig_ax=True,
        aspect=fig_aspect,
    )

    if xscale != "log":
        ax.set_xscale(xscale)

    # Highlighted non-frontier methods
    for _, row in extra_highlight_df.iterrows():
        ax.scatter(
            row["time_value"],
            row["metric_value"],
            s=140,
            marker="s",
            edgecolors="black",
            linewidths=1.5,
            color="gold",
            label=f"{row[method_col]} (highlighted)",
            zorder=5,
        )

    if annotate and show_method_labels and annotate_highlighted:
        for _, row in extra_highlight_df.iterrows():
            ax.annotate(
                row[method_col],
                (row["time_value"], row["metric_value"]),
                xytext=(6, -10),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

    if annotate and show_method_labels:
        already_annotated = set()
        if annotate_main:
            already_annotated.update(pareto_df[method_col].tolist())
        if annotate_highlighted:
            already_annotated.update(extra_highlight_df[method_col].tolist())
    else:
        already_annotated = set()

    for frontier in additional_pareto_dfs:
        frontier_df = frontier["df"]
        if frontier_df.empty:
            continue

        ax.plot(
            frontier_df["time_value"],
            frontier_df["metric_value"],
            linestyle=frontier["linestyle"],
            linewidth=2,
            color=frontier["color"],
            marker=None,
            label=frontier["label"],
            zorder=2,
        )

        if annotate and show_method_labels and annotate_additional and frontier["annotate"]:
            for _, row in frontier_df.iterrows():
                if row[method_col] in already_annotated:
                    continue
                ax.annotate(
                    row[method_col],
                    (row["time_value"], row["metric_value"]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
                already_annotated.add(row[method_col])

    if not legend:
        for leg in list(fig.legends):
            leg.remove()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return {
        "method_df": method_df,
        "pareto_df": pareto_df,
        "extra_highlight_df": extra_highlight_df,
        "additional_pareto_dfs": additional_pareto_dfs,
        "method_to_point_color": method_to_point_color,
    }

def _build_standard_pareto_plot_df(
    leaderboard: pd.DataFrame,
    *,
    method_col: str = "method",
    metric_col: str = "improvability",
    time_col: str = "time_infer_s_per_1K",
    agg_func: str = "median",
) -> pd.DataFrame:
    df = leaderboard.copy()
    metric_col = _resolve_first_existing_column(
        df,
        [metric_col, "improvability", "bestdiff", "best_diff"],
        label="metric",
    )
    time_col = _resolve_first_existing_column(
        df,
        [
            time_col,
            "time_infer_s_per_1K",
            "median_time_infer_s_per_1K",
            "time_infer_s",
            "median_time_infer_s",
            "time_train_s_per_1K",
            "median_time_train_s_per_1K",
            "time_train_s",
            "median_time_train_s",
        ],
        label="time",
    )

    if "ta_name" in df.columns:
        method_family_col = "ta_name"
    elif "config_type" in df.columns:
        method_family_col = "config_type"
    else:
        method_family_col = method_col

    if "method_subtype" in df.columns:
        subtype_map = {
            "baseline": "Baseline",
            "default": "Default",
            "tuned": "Tuned",
            "tuned_ensemble": "Tuned + Ens.",
            "tuned_ensembled": "Tuned + Ens.",
            "best": "Best",
            "holdout": "Default, Holdout",
            "holdout_tuned": "Tuned, Holdout",
            "holdout_tuned_ensembled": "Tuned + Ens., Holdout",
        }
        df["Type"] = df["method_subtype"].map(subtype_map).fillna(df["method_subtype"])
    else:
        def infer_type(method_name: str) -> str:
            method_name = str(method_name).lower()
            if "(tuned + ensemble)" in method_name or "(tuned+ensemble)" in method_name:
                return "Tuned + Ens."
            if "(tuned, holdout)" in method_name:
                return "Tuned, Holdout"
            if "(tuned + ensemble, holdout)" in method_name:
                return "Tuned + Ens., Holdout"
            if "(holdout)" in method_name:
                return "Default, Holdout"
            if "(tuned)" in method_name:
                return "Tuned"
            if "(best)" in method_name:
                return "Best"
            if "(default)" in method_name:
                return "Default"
            return "Baseline"

        df["Type"] = df[method_col].map(infer_type)

    df["Method"] = df[method_family_col].fillna(df[method_col]).map(_base_method_name)
    group_cols = [method_col]
    agg_spec: dict[str, tuple[str, str]] = {
        "metric_value": (metric_col, agg_func),
        "time_value": (time_col, agg_func),
        "Method": ("Method", "first"),
        "Type": ("Type", "first"),
    }
    if "config_type" in df.columns:
        agg_spec["config_type"] = ("config_type", "first")
    if "ta_name" in df.columns:
        agg_spec["ta_name"] = ("ta_name", "first")
    if "ta_suite" in df.columns:
        agg_spec["ta_suite"] = ("ta_suite", "first")

    return (
        df.groupby(group_cols, as_index=False)
          .agg(**agg_spec)
          .sort_values("time_value")
          .reset_index(drop=True)
    )


def _extend_style_order_and_markers(
    df: pd.DataFrame,
    *,
    style_col: str = "Type",
    style_order: list[str] | None = None,
    style_markers: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    df = df.copy()
    valid_styles = [str(v) for v in pd.unique(df[style_col].dropna())]
    if style_order is None:
        style_order = []
    if style_markers is None:
        style_markers = {}

    ordered_styles = [style for style in style_order if style in valid_styles]
    ordered_styles.extend(style for style in valid_styles if style not in ordered_styles)

    marker_cycle = ["o", "s", "X", "D", "*", "^", "P", "v", "<", ">", "h", "H", "d", "p"]
    used_markers = set(style_markers.values())
    extended_markers = dict(style_markers)
    marker_idx = 0
    for style in ordered_styles:
        if style in extended_markers:
            continue
        while marker_cycle[marker_idx % len(marker_cycle)] in used_markers:
            marker_idx += 1
        marker = marker_cycle[marker_idx % len(marker_cycle)]
        extended_markers[style] = marker
        used_markers.add(marker)
        marker_idx += 1

    return ordered_styles, extended_markers

method_style_map: dict[str, dict[str, object]] = {
    "LR": {"color": "#CC7C7CFF", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "(Prep)Linear": {"color": "#9C1818", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "TABM_GPU": {"color": "#7FD48A", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "TabM": {"color": "#7FD48A", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "(Prep)TabM": {"color": "#17810D", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "GBM": {"color": "#DF884F", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "(Prep)LightGBM": {"color": "#D15E11", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "REALTABPFN-V2.5": {"color": "#5F94DA", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "RealTabPFN-v2.5": {"color": "#5F94DA", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "RealTabPFN2.5": {"color": "#5F94DA", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "(Prep)RealTabPFN-2.5": {"color": "#16419E", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},


    "PrepDefaultEnsemble": {"color": "#E6AB02", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "PrepEnsemble": {"color": "#E6AB02", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 165},
    "DefaultEnsemble": {"color": "#A6761D", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "Prep2RealTabPFN2.5": {"color": "#76B7B2", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "Prep2TabM": {"color": "#F28E2B", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "Prep2Linear": {"color": "#B07AA1", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "Prep2LightGBM": {"color": "#EDC948", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "PrepRealTabPFN2.5": {"color": "#76B7B2", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "PrepTabM": {"color": "#F28E2B", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "PrepLinearModel": {"color": "#B07AA1", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    # "PrepLightGBM": {"color": "#EDC948", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "TABICLV2": {"color": "#66A61E", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "RF": {"color": "#D8DD95", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "CAT": {"color": "#633820", "alpha": 0.92, "edgecolors": "black", "linewidths": 0.35, "s": 115},
    "Other": {"color": "#9e9e9e", "alpha": 0.28, "edgecolors": "#7a7a7a", "linewidths": 0.35, "s": 105},
}

def _build_method_style_map() -> dict[str, dict[str, object]]:
    # Manual, editable mapping for the method colors used in the Pareto plot.
    # Keep the base methods and their aliases here so the legend is stable and easy to tune.

    return method_style_map

ablation_base_path = "//ceph/atschalz/auto_prep/tabarena_figs/icml_ablation"
base_path = "//ceph/atschalz/auto_prep/tabarena_figs/icml_final/"
comb_path = "//ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined/"
save_path = "//ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new"

parquet_path = "/ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined/"

if __name__ == "__main__":

    ta_context = TabArenaContext()
    # ta_context.load_configs_hyperparameters(methods = ["PrepLightGBM", "PrepLinearModel"], download=True)
    # ta_context.load_results_paper(methods=["PrepLightGBM", "PrepLinearModel"])
    ta_results = pd.concat([ta_context.load_hpo_results(i) for i in ta_context.methods if "AutoGluon" not in i]).reset_index(drop=True)
    # ta_results.dataset = ta_results.dataset.apply(lambda x: dat_map.get(x, x))


    results = ta_context.load_config_results("PrepLightGBM")
    hpo_results = ta_context.load_hpo_results("PrepLightGBM")
    # results.dataset = results.dataset.apply(lambda x: dat_map.get(x, x))
    # hpo_results.dataset = hpo_results.dataset.apply(lambda x: dat_map.get(x, x))

    all_model_results = pd.DataFrame()
    all_hpo_results = pd.DataFrame()
    models = ["prep_TabM", "prep_RealTabPFN"] #, "prep_RealMLP"]
    for model_name in models:
        model_results = pd.read_csv(f"{base_path}/{model_name}/model_results.csv")
        model_results["model_name"] = model_name
        # model_results.dataset = model_results.dataset.apply(lambda x: dat_map.get(x, x))
        all_model_results = pd.concat([all_model_results, model_results]).reset_index(drop=True)

        hpo_results = pd.read_csv(f"{base_path}/{model_name}/hpo_results.csv")
        hpo_results["model_name"] = model_name
        # hpo_results.dataset = hpo_results.dataset.apply(lambda x: dat_map.get(x, x))
        all_hpo_results = pd.concat([all_hpo_results, hpo_results]).reset_index(drop=True)

    all_model_results.ta_name = all_model_results.ta_name.map({"prep_TabM": "PrepTabM", 
                                                    "RealTabPFN-v2.5": "RealTabPFN2.5", 
                                                    "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
                                                    "TabM_GPU": "TabM"}).fillna(all_model_results.ta_name)

    new_res = pd.read_csv("/ceph/atschalz/irreplaceability/results/results_per_split.csv")
    tabiclnew_res = new_res[new_res.method=='TABICLV2 (default)']

    comb_results = pd.concat([
        all_hpo_results,#[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]], 
        ta_results,#[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]]
        pd.read_parquet(f"{parquet_path}/RealTabPFN-2.5.parquet"),
        pd.read_parquet(f"{parquet_path}/TabM.parquet"),
        pd.read_parquet(f"{parquet_path}/Linear.parquet"),
        pd.read_parquet(f"{parquet_path}/LightGBM.parquet"),
        tabiclnew_res,
    ]).reset_index(drop=True)

    use_cols = ["dataset", "metric_error", "metric_error_val", "time_train_s", "time_infer_s"]

    for model in ["TabM", "RealTabPFN2.5", "LightGBM", "Linear"]:
        for fold in [0]: # ta_results.fold.unique():
            ta_results_f0 = ta_results.loc[ta_results.fold==fold]

            if model == "TabM":
                ta_results_new = ta_results.loc[np.logical_and(ta_results.fold==fold, ta_results.method=="TABM_GPU (default)")].copy()
                all_hpo_results_f0 = all_hpo_results.loc[all_hpo_results.fold==fold]

                merged = pd.merge(
                    ta_results_f0.loc[ta_results_f0.method=="TABM_GPU (default)",use_cols],
                    all_hpo_results_f0.loc[all_hpo_results_f0.method=="prep_TABM (default)",use_cols],
                    on="dataset",
                    suffixes=("base", "prep")
                )
            elif model == "RealTabPFN2.5":
                ta_results_new = ta_results.loc[np.logical_and(ta_results.fold==fold, ta_results.method=="REALTABPFN-V2.5 (default)")].copy()
                all_hpo_results_f0 = all_hpo_results.loc[all_hpo_results.fold==fold]

                merged = pd.merge(
                    ta_results_f0.loc[ta_results_f0.method=="REALTABPFN-V2.5 (default)",use_cols],
                    all_hpo_results_f0.loc[all_hpo_results_f0.method=="prep_REALTABPFN-V2.5 (default)",use_cols],
                    on="dataset",
                    suffixes=("base", "prep")
                )
            elif model == "LightGBM":
                ta_results_new = ta_results.loc[np.logical_and(ta_results.fold==fold, ta_results.method=="GBM (default)")].copy()
                results_f0 = results.loc[results.fold==fold]
                merged = pd.merge(
                    ta_results_f0.loc[ta_results_f0.method=="GBM (default)",use_cols],
                    results_f0.loc[results_f0.method=="prep_LightGBM_icml_v3_c1_BAG_L1",use_cols],
                    on="dataset",
                    suffixes=("base", "prep")
                )
            elif model == "Linear":
                results_f0 = results.loc[results.fold==fold]
                merged = pd.merge(
                    ta_results_f0.loc[ta_results_f0.method=="LR (default)",use_cols],
                    ta_results_f0.loc[ta_results_f0.method=="prep_LR (default)",use_cols],
                    on="dataset",
                    suffixes=("base", "prep")
                )

            merged.index = merged.dataset
            merged = merged.drop(columns=["dataset"])
            merged["selected"] = merged.apply(lambda x: x["metric_errorbase"] if x["metric_error_valbase"] < x["metric_error_valprep"] else x["metric_errorprep"], axis=1)

            merged.style.highlight_min(subset=["metric_errorbase", "metric_errorprep", "selected"], color="lightgreen",axis=1)

            for row in merged.index:
                if merged.loc[row, "metric_error_valbase"] > merged.loc[row, "metric_error_valprep"]:
                    ta_results_new.loc[ta_results_new.dataset==row, "metric_error"] = merged.loc[row, "metric_errorprep"]
                    ta_results_new.loc[ta_results_new.dataset==row, "time_train_s"] = merged.loc[row, "time_train_sprep"]
                    ta_results_new.loc[ta_results_new.dataset==row, "time_infer_s"] = merged.loc[row, "time_infer_sprep"]


            ta_results_new["method"] = f"Prep2{model} (default)"
            ta_results_new["ta_name"] = f"Prep2{model}"
            ta_results_new["config_type"] = f"Prep2{model}"
            ta_results_new["ta_suite"] = f"Prep2{model}"
            comb_results = pd.concat([comb_results, ta_results_new]).reset_index(drop=True)

    default_ensemble = pd.concat([
        comb_results.loc[comb_results.method=="TABM_GPU (default)"],
        comb_results.loc[comb_results.method=="REALTABPFN-V2.5 (default)"],
        comb_results.loc[comb_results.method=="GBM (default)"],
        comb_results.loc[comb_results.method=="LR (default)"]
        ],axis=0).copy()

    default_ensemble = default_ensemble.loc[default_ensemble.fold==0].copy()

    default_ensemble["method"] = f"DefaultEnsemble (default)"
    default_ensemble["ta_name"] = f"DefaultEnsemble"
    default_ensemble["config_type"] = f"DefaultEnsemble"
    default_ensemble["ta_suite"] = f"DefaultEnsemble"
    # default_ensemble["model_name"] = f"DefaultEnsemble"

    default_ensemble_use = pd.DataFrame()

    for dat,idx in default_ensemble.groupby("dataset").apply(lambda x: x["metric_error_val"].idxmin()).items():
        default_ensemble_use = pd.concat([default_ensemble_use, default_ensemble.loc[[idx]]], axis=0)
    default_ensemble_use = default_ensemble_use.reset_index(drop=True)

    comb_results = pd.concat([comb_results, default_ensemble_use]).reset_index(drop=True)

    ####

    prep_ensemble = pd.concat([
        comb_results.loc[comb_results.method=="prep_TABM (default)"],
        comb_results.loc[comb_results.method=="prep_REALTABPFN-V2.5 (default)"],
        comb_results.loc[comb_results.method=="PREP_GBM (default)"],
        comb_results.loc[comb_results.method=="PREP_LR (default)"]
        ],axis=0).copy()
    prep_ensemble = prep_ensemble.loc[prep_ensemble.fold==0].copy()
    prep_ensemble["method"] = f"PrepEnsemble (default)"
    prep_ensemble["ta_name"] = f"PrepEnsemble"
    prep_ensemble["config_type"] = f"PrepEnsemble"
    prep_ensemble["ta_suite"] = f"PrepEnsemble"
    # prep_ensemble["model_name"] = f"PrepEnsemble (default)"

    prep_ensemble_use = pd.DataFrame()
    for dat,idx in prep_ensemble.groupby("dataset").apply(lambda x: x["metric_error"].idxmin()).items():
        prep_ensemble_use = pd.concat([prep_ensemble_use, prep_ensemble.loc[[idx]]], axis=0)
        prep_ensemble_use.loc[prep_ensemble_use.dataset==dat,"time_train_s"] = prep_ensemble_use.loc[prep_ensemble_use.dataset==dat,"time_train_s"].sum()
        prep_ensemble_use.loc[prep_ensemble_use.dataset==dat,"time_infer_s"] = prep_ensemble_use.loc[prep_ensemble_use.dataset==dat,"time_infer_s"].sum()
    prep_ensemble_use = prep_ensemble_use.reset_index(drop=True)

    comb_results = pd.concat([comb_results, prep_ensemble_use]).reset_index(drop=True)

    #### PREP

    prep_default_ensemble = pd.concat([
        comb_results.loc[comb_results.method=="TABM_GPU (default)"],
        comb_results.loc[comb_results.method=="REALTABPFN-V2.5 (default)"],
        comb_results.loc[comb_results.method=="GBM (default)"],
        comb_results.loc[comb_results.method=="LR (default)"],
        comb_results.loc[comb_results.method=="prep_TABM (default)"],
        comb_results.loc[comb_results.method=="prep_REALTABPFN-V2.5 (default)"],
        comb_results.loc[comb_results.method=="PREP_GBM (default)"],
        comb_results.loc[comb_results.method=="PREP_LR (default)"]
        ],axis=0).copy()
    prep_default_ensemble = prep_default_ensemble.loc[prep_default_ensemble.fold==0].copy()
    prep_default_ensemble["method"] = f"PrepDefaultEnsemble (default)"
    prep_default_ensemble["ta_name"] = f"PrepDefaultEnsemble"
    prep_default_ensemble["config_type"] = f"PrepDefaultEnsemble"
    prep_default_ensemble["ta_suite"] = f"PrepDefaultEnsemble"
    # prep_default_ensemble["model_name"] = f"PrepDefaultEnsemble (default)"

    prep_default_ensemble_use = pd.DataFrame()
    for dat,idx in prep_default_ensemble.groupby("dataset").apply(lambda x: x["metric_error"].idxmin()).items():
        prep_default_ensemble_use = pd.concat([prep_default_ensemble_use, prep_default_ensemble.loc[[idx]]], axis=0)
        prep_default_ensemble_use.loc[prep_default_ensemble_use.dataset==dat,"time_train_s"] = prep_default_ensemble_use.loc[prep_default_ensemble_use.dataset==dat,"time_train_s"].sum()
        prep_default_ensemble_use.loc[prep_default_ensemble_use.dataset==dat,"time_infer_s"] = prep_default_ensemble_use.loc[prep_default_ensemble_use.dataset==dat,"time_infer_s"].sum()
    prep_default_ensemble_use = prep_default_ensemble_use.reset_index(drop=True)

    comb_results = pd.concat([comb_results, prep_default_ensemble_use]).reset_index(drop=True)

    import os
    lb_path = "/ceph/atschalz/auto_prep/___lb/tabarena_leaderboard.csv"
    if not os.path.exists(lb_path):
        leaderboard = ta_context.compare("/ceph/atschalz/auto_prep/___lb", subset="lite", ta_results=comb_results.loc[comb_results.fold==0])
    else:
        leaderboard = pd.read_csv(lb_path)
    leaderboard.head(50)

    # Methods to highlight even if they are not on the Pareto frontier
    highlight_methods = [
        # "PREP_LR (default)",
        # "PREP_GBM (default)",
        # "PREP_LR (tuned)",
        # "PREP_GBM (tuned)",
        # "PREP_LR (tuned + ensemble)",
        # "PREP_GBM (tuned + ensemble)",
        "DefaultEnsemble (default)",
        "PrepEnsemble (default)"
    ]


    # highlight_methods = [
    # 'Prep2LightGBM (default)',
    #  'Prep2RealTabPFN2.5 (default)',
    #  'Prep2Linear (default)',
    #  'Prep2TabM (default)']

    cpu = ['CatBoost', 'ExtraTrees',
        'LightGBM', 'RandomForest',
        'XGBoost', 
        'ExplainableBM', 
        'LinearModel', 'KNeighbors', 'PrepLightGBM',
        'PrepLinearModel', 
        '(Prep)Linear',
        '(Prep)LightGBM',
        'Prep2LightGBM', 'Prep2Linear']


    cpu_frontier_methods = comb_results.method[comb_results.ta_name.apply(lambda x: x in cpu)].unique().tolist()
    non_prep_frontier_methods = leaderboard.method[leaderboard.method.apply(lambda x: "prep" not in x.lower())].values.tolist()
    cpu_nontabprep = [i for i in cpu_frontier_methods if i in non_prep_frontier_methods]


    results = plot_pareto_frontier(
        leaderboard,
        metric_col="improvability",
        time_col="time_infer_s_per_1K",
        highlight_methods=highlight_methods,
        minimize_metric=True,
        xscale="log",
        # secondary_frontier_methods=secondary_frontier_methods,
        # secondary_frontier_label="CPU-only frontier",
        additional_frontiers=[
            {
                "methods": cpu_frontier_methods,
                "label": "CPU-only frontier",
                "linestyle": "--",
                "marker": "D",
                "annotate": True,
            },
            {
                "methods": non_prep_frontier_methods,
                "label": "Frontier without TabPrep",
                "linestyle": ":",
                "marker": "s",
                "annotate": True,
            },
            {
                "methods": cpu_nontabprep,
                "label": "CPU-only frontier without TabPrep",
                "linestyle": "--",
                "marker": "D",
                "annotate": True,
            },

        ],
        annotate_main=False,
        annotate_additional=False,
        save_path="/ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new/pareto_withdefault.pdf",
    )

    include = [
       'PrepDefaultEnsemble (default)', 'PrepEnsemble (default)', 'DefaultEnsemble (default)', 
       'Prep2RealTabPFN2.5 (default)', 'Prep2TabM (default)', 'Prep2Linear (default)', 'Prep2LightGBM (default)',  
       
       
       '(Prep)RealTabPFN-2.5 (tuned + ensemble)', '(Prep)RealTabPFN-2.5 (tuned)', '(Prep)RealTabPFN-2.5 (default)',
       '(Prep)LightGBM (tuned + ensemble)', '(Prep)LightGBM (tuned)','(Prep)LightGBM (default)',
       '(Prep)TabM (tuned + ensemble)', '(Prep)TabM (tuned)' ,'(Prep)TabM (default)', 
       '(Prep)Linear (tuned + ensemble)', '(Prep)Linear (tuned)', '(Prep)Linear (default)',

       'REALTABPFN-V2.5 (tuned + ensemble)', 'REALTABPFN-V2.5 (tuned)', 'REALTABPFN-V2.5 (default)', 
       'TABM_GPU (tuned + ensemble)', 'TABM_GPU (tuned)', 'TABM_GPU (default)',
       'GBM (tuned + ensemble)', 'GBM (tuned)', 'GBM (default)',
       'LR (tuned + ensemble)', 'LR (tuned)', 'LR (default)',

       'TABICLV2 (default)',      
    #    'RF (tuned + ensemble)',  'RF (tuned)', 'RF (default)', 
       'EBM (tuned + ensemble)',  'EBM (tuned)', 'EBM (default)', 
       'CAT (tuned + ensemble)',  'CAT (tuned)', 'CAT (default)', 
       
       
    #    'PREP_GBM (tuned + ensemble)', 'PREP_GBM (tuned)',
        #'prep_REALTABPFN-V2.5 (default)',
        #'prep_TABM (default)',
    #    'PREP_LR (tuned + ensemble)', 'PREP_LR (tuned)',
    #    'PREP_LR (default)', 
       
       

    ]

    name_maps = {
        "RF": "RandomForest",
        "CAT": "CatBoost",
        "LR": "Linear",
        "GBM": "LightGBM",
        "EBM": "EBM",
         "TABM_GPU": "TabM",
         "REALTABPFN-V2.5": "TabPFN2.5",
         "(Prep)TabM": "PrepTabM",
         "(Prep)RealTabPFN-2.5": "PrepTabPFN2.5",
         "(Prep)LightGBM": "PrepLightGBM",
         "(Prep)Linear": "PrepLinear",
         "DefaultEnsemble": "DefaultEnsemble",
        #  "PrepEnsemble": "PrepEnsemble",
        "PrepDefaultEnsemble": "PrepEnsemble",
    }
    color_legend_order = [
        "Other",
        "RandomForest",
        "EBM",
        "CatBoost",
        "TABICLV2",
        "Linear",
        "PrepLinear",
        "LightGBM",
        "PrepLightGBM",
        "TabM",
        "PrepTabM",
        "TabPFN2.5",
        "PrepTabPFN2.5",
        "DefaultEnsemble",
        "PrepEnsemble",
    ]

    def render_pareto_plot(*, time_col: str, display_x_name: str, title: str, save_path: str):
        standard_pareto_df = _build_standard_pareto_plot_df(
            leaderboard=leaderboard,
            method_col="method",
            metric_col="improvability",
            time_col=time_col,
            agg_func="median",
        )
        standard_pareto_df.loc[~standard_pareto_df["method"].isin(include), "Method"] = "Other"
        standard_pareto_df["MarkerType"] = standard_pareto_df["Type"]
        star_methods = {
            "DefaultEnsemble (default)",
            "PrepEnsemble (default)",
            "PrepDefaultEnsemble (default)",
        }
        standard_pareto_df.loc[standard_pareto_df["method"].isin(star_methods), "MarkerType"] = "Ensemble"

        base_style_order = [
            "Default",
            "Tuned",
            "Ensemble",
            "Baseline",
            "Best",
            "Default, Holdout",
            "Tuned, Holdout",
            "Tuned + Ens.",
            "Tuned + Ens., Holdout",
        ]
        base_style_markers = {
            "Default": "o",
            "Tuned": "s",
            "Ensemble": "*",
            "Baseline": "D",
            "Best": "*",
            "Default, Holdout": "o",
            "Tuned, Holdout": "s",
            "Tuned + Ens.": "X",
            "Tuned + Ens., Holdout": "X",
        }
        style_order, style_markers = _extend_style_order_and_markers(
            standard_pareto_df,
            style_col="MarkerType",
            style_order=base_style_order,
            style_markers=base_style_markers,
        )

        focused_pareto_df = standard_pareto_df[standard_pareto_df["Method"] != "Other"].copy()
        other_pareto_df = standard_pareto_df[standard_pareto_df["Method"] == "Other"].copy()
        method_style_map = _build_method_style_map()
        display_y_name = "Improvability (%)"
        focused_pareto_df[display_x_name] = focused_pareto_df["time_value"]
        focused_pareto_df[display_y_name] = focused_pareto_df["metric_value"]
        focused_pareto_df[display_y_name] *= 100

        focused_pareto_df["Method"] = focused_pareto_df["Method"].apply(lambda x: name_maps.get(x, x))
        iter_ = method_style_map.copy()
        for name in iter_:
            if name in name_maps:
                method_style_map[name_maps[name]] = iter_[name]

        fig, ax = plot_pareto(
            data=focused_pareto_df,
            x_name=display_x_name,
            y_name=display_y_name,
            title=title,
            hue="Method",
            hue_style_map=method_style_map,
            hue_order=color_legend_order,
            style_col="MarkerType",
            style_order=style_order,
            style_markers=style_markers,
            label_col="method",
            max_X=False,
            max_Y=False,
            sort_y=True,
            save_path=None,
            show=False,
            legend_in_plot=True,
            annotate_frontier=True,
            aspect=4 / 3,
            close=False,
            return_fig_ax=True,
        )

        if not other_pareto_df.empty:
            other_pareto_df = other_pareto_df.copy()
            other_pareto_df[display_y_name] = other_pareto_df["metric_value"] * 100
            other_marker_map = {lvl: style_markers.get(lvl, "o") for lvl in pd.unique(other_pareto_df["MarkerType"])}
            for type_level, subset in other_pareto_df.groupby("MarkerType", dropna=False):
                style_spec = method_style_map["Other"]
                ax.scatter(
                    subset["time_value"],
                    subset[display_y_name],
                    s=style_spec["s"],
                    alpha=style_spec["alpha"],
                    linewidths=style_spec["linewidths"],
                    edgecolors=style_spec["edgecolors"],
                    color=style_spec["color"],
                    marker=other_marker_map.get(type_level, "o"),
                    label="_nolegend_",
                    zorder=1.5,
                )

        fig.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)

    render_pareto_plot(
        time_col="median_time_infer_s_per_1K",
        display_x_name="Inference time per 1K samples (s) (median)",
        title="Pareto Frontier: Improvability vs Inference time per 1K samples (s) (median)",
        save_path="/ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new/pareto_standard.pdf",
    )

    render_pareto_plot(
        time_col="median_time_train_s_per_1K",
        display_x_name="Training time per 1K samples (s) (median)",
        title="Pareto Frontier: Improvability vs Training time per 1K samples (s) (median)",
        save_path="/ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new/pareto_train_standard.pdf",
    )
