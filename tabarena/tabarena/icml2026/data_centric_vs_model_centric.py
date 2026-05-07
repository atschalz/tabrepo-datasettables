from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ablation_base_path = "//ceph/atschalz/auto_prep/tabarena_figs/icml_ablation"
base_path = "//ceph/atschalz/auto_prep/tabarena_figs/icml_final/"
comb_path = "//ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined/"
save_path = "//ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new"
use_folds = [0,1,2]
# Default upper anchor for the min-normalized score.
normalization_upper = "second-best" # ['median', 'max', 'third-best', 'second-best']
y_axis_metric = "relative_improvement_over_best_on_ta" # ['normalized_error', 'relative_improvement_over_best_on_ta']
use_four_tabprep_models = True
show_best_tabprep_model_per_dataset_only = True
y_axis_log_scale = True
equidistant_y_ticks = False
# Relative-improvement gap range between the top two TabPrep models that triggers a green star.
green_star_improvement_range = (0.0, 0.0000000000) #(0.0, 0.005)

dat_map = {
    "HR_Analytics_Job_Change_of_Data_Scientists": "HR_Analytics",
    "students_dropout_and_academic_success": "students_dropout",
    "blood-transfusion-service-center": "blood-transfusion",
    "Another-Dataset-on-used-Fiat-500": "Fiat-500",
    "coil2000_insurance_policies": "coil2000",
    "hazelnut-spread-contaminant-detection": "hazelnut-spread",
    "taiwanese_bankruptcy_prediction": "taiwanese_bankruptcy",
    "polish_companies_bankruptcy": "polish_companies",
    "healthcare_insurance_expenses": "healthcare_insurance",
    "in_vehicle_coupon_recommendation": "in_vehicle",
    "Amazon_employee_access": "Amazon_employee",
    "concrete_compressive_strength": "concrete",
    "customer_satisfaction_in_airline": "customer_satisfaction",
    "E-CommereShippingData": "E-Commerce",
    "online_shoppers_intention": "online_shoppers",
    "Is-this-a-good-customer": "Is-good-customer",
    "Food_Delivery_Time": "Food_Delivery",
    "credit_card_clients_default": "credit_card_clients",
}

prep_model_name_map = {
    "(Prep)LightGBM": "PrepLightGBM",
    "(Prep)Linear": "PrepLinearModel",
    "(Prep)RealTabPFN-2.5": "PrepRealTabPFN2.5",
    "(Prep)TabM": "PrepTabM",
}

fold_suffix = "folds_" + "_".join(str(fold) for fold in use_folds)


def _select_normalization_upper(values: pd.Series, choice: str) -> float:
    sorted_values = values.sort_values(kind="stable").reset_index(drop=True)
    if sorted_values.empty:
        raise ValueError("Cannot select a normalization anchor from an empty series.")

    if choice == "second-best":
        return float(sorted_values.iloc[min(1, len(sorted_values) - 1)])
    if choice == "third-best":
        return float(sorted_values.iloc[min(2, len(sorted_values) - 1)])
    if choice == "median":
        return float(sorted_values.median())
    if choice == "max":
        return float(sorted_values.iloc[-1])

    raise ValueError(
        "normalization_upper must be one of "
        "['median', 'max', 'third-best', 'second-best']"
    )


def _format_normalization_upper(choice: str) -> str:
    if choice == "median":
        return "Median"
    if choice == "second-best":
        return "2nd best"
    if choice == "third-best":
        return "3rd best"
    if choice == "max":
        return "Worst"
    raise ValueError(f"Unsupported normalization_upper value: {choice}")


def _format_y_axis_metric(choice: str) -> tuple[str, str, float]:
    if choice == "normalized_error":
        return "Normalized score", "normalized_error", 1.0
    if choice == "relative_improvement_over_best_on_ta":
        return "Relative improvement", "relative_improvement_over_best_on_ta", 0.0
    raise ValueError(
        "y_axis_metric must be one of "
        "['normalized_error', 'relative_improvement_over_best_on_ta']"
    )


def _add_grouped_legend(
    ax,
    model_order: list[str],
    model_to_handle: dict[str, object],
    star_counts: dict[str, int],
    use_four_tabprep_models: bool,
    prep_model_order: list[str],
    tabprep_group_label: str = "TabPrep",
    anchor_y: float = 1.12,
) -> None:
    display_name_map = {
        "TabPFN-v2.6": "TabPFN-2.6",
        "PrepLinearModel": "PrepLinear",
        "PrepRealTabPFN2.5": "PrepTabPFN-2.5",
    }

    def display_model_name(model: str) -> str:
        return display_name_map.get(model, model)

    legend_labels = {
        model: f"{display_model_name(model)} [{star_counts.get(model, 0)}]"
        for model in model_order
    }

    if use_four_tabprep_models:
        legend_fontsize = 13
        other_models = [model for model in model_order if model not in prep_model_order]
        other_handles = [model_to_handle[model] for model in other_models if model in model_to_handle]
        other_labels = [legend_labels[model] for model in other_models if model in model_to_handle]
        prep_handles = [model_to_handle[model] for model in prep_model_order if model in model_to_handle]
        prep_labels = [legend_labels[model] for model in prep_model_order if model in model_to_handle]

        if other_handles:
            other_legend = ax.legend(
                other_handles,
                other_labels,
                loc="upper left",
                bbox_to_anchor=(0.0, anchor_y),
                ncol=len(other_handles),
                frameon=False,
                fontsize=legend_fontsize,
                borderaxespad=0.0,
                handletextpad=0.25,
                columnspacing=0.55,
                markerscale=1.35,
            )
            ax.add_artist(other_legend)

        if prep_handles:
            prep_handles = [Line2D([], [], linestyle="none", marker=None, color="none")] + prep_handles
            prep_labels = [tabprep_group_label] + prep_labels
            prep_legend = ax.legend(
                prep_handles,
                prep_labels,
                loc="upper right",
                bbox_to_anchor=(1.0, anchor_y),
                ncol=len(prep_handles),
                frameon=True,
                fontsize=legend_fontsize,
                borderaxespad=0.0,
                handletextpad=0.25,
                columnspacing=0.55,
                markerscale=1.35,
            )
            frame = prep_legend.get_frame()
            frame.set_edgecolor("green")
            frame.set_facecolor("white")
            prep_texts = prep_legend.get_texts()
            if prep_texts:
                prep_texts[0].set_color("green")
                prep_texts[0].set_fontweight("bold")
        return

    handles = [model_to_handle[model] for model in model_order if model in model_to_handle]
    labels = [legend_labels[model] for model in model_order if model in model_to_handle]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=len(model_order),
        frameon=False,
        fontsize=13,
        borderaxespad=0.0,
        handletextpad=0.25,
        columnspacing=0.55,
        markerscale=1.35,
    )


def _equal_spaced_axis_scale(values: list[float]) -> tuple:
    anchor_values = np.array(sorted(set(values)), dtype=float)
    anchor_positions = np.arange(len(anchor_values), dtype=float)

    if anchor_values.size == 1:
        def forward(y):
            y_arr = np.asarray(y, dtype=float)
            return np.zeros_like(y_arr, dtype=float)

        def inverse(y):
            y_arr = np.asarray(y, dtype=float)
            return np.full_like(y_arr, anchor_values[0], dtype=float)

        return forward, inverse

    left_forward_slope = (anchor_positions[1] - anchor_positions[0]) / (anchor_values[1] - anchor_values[0])
    right_forward_slope = (anchor_positions[-1] - anchor_positions[-2]) / (anchor_values[-1] - anchor_values[-2])
    left_inverse_slope = (anchor_values[1] - anchor_values[0]) / (anchor_positions[1] - anchor_positions[0])
    right_inverse_slope = (anchor_values[-1] - anchor_values[-2]) / (anchor_positions[-1] - anchor_positions[-2])

    def forward(y):
        y_arr = np.asarray(y, dtype=float)
        mapped = np.interp(y_arr, anchor_values, anchor_positions)
        mapped = np.where(
            y_arr < anchor_values[0],
            anchor_positions[0] + (y_arr - anchor_values[0]) * left_forward_slope,
            mapped,
        )
        mapped = np.where(
            y_arr > anchor_values[-1],
            anchor_positions[-1] + (y_arr - anchor_values[-1]) * right_forward_slope,
            mapped,
        )
        return mapped

    def inverse(y):
        y_arr = np.asarray(y, dtype=float)
        mapped = np.interp(y_arr, anchor_positions, anchor_values)
        mapped = np.where(
            y_arr < anchor_positions[0],
            anchor_values[0] + (y_arr - anchor_positions[0]) * left_inverse_slope,
            mapped,
        )
        mapped = np.where(
            y_arr > anchor_positions[-1],
            anchor_values[-1] + (y_arr - anchor_positions[-1]) * right_inverse_slope,
            mapped,
        )
        return mapped

    return forward, inverse

if __name__ == "__main__":
    comb_results = pd.read_csv("/ceph/atschalz/auto_prep/tabarena/tabarena/comb_results_meta.csv")
    comb_results.dataset = comb_results.dataset.apply(lambda x: dat_map.get(x, x))
    comb_results = comb_results.loc[comb_results.fold.isin(use_folds)].reset_index(drop=True)

    comb_results_use = comb_results.loc[comb_results["method_subtype"] == "tuned_ensemble"].copy()
    comb_results_use_bar = comb_results_use.copy()
    for m in ["LightGBM", "Linear", "RealTabPFN-2.5", "TabM"]:
        comb_results_use_bar = pd.concat([comb_results_use_bar, pd.read_parquet(f"{comb_path}/{m}.parquet")]).reset_index(drop=True)

    comb_results_use_bar = comb_results_use_bar.loc[comb_results_use_bar.fold.isin(use_folds)]
    comb_results_use_bar = comb_results_use_bar[['dataset', 'fold', 'ta_name', 'metric_error', 'metric_error_val','method_subtype']]
    comb_results_use_bar.dataset = comb_results_use_bar.dataset.apply(lambda x: dat_map.get(x, x))

    # comb_results_use_bar.ta_name = comb_results_use_bar.ta_name.map({
    #     "(Prep)TabM": "PrepTabM", 
    #     "(Prep)RealTabPFN-v2.5": "RealTabPFN2.5", 
    #     "(Prep)RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
    #     "(Prep)TabM": "TabM"}).fillna(comb_results_use_bar.ta_name)
    



    # comb_results_use_bar = comb_results_use.copy()

    # for dataset_name in comb_results_use_bar.dataset.unique():

    #     for model_name in ["TabM", "LinearModel", "LightGBM", "RealTabPFN2.5"]:
    #         prep = comb_results_use_bar.loc[np.logical_and(comb_results_use_bar.dataset==dataset_name, comb_results_use_bar.ta_name==f"Prep{model_name}")]
    #         base = comb_results_use_bar.loc[np.logical_and(comb_results_use_bar.dataset==dataset_name, comb_results_use_bar.ta_name==model_name)]
    #         if prep.shape[0]==0 or base.shape[0]==0:
    #             continue
    #         if base.metric_error_val.values[0] < prep.metric_error_val.values[0]:       
    #             comb_results_use_bar.loc[np.logical_and(comb_results_use_bar.dataset==dataset_name, comb_results_use_bar.ta_name==f"Prep{model_name}"),"metric_error"] = base.metric_error.values[0]
    #             # print(f"{model_name}: {base.metric_error_val.values[0]:.4f}, {prep.metric_error_val.values[0]:.4f}")
    
    comb_results_use.loc[comb_results_use.method_subtype=="default","ta_name"] += "_default"
    comb_results_use.loc[comb_results_use.method_subtype=="tuned","ta_name"] += "_tuned"
    comb_results_use_bar.loc[comb_results_use_bar.method_subtype=="default","ta_name"] += "_default"
    comb_results_use_bar.loc[comb_results_use_bar.method_subtype=="tuned","ta_name"] += "_tuned"

    ### PERFORMANCE ACROSS DATASETS PLOT
    base_marker = "."
    prep_marker = "*"


    tabpfn26_res = pd.read_parquet("/ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined/TABPFN26.parquet")
    tabiclv2_res = pd.read_parquet("/ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined/TABICLV2.parquet")
    tabpfn26_res.dataset = tabpfn26_res.dataset.apply(lambda x: dat_map.get(x, x))
    tabiclv2_res.dataset = tabiclv2_res.dataset.apply(lambda x: dat_map.get(x, x))

    tabpfn26_res = tabpfn26_res.loc[tabpfn26_res.fold.isin(use_folds)]
    tabiclv2_res = tabiclv2_res.loc[tabiclv2_res.fold.isin(use_folds)]

    prep_res = comb_results_use_bar.loc[comb_results_use_bar.ta_name.str.startswith("(Prep)")].copy()

    comb_results_use_norm = comb_results_use_bar.copy()
    comb_results_use_norm = comb_results_use_norm.loc[~comb_results_use_norm.ta_name.str.startswith("(Prep)")]
    comb_results_use_norm = comb_results_use_norm.loc[~comb_results_use_norm.ta_name.str.startswith("Prep")]

    dat_min = (
        comb_results_use_norm.groupby(["dataset", "fold"])["metric_error"]
        .min()
        .reset_index(name="min")
    )
    # Normalize per dataset/fold relative to the minimum and the selected upper anchor.
    comb_results_use_norm["normalized_error"] = np.nan
    comb_results_use_norm["relative_improvement_over_best_on_ta"] = np.nan
    prep_res["normalized_error"] = np.nan
    prep_res["relative_improvement_over_best_on_ta"] = np.nan
    tabpfn26_res["normalized_error"] = np.nan
    tabpfn26_res["relative_improvement_over_best_on_ta"] = np.nan
    tabiclv2_res["normalized_error"] = np.nan
    tabiclv2_res["relative_improvement_over_best_on_ta"] = np.nan
    for dataset, fold in comb_results_use_norm[["dataset", "fold"]].drop_duplicates().itertuples(index=False):
        lower = dat_min.loc[
            (dat_min.dataset == dataset) & (dat_min.fold == fold),
            "min",
        ].values[0]
        upper = _select_normalization_upper(
            comb_results_use_norm.loc[
                (comb_results_use_norm.dataset == dataset) & (comb_results_use_norm.fold == fold),
                "metric_error",
            ],
            normalization_upper,
        )
        group_mask = lambda df: (df.dataset == dataset) & (df.fold == fold)
        if lower == upper or (y_axis_metric == "relative_improvement_over_best_on_ta" and lower == 0):
            comb_results_use_norm.loc[group_mask(comb_results_use_norm), "normalized_error"] = 1.0
            comb_results_use_norm.loc[group_mask(comb_results_use_norm), "relative_improvement_over_best_on_ta"] = 0.0
            prep_res.loc[group_mask(prep_res), "normalized_error"] = 1.0
            prep_res.loc[group_mask(prep_res), "relative_improvement_over_best_on_ta"] = 0.0
            tabpfn26_res.loc[group_mask(tabpfn26_res), "normalized_error"] = 1.0
            tabpfn26_res.loc[group_mask(tabpfn26_res), "relative_improvement_over_best_on_ta"] = 0.0
            tabiclv2_res.loc[group_mask(tabiclv2_res), "normalized_error"] = 1.0
            tabiclv2_res.loc[group_mask(tabiclv2_res), "relative_improvement_over_best_on_ta"] = 0.0
        else:
            comb_results_use_norm.loc[group_mask(comb_results_use_norm), "normalized_error"] = (
                comb_results_use_norm.loc[group_mask(comb_results_use_norm), "metric_error"] - upper
            ) / (lower - upper)
            comb_results_use_norm.loc[group_mask(comb_results_use_norm), "relative_improvement_over_best_on_ta"] = (
                lower - comb_results_use_norm.loc[group_mask(comb_results_use_norm), "metric_error"]
            ) / lower
            prep_res.loc[group_mask(prep_res), "normalized_error"] = (
                prep_res.loc[group_mask(prep_res), "metric_error"] - upper
            ) / (lower - upper)
            prep_res.loc[group_mask(prep_res), "relative_improvement_over_best_on_ta"] = (
                lower - prep_res.loc[group_mask(prep_res), "metric_error"]
            ) / lower
            tabpfn26_res.loc[group_mask(tabpfn26_res), "normalized_error"] = (
                tabpfn26_res.loc[group_mask(tabpfn26_res), "metric_error"] - upper
            ) / (lower - upper)
            tabpfn26_res.loc[group_mask(tabpfn26_res), "relative_improvement_over_best_on_ta"] = (
                lower - tabpfn26_res.loc[group_mask(tabpfn26_res), "metric_error"]
            ) / lower
            tabiclv2_res.loc[group_mask(tabiclv2_res), "normalized_error"] = (
                tabiclv2_res.loc[group_mask(tabiclv2_res), "metric_error"] - upper
            ) / (lower - upper)
            tabiclv2_res.loc[group_mask(tabiclv2_res), "relative_improvement_over_best_on_ta"] = (
                lower - tabiclv2_res.loc[group_mask(tabiclv2_res), "metric_error"]
            ) / lower


    prep_res = prep_res[prep_res.method_subtype=="tuned_ensemble"].copy()
    prep_res.ta_name = prep_res.ta_name.replace(prep_model_name_map)

    if use_four_tabprep_models:
        prep_model_order = ["PrepLightGBM", "PrepLinearModel", "PrepRealTabPFN2.5", "PrepTabM"]
        use_models = ['TabPFN-v2.6', 'TabICLv2'] + prep_model_order
    else:
        prep_plot_res = prep_res.loc[prep_res.groupby(["dataset", "fold"])["metric_error"].idxmin()].copy()
        prep_plot_res.ta_name = "TabPrep"
        prep_model_order = ["TabPrep"]
        use_models = [
           'TabPFN-v2.6', 'TabICLv2',
           "TabPrep"
        ]

    if use_four_tabprep_models:
        prep_plot_res = prep_res.copy()

    comb_results_use_norm = pd.concat([comb_results_use_norm, prep_plot_res, tabpfn26_res, tabiclv2_res], ignore_index=True)
    comb_results_use_norm = comb_results_use_norm.loc[comb_results_use_norm.ta_name.isin(use_models)]

    plot_df = comb_results_use_norm.groupby(["dataset", "ta_name"], as_index=False)[
        ["normalized_error", "relative_improvement_over_best_on_ta", "metric_error"]
    ].mean()
    y_axis_label, plot_y_col, best_on_ta_line = _format_y_axis_metric(y_axis_metric)
    if y_axis_metric == "normalized_error":
        metric_suffix = "normalized_error"
    else:
        metric_suffix = "relative_improvement_over_best_on_ta"
    # dataset_order = (
    #     plot_df.groupby("dataset")["normalized_error"]
    #     .max()
    #     .sort_values(ascending=True)
    #     .index.tolist()
    # )

    prep_order_source = plot_df.loc[plot_df.ta_name.isin(prep_model_order)]
    dataset_order = (
        prep_order_source.groupby("dataset")[plot_y_col]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )

    if y_axis_metric == "normalized_error" and normalization_upper =="median":
        plot_df[plot_y_col] = plot_df[plot_y_col].clip(-0.5, 2.0)
        yticks = [2.0, 1.0, -0]
        yticklabels = [
            ">75% better",
            "Best on TA",
            "Top 75%",
        ]
    elif y_axis_metric == "normalized_error" and normalization_upper == "second-best":
        plot_df[plot_y_col] = plot_df[plot_y_col].clip(-0.9, 20.0)
        yticks = [20.0, 10.0,  2.0, 1.0, 0.0, -1]
        yticklabels = [
            ">20x better",
            "10x better",
            "2x better",
            "Best on TA",
            "2nd best",
            "Worse",
        ]
        y_limits = (-1.0, 23.0)
    elif y_axis_metric == "normalized_error" and normalization_upper == "third-best":
        plot_df[plot_y_col] = plot_df[plot_y_col].clip(-0.9, 20.0)
        yticks = [20.0, 10.0,  2.0, 1.0, 0.0, -1]
        yticklabels = [
            ">20x better",
            "10x better",
            "2x better",
            "Best on TA",
            "3rd best",
            "Worse",
        ]
        y_limits = (-1.0, 23.0)
    elif y_axis_metric == "normalized_error" and normalization_upper == "max":
        plot_df[plot_y_col] = plot_df[plot_y_col].clip(0.0, 1.0)
        yticks = [1.0, 0.5, 0.0]
        yticklabels = [
            "Best on TA",
            "Halfway",
            "Worst on TA",
        ]
    elif y_axis_metric == "relative_improvement_over_best_on_ta":
        upper_bound = 0.5 # 0.075
        lower_bound = -0.05
        plot_df[plot_y_col] = plot_df[plot_y_col].clip(lower_bound+0.01, upper_bound)
        yticks = [upper_bound, 0.05, 0.0, lower_bound]
        yticklabels = [
            ">50% better",
            "5% better",
            # "2.5% better",
            "Previous \n Best on TA",
            ">5% worse",
        ]
        y_limits = (lower_bound, upper_bound+0.1)

    plot_df_visible = plot_df.copy()
    if (
        use_four_tabprep_models
        and y_axis_metric == "relative_improvement_over_best_on_ta"
        and show_best_tabprep_model_per_dataset_only
    ):
        prep_order_source_visible = plot_df_visible.loc[plot_df_visible.ta_name.isin(prep_model_order)]
        prep_best_mask = (
            prep_order_source_visible.groupby("dataset")[plot_y_col].transform("max")
            == prep_order_source_visible[plot_y_col]
        )
        prep_keep_index = pd.MultiIndex.from_frame(
            prep_order_source_visible.loc[prep_best_mask, ["dataset", "ta_name"]].drop_duplicates()
        )
        plot_index = pd.MultiIndex.from_frame(plot_df_visible[["dataset", "ta_name"]])
        visible_mask = ~plot_df_visible.ta_name.isin(prep_model_order) | plot_index.isin(prep_keep_index)
        plot_df_visible = plot_df_visible.loc[visible_mask].reset_index(drop=True)

    tabprep_improve_dataset_count = int(
        plot_df_visible.loc[
            plot_df_visible.ta_name.isin(prep_model_order)
            & (plot_df_visible[plot_y_col] > best_on_ta_line),
            "dataset",
        ].nunique()
    )

    improvement_counts = (
        plot_df_visible.loc[plot_df_visible[plot_y_col] > best_on_ta_line, "ta_name"]
        .value_counts()
        .to_dict()
    )
    prep_model_order_sorted = prep_model_order
    if use_four_tabprep_models:
        prep_rank = {model: idx for idx, model in enumerate(prep_model_order)}
        prep_model_order_sorted = sorted(
            prep_model_order,
            key=lambda model: (-improvement_counts.get(model, 0), prep_rank[model]),
        )


    model_order = [model for model in use_models if model in plot_df.ta_name.unique()]
    ds_to_x = {dataset: i for i, dataset in enumerate(dataset_order)}
    model_to_color = {}
    upper_suffix = normalization_upper.replace("-", "_")
    if use_four_tabprep_models:
        tabprep_mode_suffix = "four_tabprep_models_best_only" if (y_axis_metric == "relative_improvement_over_best_on_ta" and show_best_tabprep_model_per_dataset_only) else "four_tabprep_models"
    else:
        tabprep_mode_suffix = "single_tabprep_model"

    fig_width = max(6.0, 0.30 * len(dataset_order))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    color_cycle = [
        "#1f77b4",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    model_to_handle = {}
    strict_lower_cutoff = y_axis_metric == "relative_improvement_over_best_on_ta"
    if normalization_upper == "second-best":
        lower_visible_y = y_limits[0]
    elif y_axis_metric == "relative_improvement_over_best_on_ta":
        lower_visible_y = y_limits[0]
    elif y_axis_metric == "normalized_error" and normalization_upper == "median":
        lower_visible_y = -0.5
    elif y_axis_metric == "normalized_error" and normalization_upper in {"third-best", "second-best"}:
        lower_visible_y = -0.9
    elif y_axis_metric == "normalized_error" and normalization_upper == "max":
        lower_visible_y = 0.0
    else:
        lower_visible_y = -np.inf

    prep_star_green_datasets: set[str] = set()
    if use_four_tabprep_models and y_axis_metric == "relative_improvement_over_best_on_ta":
        prep_star_source = plot_df.loc[
            plot_df.ta_name.isin(prep_model_order),
            ["dataset", plot_y_col],
        ]
        for dataset, group in prep_star_source.groupby("dataset"):
            top_two = group[plot_y_col].sort_values(ascending=False).to_numpy()
            if (
                green_star_improvement_range is not None
                and len(top_two) >= 2
                and green_star_improvement_range[0]
                <= (top_two[0] - top_two[1])
                <= green_star_improvement_range[1]
            ):
                prep_star_green_datasets.add(dataset)

    for idx, model_name in enumerate(model_order):
        legend_label = f"{model_name} [{improvement_counts.get(model_name, 0)}]"
        is_prep_model = model_name in prep_model_order
        linewidth = 2.8 if (model_name == "TabPrep" or is_prep_model) else 2.2
        alpha = 0.95 if (model_name == "TabPrep" or is_prep_model) else 0.85
        linestyle = "None"
        line_color = color_cycle[idx % len(color_cycle)]
        model_to_color[model_name] = line_color
        sub = plot_df_visible.loc[plot_df_visible.ta_name == model_name, ["dataset", plot_y_col]].copy()
        sub["x"] = sub["dataset"].map(ds_to_x)
        sub = sub.sort_values("x")
        sub = sub.loc[sub[plot_y_col] > lower_visible_y if strict_lower_cutoff else sub[plot_y_col] >= lower_visible_y]
        if sub.empty:
            line = ax.plot(
                [],
                [],
                marker="o",
                linestyle=linestyle,
                markersize=6.0,
                linewidth=linewidth,
                alpha=alpha,
                color=line_color,
                label=legend_label,
            )[0]
            model_to_handle[model_name] = line
            continue
        line = ax.plot(
            sub["x"],
            sub[plot_y_col],
            marker="o",
            linestyle=linestyle,
            markersize=6.0,
            linewidth=linewidth,
            alpha=alpha,
            color=line_color,
            label=legend_label,
        )[0]
        model_to_handle[model_name] = line

    tabprep_df = (
        plot_df_visible.loc[plot_df_visible.ta_name.isin(prep_model_order), ["dataset", plot_y_col]]
        .groupby("dataset", as_index=True)[plot_y_col]
        .max()
        .reindex(dataset_order)
    )
    gap_x = np.arange(len(dataset_order))
    gap_mask = tabprep_df.gt(best_on_ta_line).fillna(False).to_numpy()
    if gap_mask.any():
        gap_end_x = min(float(gap_x[gap_mask][-1]) + 0.48, len(dataset_order) - 0.45)
        tabprep_visible = (
            tabprep_df.to_numpy(dtype=float) > lower_visible_y
            if strict_lower_cutoff
            else tabprep_df.to_numpy(dtype=float) >= lower_visible_y
        )
        ax.fill_between(
            gap_x,
            np.full(len(dataset_order), best_on_ta_line, dtype=float),
            tabprep_df.to_numpy(dtype=float),
            where=gap_mask & tabprep_visible,
            color="green",
            alpha=0.16,
            interpolate=True,
            zorder=1,
        )
        ax.text(
            gap_end_x-0.6,
            best_on_ta_line+0.003,
            "Feature engineering gap",
            transform=ax.transData,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="green",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="green", alpha=0.9),
            zorder=7,
        )

    ax.plot(
        gap_x,
        tabprep_df.to_numpy(dtype=float),
        color="green",
        linewidth=2.2,
        alpha=0.95,
        zorder=4,
        label="_nolegend_",
    )

    best_mask = plot_df_visible[plot_y_col].eq(plot_df_visible.groupby("dataset")[plot_y_col].transform("max"))
    star_df = plot_df_visible.loc[best_mask & (plot_df_visible[plot_y_col] > best_on_ta_line)].copy()
    if not star_df.empty:
        star_df["x"] = star_df["dataset"].map(ds_to_x)
        star_df = star_df.loc[star_df[plot_y_col] > lower_visible_y if strict_lower_cutoff else star_df[plot_y_col] >= lower_visible_y]
        for _, row in star_df.iterrows():
            ax.scatter(
                row["x"],
                row[plot_y_col],
                marker="*",
                s=240,
                color=(
                    "green"
                    if row["dataset"] in prep_star_green_datasets and row["ta_name"] in prep_model_order
                    else model_to_color.get(row["ta_name"], "gold")
                ),
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
                label="_nolegend_",
            )

    ax.axhline(best_on_ta_line, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_xlim(-0.6, len(dataset_order) - 0.4)
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, rotation=35, ha="right", fontsize=13)
    if equidistant_y_ticks:
        forward, inverse = _equal_spaced_axis_scale(yticks)
        ax.set_yscale("function", functions=(forward, inverse))
    elif y_axis_log_scale:
        log_linthresh = 0.005 if y_axis_metric == "relative_improvement_over_best_on_ta" else 0.25
        ax.set_yscale("symlog", linthresh=log_linthresh)
    # ax.invert_yaxis()
    # ax.set_ylim(1.0, -2.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=13)
    if normalization_upper == "second-best":
        ax.set_ylim(*y_limits)
    ax.set_ylabel(y_axis_label, fontsize=15, labelpad=4)
    ax.set_xlabel("Dataset", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
    _add_grouped_legend(
        ax,
        model_order,
        model_to_handle,
        improvement_counts,
        use_four_tabprep_models,
        prep_model_order_sorted,
        tabprep_group_label=f"TabPrep [{tabprep_improve_dataset_count}]",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    plot_dir = Path(save_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_name = (
        f"data_centric_vs_model_centric_{metric_suffix}_{fold_suffix}_upper_{upper_suffix}_{tabprep_mode_suffix}.pdf"
    )
    fig.savefig(plot_dir / fig_name, dpi=300, bbox_inches="tight")

    ranked_fig_width = max(8.0, 0.30 * len(dataset_order))
    fig_ranked, ax_ranked = plt.subplots(figsize=(ranked_fig_width, 6.0))
    ranked_model_to_handle = {}

    for idx, model_name in enumerate(model_order):
        legend_label = f"{model_name} [{improvement_counts.get(model_name, 0)}]"
        linewidth = 2.8 if (model_name == "TabPrep" or model_name in prep_model_order) else 2.2
        alpha = 0.95 if (model_name == "TabPrep" or model_name in prep_model_order) else 0.85
        linestyle = "None"
        fig_color = color_cycle[idx % len(color_cycle)]
        sub = plot_df_visible.loc[plot_df_visible.ta_name == model_name, ["dataset", plot_y_col]].copy()
        sub = sub.sort_values(plot_y_col, ascending=False).reset_index(drop=True)
        sub["x"] = np.arange(len(sub))
        sub = sub.loc[sub[plot_y_col] > lower_visible_y if strict_lower_cutoff else sub[plot_y_col] >= lower_visible_y]
        if sub.empty:
            line = ax_ranked.plot(
                [],
                [],
                marker="o",
                linestyle=linestyle,
                markersize=6.5,
                linewidth=linewidth,
                alpha=alpha,
                color=fig_color,
                label=legend_label,
            )[0]
            ranked_model_to_handle[model_name] = line
            continue
        line = ax_ranked.plot(
            sub["x"],
            sub[plot_y_col],
            marker="o",
            linestyle=linestyle,
            markersize=6.5,
            linewidth=linewidth,
            alpha=alpha,
            color=fig_color,
            label=legend_label,
        )[0]
        ranked_model_to_handle[model_name] = line

    ax_ranked.axhline(best_on_ta_line, color="0.35", linestyle=":", linewidth=1.2)
    ax_ranked.set_xlim(-0.6, max(0, len(dataset_order) - 0.4))
    ax_ranked.set_xticks([])
    ax_ranked.tick_params(axis="x", bottom=False, labelbottom=False)
    if equidistant_y_ticks:
        forward, inverse = _equal_spaced_axis_scale(yticks)
        ax_ranked.set_yscale("function", functions=(forward, inverse))
    elif y_axis_log_scale:
        log_linthresh = 0.005 if y_axis_metric == "relative_improvement_over_best_on_ta" else 0.25
        ax_ranked.set_yscale("symlog", linthresh=log_linthresh)
    ax_ranked.set_yticks(yticks)
    ax_ranked.set_yticklabels(yticklabels, fontsize=13)
    if normalization_upper == "second-best":
        ax_ranked.set_ylim(*y_limits)
    ax_ranked.set_ylabel(y_axis_label, fontsize=15, labelpad=4)
    ax_ranked.set_xlabel(f"Datasets ranked within each model by {y_axis_label.lower()}", fontsize=15)
    ax_ranked.grid(axis="y", alpha=0.3)
    _add_grouped_legend(
        ax_ranked,
        model_order,
        ranked_model_to_handle,
        improvement_counts,
        use_four_tabprep_models,
        prep_model_order_sorted,
        tabprep_group_label=f"TabPrep [{tabprep_improve_dataset_count}]",
    )
    fig_ranked.tight_layout(rect=(0, 0, 1, 0.93))
    fig_ranked_name = (
        f"data_centric_vs_model_centric_{metric_suffix}_ranked_by_model_{fold_suffix}_upper_{upper_suffix}_{tabprep_mode_suffix}.pdf"
    )
    fig_ranked.savefig(
        plot_dir / fig_ranked_name,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved plot to {plot_dir / fig_name}")
    print(f"Saved plot to {plot_dir / fig_ranked_name}")
