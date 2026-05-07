from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
datasets_metadata = load_task_metadata()



ablation_base_path = "//ceph/atschalz/auto_prep/tabarena_figs/icml_ablation"
base_path = "//ceph/atschalz/auto_prep/tabarena_figs/icml_final/"
comb_path = "//ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined/"
save_path = "//ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new"

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

if __name__ == "__main__":
    ta_context = TabArenaContext()
    # ta_context.load_configs_hyperparameters(methods = ["PrepLightGBM", "PrepLinearModel"], download=True)
    # ta_context.load_results_paper(methods=["PrepLightGBM", "PrepLinearModel"])
    ta_results = pd.concat([ta_context.load_hpo_results(i) for i in ta_context.methods if "AutoGluon" not in i]).reset_index(drop=True)
    ta_results.dataset = ta_results.dataset.apply(lambda x: dat_map.get(x, x))


    results = ta_context.load_config_results("PrepLightGBM")
    hpo_results = ta_context.load_hpo_results("PrepLightGBM")
    results.dataset = results.dataset.apply(lambda x: dat_map.get(x, x))
    hpo_results.dataset = hpo_results.dataset.apply(lambda x: dat_map.get(x, x))

    # metadata = load_task_metadata()
    # task_map = dict(metadata[["name","tid"]].values)
    # results["task"] = results["dataset"].map(task_map)
    # hpo_results["task"] = hpo_results["dataset"].map(task_map)

    all_model_results = pd.DataFrame()
    all_hpo_results = pd.DataFrame()
    models = ["prep_TabM", "prep_RealTabPFN"] #, "prep_RealMLP"]
    for model_name in models:
        model_results = pd.read_csv(f"{base_path}/{model_name}/model_results.csv")
        model_results["model_name"] = model_name
        model_results.dataset = model_results.dataset.apply(lambda x: dat_map.get(x, x))
        all_model_results = pd.concat([all_model_results, model_results]).reset_index(drop=True)

        hpo_results = pd.read_csv(f"{base_path}/{model_name}/hpo_results.csv")
        hpo_results["model_name"] = model_name
        hpo_results.dataset = hpo_results.dataset.apply(lambda x: dat_map.get(x, x))
        all_hpo_results = pd.concat([all_hpo_results, hpo_results]).reset_index(drop=True)

    all_model_results.ta_name = all_model_results.ta_name.map({"prep_TabM": "PrepTabM", 
                                                    "RealTabPFN-v2.5": "RealTabPFN2.5", 
                                                    "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
                                                    "TabM_GPU": "TabM"}).fillna(all_model_results.ta_name)

    comb_results = pd.concat([
    all_hpo_results[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]], 
    ta_results[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]]
    ]).reset_index(drop=True)

    comb_results.ta_name = comb_results.ta_name.map({
        "prep_TabM": "PrepTabM", 
        "RealTabPFN-v2.5": "RealTabPFN2.5", 
        "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
        "TabM_GPU": "TabM"}).fillna(comb_results.ta_name)
    
    comb_results_use = comb_results.loc[comb_results['method_subtype']=="tuned_ensemble"]
    comb_results_use = comb_results_use.loc[comb_results_use.fold==0]

    comb_results_use.dataset = comb_results_use.dataset.apply(lambda x: dat_map.get(x, x))

    list(comb_results_use.dataset.unique())

    comb_results_use_bar = comb_results_use.copy()
    for m in ["LightGBM", "Linear", "RealTabPFN-2.5", "TabM"]:
        comb_results_use_bar = pd.concat([comb_results_use_bar, pd.read_parquet(f"{comb_path}/{m}.parquet")]).reset_index(drop=True)

    # REBUTTAL
    hpo_results_subset = comb_results.loc[comb_results.fold==0,["ta_name", "method_subtype", "dataset", "metric_error", "time_train_s", "time_infer_s"]]
    
    comb_results_meta = pd.merge(comb_results,datasets_metadata[["dataset", "n_features"]],on="dataset")

    #\REBUTTAL    


    ta_results = ta_results.loc[ta_results.method!='MITRA_GPU (default)']

    
    comb_results_use_bar = comb_results_use_bar.loc[comb_results_use_bar.fold==0]
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

    tabpfn26_res = tabpfn26_res.loc[tabpfn26_res.fold==0]
    tabiclv2_res = tabiclv2_res.loc[tabiclv2_res.fold==0]

    prep_res = comb_results_use_bar.loc[comb_results_use_bar.ta_name.str.startswith("(Prep)")].copy()

    comb_results_use_norm = comb_results_use_bar.copy()
    comb_results_use_norm = comb_results_use_norm.loc[~comb_results_use_norm.ta_name.str.startswith("(Prep)")]
    comb_results_use_norm = comb_results_use_norm.loc[~comb_results_use_norm.ta_name.str.startswith("Prep")]

    dat_median = comb_results_use_norm.groupby("dataset")["metric_error"].median().reset_index(name="median")
    dat_min = comb_results_use_norm.groupby("dataset")["metric_error"].min().reset_index(name="min")
    # Normalite to median = -1 and min = 1 per dataset
    comb_results_use_norm["normalized_error"] = np.nan
    prep_res["normalized_error"] = np.nan
    tabpfn26_res["normalized_error"] = np.nan
    tabiclv2_res["normalized_error"] = np.nan
    for dataset in comb_results_use_norm.dataset.unique():
        median = dat_median.loc[dat_median.dataset==dataset, "median"].values[0]
        min_ = dat_min.loc[dat_min.dataset==dataset, "min"].values[0]
        if median==min_:
            comb_results_use_norm.loc[comb_results_use_norm.dataset==dataset, "normalized_error"] = 0.0
        else:
            comb_results_use_norm.loc[comb_results_use_norm.dataset==dataset, "normalized_error"] = (comb_results_use_norm.loc[comb_results_use_norm.dataset==dataset, "metric_error"] - median) / (min_ - median)

        prep_res.loc[prep_res.dataset==dataset, "normalized_error"] = (prep_res.loc[prep_res.dataset==dataset, "metric_error"] - median) / (min_ - median)
        tabpfn26_res.loc[tabpfn26_res.dataset==dataset, "normalized_error"] = (tabpfn26_res.loc[tabpfn26_res.dataset==dataset, "metric_error"] - median) / (min_ - median)
        tabiclv2_res.loc[tabiclv2_res.dataset==dataset, "normalized_error"] = (tabiclv2_res.loc[tabiclv2_res.dataset==dataset, "metric_error"] - median) / (min_ - median)


    prep_res = prep_res[prep_res.method_subtype=="tuned_ensemble"]
    comb_results_use_norm = pd.concat([comb_results_use_norm, prep_res, tabpfn26_res, tabiclv2_res], ignore_index=True)

    use_models = [
    #     '(Prep)LightGBM_default', '(Prep)LightGBM_tuned', 
    '(Prep)LightGBM',
    #    '(Prep)Linear_default', '(Prep)Linear_tuned', 
    '(Prep)Linear',
    #    '(Prep)RealTabPFN-2.5_default', '(Prep)RealTabPFN-2.5_tuned',
       '(Prep)RealTabPFN-2.5', 
    # '(Prep)TabM_default', '(Prep)TabM_tuned',
       '(Prep)TabM', 
       'TabPFN-v2.6', 'TabICLv2',
    #    "TabPrep"
       ]

    # Select best Prep model per dataset
    best_prep = prep_res.loc[prep_res.groupby("dataset")["metric_error"].idxmin()]
    best_prep.ta_name = "TabPrep"
    comb_results_use_norm = pd.concat([comb_results_use_norm, best_prep], ignore_index=True)

    comb_results_use_norm = comb_results_use_norm.loc[comb_results_use_norm.ta_name.isin(use_models)]

    plot_df = (
        comb_results_use_norm.groupby(["dataset", "ta_name"], as_index=False)["normalized_error"]
        .mean()
    )
    dataset_order = (
        plot_df.groupby("dataset")["normalized_error"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # dataset_order = plot_df.loc[plot_df.ta_name=="TabPrep"].sort_values("normalized_error")["dataset"].values.tolist()

    plot_df["normalized_error"] = plot_df["normalized_error"].clip(-0.5, 2.0)
    yticks = [2.0, 1.0, -0]
    # yticks = [3.0, 2.0, 1.0, 0.0, -1.0]
    yticklabels = [
            # "Improves over \n Best as much \n as Best \nover Top 50%",
            "Improves over \n Best as much \n as Best \nover Top 75%",
            "Best on\nTabArena",
            "Top\n75% on\nTabArena",
            # "Top 50%\nor worse on\nTabArena",
        ]


    model_order = [model for model in use_models if model in plot_df.ta_name.unique()]
    ds_to_x = {dataset: i for i, dataset in enumerate(dataset_order)}
    model_to_color = {}

    fig_width = max(6.0, 0.30 * len(dataset_order))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, model_name in enumerate(model_order):
        sub = plot_df.loc[plot_df.ta_name == model_name, ["dataset", "normalized_error"]].copy()
        sub["x"] = sub["dataset"].map(ds_to_x)
        sub = sub.sort_values("x")
        if sub.empty:
            continue
        linewidth = 2.8 if model_name == "TabPrep" else 2.2
        alpha = 0.95 if model_name == "TabPrep" else 0.85
        line_color = color_cycle[idx % len(color_cycle)]
        model_to_color[model_name] = line_color
        ax.scatter(
            sub["x"],
            sub["normalized_error"],
            s=25,
            marker="o",
            alpha=alpha,
            color=line_color,
            label=model_name,
        )

    best_mask = plot_df["normalized_error"].eq(plot_df.groupby("dataset")["normalized_error"].transform("max"))
    star_df = plot_df.loc[best_mask & (plot_df["normalized_error"] > 1.0)].copy()
    star_counts = star_df.groupby("ta_name").size().to_dict()
    if not star_df.empty:
        star_df["x"] = star_df["dataset"].map(ds_to_x)
        for _, row in star_df.iterrows():
            ax.scatter(
                row["x"],
                row["normalized_error"],
                marker="*",
                s=180,
                color=model_to_color.get(row["ta_name"], "gold"),
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
                label="_nolegend_",
            )

    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, rotation=35, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(dataset_order) - 0.4)
    # ax.invert_yaxis()
    # ax.set_ylim(1.0, -2.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,
        fontsize=8,
    )
    ax.set_ylabel("normalized score")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.3)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_labels = [
        f"{label} ({star_counts.get(label, 0)})" if label in model_order else label
        for label in legend_labels
    ]
    ax.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(model_order), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    plot_dir = Path(save_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "data_centric_vs_model_centric_normalized_error_allmodels.pdf", dpi=300, bbox_inches="tight")

    ranked_fig_width = max(8.0, 0.30 * len(dataset_order))
    fig_ranked, ax_ranked = plt.subplots(figsize=(ranked_fig_width, 6.0))

    for idx, model_name in enumerate(model_order):
        sub = plot_df.loc[plot_df.ta_name == model_name, ["dataset", "normalized_error"]].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("normalized_error", ascending=False).reset_index(drop=True)
        sub["x"] = np.arange(len(sub))
        linewidth = 2.8 if model_name == "TabPrep" else 2.2
        alpha = 0.95 if model_name == "TabPrep" else 0.85
        fig_color = color_cycle[idx % len(color_cycle)]
        ax_ranked.plot(
            sub["x"],
            sub["normalized_error"],
            marker="o",
            markersize=4.5,
            linewidth=linewidth,
            alpha=alpha,
            color=fig_color,
            label=model_name,
        )

    ax_ranked.axhline(0.0, color="0.35", linestyle=":", linewidth=1.2)
    ax_ranked.set_xlim(-0.6, max(0, len(dataset_order) - 0.4))
    ax_ranked.set_xticks([])
    ax_ranked.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_ranked.set_yticks(yticks)
    ax_ranked.set_yticklabels(yticklabels, fontsize=8)
    ax_ranked.set_ylabel("Quantile-anchored normalized score")
    ax_ranked.set_xlabel("Datasets ranked within each model by normalized error")
    ax_ranked.set_title("Data-centric vs model-centric normalized error (ranked per model)")
    ax_ranked.grid(axis="y", alpha=0.3)
    ax_ranked.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(model_order), frameon=False)
    fig_ranked.tight_layout(rect=(0, 0, 1, 0.93))
    fig_ranked.savefig(
        plot_dir / "data_centric_vs_model_centric_normalized_error_ranked_by_model_allmodels.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved plot to {plot_dir / 'data_centric_vs_model_centric_normalized_error_allmodels.pdf'}")
    print(
        "Saved plot to "
        f"{plot_dir / 'data_centric_vs_model_centric_normalized_error_ranked_by_model_allmodels.pdf'}"
    )
