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
    tabpfn26_res = new_res[new_res.method=='TABPFN-V2.6 (default)']


    comb_results = pd.concat([
        all_hpo_results,#[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]], 
        ta_results,#[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]]
        pd.read_parquet(f"{parquet_path}/RealTabPFN-2.5.parquet"),
        pd.read_parquet(f"{parquet_path}/TabM.parquet"),
        pd.read_parquet(f"{parquet_path}/Linear.parquet"),
        pd.read_parquet(f"{parquet_path}/LightGBM.parquet"),
        tabiclnew_res,
        tabpfn26_res
    ]).reset_index(drop=True)

    