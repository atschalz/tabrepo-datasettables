from pathlib import Path
import importlib.util
import sys

import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[4]
_TABARENA_PKG_ROOT = _SCRIPT_PATH.parents[3]
for _path in (_REPO_ROOT, _TABARENA_PKG_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_PLOT_MODULE_PATH = _SCRIPT_PATH.parent / "plotting" / "per_dataset_results.py"
_PLOT_SPEC = importlib.util.spec_from_file_location("per_dataset_results_local", _PLOT_MODULE_PATH)
if _PLOT_SPEC is None or _PLOT_SPEC.loader is None:
    raise ImportError(f"Could not load plotting helper from {_PLOT_MODULE_PATH}")
_PLOT_MODULE = importlib.util.module_from_spec(_PLOT_SPEC)
_PLOT_SPEC.loader.exec_module(_PLOT_MODULE)
plot_model_performance_across_datasets = _PLOT_MODULE.plot_model_performance_across_datasets


DAT_MAP = {
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

BASE_PATH = Path("/ceph/atschalz/auto_prep/tabarena_figs/icml_final")
COMB_PATH = Path("/ceph/atschalz/auto_prep/tabarena/examples/icml2026/results/hpo_combined")
SAVE_PATH = Path("/ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new")
HPO_RESULTS_PATH = BASE_PATH / "hpo_results.csv"


def normalize_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].apply(lambda x: DAT_MAP.get(x, x))
    return out


def normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ta_name" not in out.columns:
        return out

    out["ta_name"] = out["ta_name"].replace({
        "prep_LightGBM": "PrepLightGBM",
        "prep_LinearModel": "PrepLinearModel",
        "prep_RealMLP": "PrepRealMLP",
    })
    out["ta_name"] = out["ta_name"].str.replace("_default", "", regex=False)
    out["ta_name"] = out["ta_name"].str.replace("_tuned", "", regex=False)
    return out


def load_tabarena_results() -> pd.DataFrame:
    ta_results = pd.read_csv(HPO_RESULTS_PATH)
    ta_results = normalize_dataset_names(ta_results)
    ta_results = ta_results.loc[ta_results.method != "MITRA_GPU (default)"].copy()
    return ta_results


def load_combined_results(ta_results: pd.DataFrame) -> pd.DataFrame:
    all_hpo_results = []
    for model_name in ["prep_TabM", "prep_RealTabPFN"]:
        hpo_results = pd.read_csv(f"{BASE_PATH}/{model_name}/hpo_results.csv")
        hpo_results["model_name"] = model_name
        hpo_results = normalize_dataset_names(hpo_results)
        all_hpo_results.append(hpo_results)

    all_hpo_results = pd.concat(all_hpo_results, ignore_index=True)

    comb_results = pd.concat(
        [
            all_hpo_results[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]],
            ta_results[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "time_train_s", "time_infer_s", "method_subtype"]],
        ]
    ).reset_index(drop=True)

    comb_results.ta_name = comb_results.ta_name.map({
        "prep_TabM": "PrepTabM",
        "RealTabPFN-v2.5": "RealTabPFN2.5",
        "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5",
        "TabM_GPU": "TabM",
    }).fillna(comb_results.ta_name)

    comb_results_use = comb_results.loc[comb_results["method_subtype"] == "tuned_ensemble"].copy()
    comb_results_use = comb_results_use.loc[comb_results_use.fold == 0].copy()
    comb_results_use.dataset = comb_results_use.dataset.apply(lambda x: DAT_MAP.get(x, x))

    comb_results_use_bar = comb_results_use.copy()
    for model_name in ["LightGBM", "Linear", "RealTabPFN-2.5", "TabM"]:
        comb_results_use_bar = pd.concat([comb_results_use_bar, pd.read_parquet(f"{COMB_PATH}/{model_name}.parquet")]).reset_index(drop=True)

    comb_results_use_bar = comb_results_use_bar.loc[comb_results_use_bar.fold == 0].copy()
    comb_results_use_bar = comb_results_use_bar[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "method_subtype"]]
    comb_results_use_bar.dataset = comb_results_use_bar.dataset.apply(lambda x: DAT_MAP.get(x, x))

    comb_results_use.loc[comb_results_use.method_subtype == "default", "ta_name"] += "_default"
    comb_results_use.loc[comb_results_use.method_subtype == "tuned", "ta_name"] += "_tuned"
    comb_results_use_bar.loc[comb_results_use_bar.method_subtype == "default", "ta_name"] += "_default"
    comb_results_use_bar.loc[comb_results_use_bar.method_subtype == "tuned", "ta_name"] += "_tuned"

    comb_results_use = normalize_model_names(comb_results_use)
    comb_results_use_bar = normalize_model_names(comb_results_use_bar)

    return comb_results_use, comb_results_use_bar


def plot_appendix_figures(comb_results_use: pd.DataFrame, comb_results_use_bar: pd.DataFrame) -> None:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    prep_only_models = [
        m
        for m in ["PrepLightGBM", "PrepLinearModel", "PrepRealMLP", "PrepRealTabPFN2.5", "PrepTabM"]
        if m in set(comb_results_use.ta_name.unique())
    ]
    combined_models = [
        m
        for m in [
            "PrepLightGBM",
            "PrepLinearModel",
            "PrepRealMLP",
            "PrepRealTabPFN2.5",
            "PrepTabM",
            "(Prep)LightGBM",
            "(Prep)Linear",
            "(Prep)RealTabPFN-2.5",
            "(Prep)TabM",
        ]
        if m in set(comb_results_use_bar.ta_name.unique())
    ]

    plot_model_performance_across_datasets(
        comb_results_use_bar,
        model_col="ta_name",
        mode="median_centered_signed",
        value_label="Quantile-anchored normalized score",
        display_models=combined_models,
        title=None,
        sort_direction="worst_to_best",
        clip_good_side=True,
        bad_side_cap=1,
        good_side_cap=-2,
        show_model_averages=False,
        default_markers=("o", "s", "^", "D", "v", "P", "X", ">", "<", "*"),
        figsize=(16, 6),
        y_tick_labels={
            -2: "Improves over \n Best as much \n as Best \nover Top 75%",
            -1.0: "Best on\nTabArena",
            0.0: "Top\n75% on\nTabArena",
            1.0: "Top 50%\nor worse on\nTabArena",
        },
        font_size=13,
        title_font_size=13,
        legend_font_size=14,
        tick_font_size=12,
        save_path=f"{SAVE_PATH}/final_model_performance_across_datasets_combine.pdf",
    )

    plot_model_performance_across_datasets(
        comb_results_use,
        model_col="ta_name",
        mode="median_centered_signed",
        value_label="Quantile-anchored normalized score",
        display_models=prep_only_models,
        title=None,
        sort_direction="worst_to_best",
        clip_good_side=True,
        bad_side_cap=1,
        good_side_cap=-2,
        show_model_averages=False,
        default_markers=("o", "s", "^", "D", "v", "P", "X", ">", "<", "*"),
        figsize=(16, 6),
        y_tick_labels={
            -2: "Improves over \n Best as much \n as Best \nover Top 75%",
            -1.0: "Best on\nTabArena",
            0.0: "Top\n75% on\nTabArena",
            1.0: "Top 50%\nor worse on\nTabArena",
        },
        font_size=13,
        title_font_size=13,
        legend_font_size=14,
        tick_font_size=12,
        save_path=f"{SAVE_PATH}/final_model_performance_across_datasets.pdf",
    )


def main() -> None:
    ta_results = load_tabarena_results()
    comb_results_use, comb_results_use_bar = load_combined_results(ta_results)
    plot_appendix_figures(comb_results_use, comb_results_use_bar)


if __name__ == "__main__":
    main()
