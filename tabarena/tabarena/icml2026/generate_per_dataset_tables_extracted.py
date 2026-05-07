from pathlib import Path
import importlib.util
import sys

import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[4]
_TABARENA_NAMESPACE_ROOT = _SCRIPT_PATH.parents[3]
_TABARENA_PKG_ROOT = _SCRIPT_PATH.parents[2]
for _path in (_REPO_ROOT, _TABARENA_NAMESPACE_ROOT, _TABARENA_PKG_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_PER_DATASET_TABLES_PATH = _SCRIPT_PATH.parents[1] / "nips2025_utils" / "per_dataset_tables.py"
_PER_DATASET_TABLES_SPEC = importlib.util.spec_from_file_location(
    "per_dataset_tables_local", _PER_DATASET_TABLES_PATH
)
if _PER_DATASET_TABLES_SPEC is None or _PER_DATASET_TABLES_SPEC.loader is None:
    raise ImportError(f"Could not load helper module from {_PER_DATASET_TABLES_PATH}")
_PER_DATASET_TABLES_MODULE = importlib.util.module_from_spec(_PER_DATASET_TABLES_SPEC)
_PER_DATASET_TABLES_SPEC.loader.exec_module(_PER_DATASET_TABLES_MODULE)
get_per_dataset_tables = _PER_DATASET_TABLES_MODULE.get_per_dataset_tables


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
MODEL_NAMES = ["prep_TabM", "prep_RealTabPFN"]
COMBINED_FILES = ["LightGBM", "Linear", "RealTabPFN-2.5", "TabM"]


def normalize_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].apply(lambda x: DAT_MAP.get(x, x))
    return out


def load_tabarena_results() -> pd.DataFrame:
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext

    ta_context = TabArenaContext()
    ta_results = pd.concat(
        [ta_context.load_hpo_results(method) for method in ta_context.methods if "AutoGluon" not in method]
    ).reset_index(drop=True)
    ta_results = normalize_dataset_names(ta_results)
    return ta_results.loc[ta_results.method != "MITRA_GPU (default)"].reset_index(drop=True)


def load_combined_results(ta_results: pd.DataFrame) -> pd.DataFrame:
    all_hpo_results = pd.DataFrame()
    for model_name in MODEL_NAMES:
        hpo_results = pd.read_csv(BASE_PATH / model_name / "hpo_results.csv")
        hpo_results["model_name"] = model_name
        hpo_results = normalize_dataset_names(hpo_results)
        all_hpo_results = pd.concat([all_hpo_results, hpo_results]).reset_index(drop=True)

    comb_results = pd.concat([all_hpo_results, ta_results]).reset_index(drop=True)

    comb_results.ta_name = comb_results.ta_name.map(
        {
            "prep_TabM": "PrepTabM",
            "RealTabPFN-v2.5": "RealTabPFN2.5",
            "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5",
            "TabM_GPU": "TabM",
        }
    ).fillna(comb_results.ta_name)

    comb_results_use = comb_results.loc[comb_results["method_subtype"] == "tuned_ensemble"]
    comb_results_use = comb_results_use.loc[comb_results_use.fold == 0]
    comb_results_use = normalize_dataset_names(comb_results_use)

    comb_results_use_bar = comb_results_use.copy()
    for model_name in COMBINED_FILES:
        comb_results_use_bar = pd.concat(
            [comb_results_use_bar, pd.read_parquet(COMB_PATH / f"{model_name}.parquet")]
        ).reset_index(drop=True)

    lightgbm_rows = comb_results_use_bar.loc[comb_results_use_bar.ta_name == "(Prep)LightGBM"]
    return pd.concat([ta_results, lightgbm_rows]).reset_index(drop=True)


def main() -> None:
    df_results = load_combined_results(load_tabarena_results())
    get_per_dataset_tables(df_results=df_results, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()
