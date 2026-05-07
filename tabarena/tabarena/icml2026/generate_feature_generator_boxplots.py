from pathlib import Path
import sys

import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[4]
_TABARENA_PKG_ROOT = _SCRIPT_PATH.parents[3]
for _path in (_REPO_ROOT, _TABARENA_PKG_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tabarena.icml2026.plotting.two_figures_boxplots import boxplot_dataframe_pubready
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


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

ABALATION_BASE_PATH = Path("/ceph/atschalz/auto_prep/tabarena_figs/icml_ablation")
SAVE_PATH = Path("/ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new")
PLOT_FOLD = 0


def normalize_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].apply(lambda x: DAT_MAP.get(x, x))
    return out


def load_tabarena_hpo_results() -> pd.DataFrame:
    ta_context = TabArenaContext()
    frames = [
        ta_context.load_hpo_results(method_name)
        for method_name in ta_context.methods
        if "AutoGluon" not in method_name
    ]
    ta_results = pd.concat(frames).reset_index(drop=True)
    return normalize_dataset_names(ta_results)


def load_ablation_results() -> pd.DataFrame:
    ablation_model_results = pd.read_csv(f"{ABALATION_BASE_PATH}/model_results.csv")
    return normalize_dataset_names(ablation_model_results)


def dataset_metric_series(df: pd.DataFrame, method: str, dataset_index: pd.Index) -> pd.Series:
    series = (
        df.loc[df.method == method, ["dataset", "metric_error"]]
        .drop_duplicates(subset=["dataset"])
        .set_index("dataset")
        .reindex(dataset_index)["metric_error"]
    )
    return series


def build_feature_generator_frames(
    ablation_model_results: pd.DataFrame,
    ta_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ablation_model_results = ablation_model_results.loc[ablation_model_results.fold == PLOT_FOLD].copy()
    ta_results = ta_results.loc[ta_results.fold == PLOT_FOLD].copy()

    setting_map = {
        "prep_LightGBM-ablation_c1_BAG_L1": "+Arithmetic",
        "prep_LightGBM-ablation_c2_BAG_L1": "RSFC",
        "prep_LightGBM-ablation_c3_BAG_L1": "Combine-TE",
        "prep_LightGBM-ablation_c4_BAG_L1": "OOF-TE",
        "prep_LightGBM-ablation_c5_BAG_L1": "GroupBy",
        "prep_LightGBM-ablation_c6_BAG_L1": "AbsoluteGroupBy",
        "prep_LightGBM-ablation_c7_BAG_L1": "OOF-TE-keepcat",
        "prep_LightGBM-ablation_c8_BAG_L1": "OOF-TE_w_GroupBy",
        "prep_LightGBM-ablation_c9_BAG_L1": "Arithmetic (2-order)",
        "prep_LightGBM-ablation_c10_BAG_L1": "Arithmetic (prod,ratio)",
        "prep_LightGBM-ablation_c11_BAG_L1": "Arithmetic (sum,diff)",
        "prep_LightGBM-ablation_c12_BAG_L1": "Cat-Pipeline",
        "prep_LightGBM-ablation_c13_BAG_L1": "Cat-Pipeline (keepcat)",
        "prep_LightGBM-ablation_c14_BAG_L1": "+OOF-TE",
        "prep_LightGBM-ablation_c15_BAG_L1": "+Combine-TE",
        "prep_LightGBM-ablation_c16_BAG_L1": "+GroupBy",
        "prep_LightGBM-ablation_c17_BAG_L1": "-Arithmetic",
        "prep_LightGBM-ablation_c18_BAG_L1": "-OOF-TE",
        "prep_LightGBM-ablation_c19_BAG_L1": "-Combine-TE",
        "prep_LightGBM-ablation_c20_BAG_L1": "-GroupBy",
        "prep_LightGBM-ablation_c21_BAG_L1": "1000 features",
        "prep_LightGBM-ablation_c22_BAG_L1": "500 features",
        "prep_LightGBM-ablation_c23_BAG_L1": "100 features",
    }

    prep_lgb_df = ta_results.loc[
        ta_results.method == "PREP_GBM (default)",
        ["dataset", "metric_error"],
    ].copy()
    prep_lgb_df["method"] = "+RSFC"
    prep_lgb_df = prep_lgb_df[["dataset", "method", "metric_error"]]

    prep_ablation_df = ablation_model_results.loc[
        ablation_model_results.method.apply(lambda x: x.startswith("prep_LightGBM-ablation")),
        ["dataset", "method", "metric_error"],
    ].copy()
    prep_ablation_df["method"] = prep_ablation_df["method"].map(setting_map)
    prep_ablation_df = pd.concat([prep_ablation_df, prep_lgb_df], axis=0).reset_index(drop=True)

    df_components = pd.DataFrame(
        prep_ablation_df.pivot(
            index="dataset",
            columns="method",
            values="metric_error",
        )
    )
    df_components["Default LightGBM"] = dataset_metric_series(ta_results, "GBM (default)", df_components.index)
    df_components["+HPO"] = dataset_metric_series(ta_results, "PREP_GBM (tuned + ensemble)", df_components.index)
    df_components["dataset"] = df_components.index

    df_components_best_by_stage = df_components.copy()
    df_components_best_by_stage["+Arithmetic"] = df_components[["Default LightGBM", "+Arithmetic"]].min(axis=1)
    df_components_best_by_stage["+OOF-TE"] = df_components[["Default LightGBM", "+Arithmetic", "+OOF-TE"]].min(axis=1)
    df_components_best_by_stage["+Combine-TE"] = df_components[["Default LightGBM", "+Arithmetic", "+OOF-TE", "+Combine-TE"]].min(axis=1)
    df_components_best_by_stage["+GroupBy"] = df_components[["Default LightGBM", "+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy"]].min(axis=1)
    df_components_best_by_stage["+RSFC"] = df_components[["Default LightGBM", "+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC"]].min(axis=1)

    return df_components, df_components_best_by_stage, prep_ablation_df


def plot_feature_generator_boxplots(
    df_components: pd.DataFrame,
    df_components_best_by_stage: pd.DataFrame,
    prep_ablation_df: pd.DataFrame,
) -> None:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    boxplot_dataframe_pubready(
        df_components,
        baseline_col="Default LightGBM",
        competitor_cols=["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC", "+HPO"][::-1],
        dpi=300,
        transparent=True,
        font_size=14.0,
        title_size=14.0,
        point_size=14.0,
        cap=[-0.1, 0.25],
        figsize=(8, 4),
        save_path=f"{SAVE_PATH}/ablation_contribution_boxplot_withtuned_v1.pdf",
    )

    boxplot_dataframe_pubready(
        df_components.rename({"+RSFC": "TabPrep", "+GroupBy": "-RSFC"}, axis=1).apply(
            lambda x: x.fillna(df_components.loc[x.index, "+Arithmetic"]),
            axis=0,
        ),
        baseline_col="Default LightGBM",
        competitor_cols=["TabPrep", "-Combine-TE", "-GroupBy", "-OOF-TE", "-RSFC", "-Arithmetic"][::-1],
        dpi=300,
        transparent=True,
        font_size=14.0,
        title_size=14.0,
        point_size=14.0,
        cap=[-0.1, 0.25],
        figsize=(8, 4),
        save_path=f"{SAVE_PATH}/ablation_OOF_contribution_boxplot.pdf",
    )

    boxplot_dataframe_pubready(
        df_components.rename({"+RSFC": "2000 features"}, axis=1).apply(
            lambda x: x.fillna(df_components.loc[x.index, "+Arithmetic"]),
            axis=0,
        ),
        baseline_col="Default LightGBM",
        competitor_cols=["OOF-TE", "100 features", "500 features", "1000 features", "2000 features"][::-1],
        dpi=300,
        transparent=True,
        font_size=14.0,
        title_size=14.0,
        point_size=14.0,
        cap=[-0.1, 0.25],
        figsize=(8, 4),
        save_path=f"{SAVE_PATH}/relative_improvement_feature_size_boxplot.pdf",
    )


def main() -> None:
    ta_results = load_tabarena_hpo_results()
    ablation_model_results = load_ablation_results()
    df_components, df_components_best_by_stage, prep_ablation_df = build_feature_generator_frames(
        ablation_model_results=ablation_model_results,
        ta_results=ta_results,
    )
    plot_feature_generator_boxplots(
        df_components=df_components,
        df_components_best_by_stage=df_components_best_by_stage,
        prep_ablation_df=prep_ablation_df,
    )


if __name__ == "__main__":
    main()
