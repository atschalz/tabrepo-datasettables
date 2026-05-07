from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from tabarena.icml2026.plotting.two_figures_boxplots import boxplot_two_dataframes_pubready
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


ABLATION_BASE_PATH = Path("//ceph/atschalz/auto_prep/tabarena_figs/icml_ablation")
TASK_METADATA_PATH = Path("//ceph/atschalz/auto_prep/ta_repos/TabArena/refactored/task_metadata.parquet")
SAVE_PATH = Path("//ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/figures/new")
PLOT_FOLDS = [0,1,2]

PREP_LR_METHOD_ALIASES = ["PrepLinearModel", "PREP_LR (default)", "prep_LR (default)"]
PREP_LGB_METHOD_ALIASES = ["PrepLightGBM", "PREP_GBM (default)", "prep_GBM (default)"]


def normalize_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].apply(lambda x: DAT_MAP.get(x, x))
    return out


def normalize_fold_selection(folds: list[int]) -> tuple[int, ...]:
    return tuple(sorted(dict.fromkeys(folds)))


def fold_suffix(folds: list[int]) -> str:
    normalized_folds = normalize_fold_selection(folds)
    if len(normalized_folds) == 1:
        return f"fold{normalized_folds[0]}"
    return "folds_" + "".join(str(fold) for fold in normalized_folds)


def should_average_folds(folds: list[int]) -> bool:
    return len(normalize_fold_selection(folds)) > 1


def resolve_method_name(df: pd.DataFrame, method_aliases: list[str]) -> str:
    available = set(df["method"].dropna().unique())
    for method_name in method_aliases:
        if method_name in available:
            return method_name
    return method_aliases[0]


def load_tabarena_hpo_results() -> pd.DataFrame:
    ta_context = TabArenaContext()
    frames = []
    for method_name in ta_context.methods:
        if "AutoGluon" in method_name:
            continue
        frames.append(ta_context.load_hpo_results(method_name))

    ta_results = pd.concat(frames).reset_index(drop=True)
    return normalize_dataset_names(ta_results)


def load_ablation_results() -> pd.DataFrame:
    ablation_model_results = pd.read_csv(f"{ABLATION_BASE_PATH}/model_results.csv")
    ablation_model_results = normalize_dataset_names(ablation_model_results)
    return ablation_model_results.loc[ablation_model_results.fold.isin(PLOT_FOLDS)].copy()


def load_ablation_results_for_plotting() -> pd.DataFrame:
    ablation_model_results = pd.read_csv(f"{ABLATION_BASE_PATH}/model_results.csv")
    ablation_model_results = normalize_dataset_names(ablation_model_results)
    return ablation_model_results.loc[ablation_model_results.fold.isin(PLOT_FOLDS)].copy()


def average_metric_frame_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average numeric metric columns per dataset when multiple folds are selected.
    """
    if not should_average_folds(PLOT_FOLDS):
        return df

    out = df.drop(columns=["fold"], errors="ignore").copy()
    return out.groupby("dataset", as_index=False).mean(numeric_only=True)


def average_relative_improvements_by_dataset(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    competitor_cols: list[str],
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Convert fold-level metric values into per-dataset mean relative improvements.

    The returned dataframe is synthetic but preserves the improvement values when
    consumed by the relative-improvement boxplot helper.
    """
    if not should_average_folds(PLOT_FOLDS):
        return df

    required_cols = ["dataset", "fold", baseline_col, *competitor_cols]
    working = df.loc[:, [c for c in required_cols if c in df.columns]].copy()
    base = working[baseline_col].to_numpy(dtype=float)
    score_data = {"dataset": working["dataset"].to_numpy()}

    denom = np.maximum(np.abs(base), eps)
    for comp_col in competitor_cols:
        comp = working[comp_col].to_numpy(dtype=float)
        scores = np.where(
            np.isnan(base) | np.isnan(comp),
            np.nan,
            (base - comp) / denom,
        )
        score_data[comp_col] = scores

    score_df = pd.DataFrame(score_data)
    mean_scores = score_df.groupby("dataset", as_index=False)[competitor_cols].mean()
    out = mean_scores.copy()
    out[baseline_col] = 1.0
    for comp_col in competitor_cols:
        out[comp_col] = 1.0 - out[comp_col]
    return out[["dataset", baseline_col, *competitor_cols]]


def load_task_metadata() -> pd.DataFrame:
    task_metadata = pd.read_parquet(TASK_METADATA_PATH)
    return normalize_dataset_names(task_metadata)


def count_successful_datasets(df: pd.DataFrame, method_names: list[str]) -> int:
    """
    Count datasets where a method produced a test result.

    A run is treated as successful when `metric_error` is present.
    """
    method_df = df.loc[df.method.isin(method_names)]
    return method_df.loc[method_df["metric_error"].notna(), "dataset"].nunique()


def print_successful_dataset_counts(
    ablation_model_results: pd.DataFrame,
    ta_results: pd.DataFrame,
) -> None:
    prep_lr_method = resolve_method_name(ta_results, PREP_LR_METHOD_ALIASES)
    prep_lgb_method = resolve_method_name(ta_results, PREP_LGB_METHOD_ALIASES)
    print("Successful datasets by method:")
    print(
        f"  AutoFeat: {count_successful_datasets(ablation_model_results, ['AutoFeatLinearModel_c1_BAG_L1'])}"
    )
    print(
        f"  OpenFE: {count_successful_datasets(ablation_model_results, ['OpenFELGBModel_c1_BAG_L1'])}"
    )
    print(
        f"  PrepLR: {count_successful_datasets(ta_results, [prep_lr_method])}"
    )
    print(
        f"  PrepLGB: {count_successful_datasets(ta_results, [prep_lgb_method])}"
    )


def print_missing_result_diagnostics(
    df: pd.DataFrame,
    *,
    method_label: str,
    method_names: list[str],
    task_metadata: pd.DataFrame,
) -> None:
    method_df = df.loc[df.method.isin(method_names)].copy()
    if "fold" in method_df.columns:
        method_df = method_df.loc[method_df.fold.isin(PLOT_FOLDS)].copy()

    if method_df.empty:
        print(f"\n{method_label}: no rows found.")
        return

    successful_rows = (
        method_df.loc[method_df["metric_error"].notna(), ["dataset", "metric_error"]]
        .drop_duplicates(subset=["dataset"])
        .merge(
            task_metadata.loc[:, ["dataset", "n_samples_train_per_fold", "n_features"]],
            on="dataset",
            how="left",
        )
    )
    present_datasets = set(method_df["dataset"].dropna().unique())
    expected_datasets = set(task_metadata["dataset"].dropna().unique())
    missing_datasets = sorted(expected_datasets - present_datasets)
    failed_rows = method_df.loc[method_df["metric_error"].isna()].copy()
    failed_datasets = sorted(failed_rows["dataset"].dropna().unique())
    successful_datasets = successful_rows["dataset"].nunique()

    print(f"\n{method_label}:")
    print(f"  successful datasets: {successful_datasets}/{len(expected_datasets)}")
    print(f"  datasets with NaN metric_error: {len(failed_datasets)}")
    print(f"  datasets with no result row: {len(missing_datasets)}")

    if not successful_rows.empty:
        max_feature_count = successful_rows["n_features"].max()
        max_sample_count = successful_rows["n_samples_train_per_fold"].max()
        max_feature_datasets = successful_rows.loc[
            successful_rows["n_features"] == max_feature_count,
            ["dataset", "n_features", "n_samples_train_per_fold"],
        ].sort_values("dataset")
        max_sample_datasets = successful_rows.loc[
            successful_rows["n_samples_train_per_fold"] == max_sample_count,
            ["dataset", "n_samples_train_per_fold", "n_features"],
        ].sort_values("dataset")
        print(
            "  highest feature-count successful datasets: "
            + ", ".join(
                f"{row.dataset} (n_features={int(row.n_features)}, n_samples={int(row.n_samples_train_per_fold)})"
                for row in max_feature_datasets.itertuples(index=False)
            )
        )
        print(
            "  highest sample-count successful datasets: "
            + ", ".join(
                f"{row.dataset} (n_samples={int(row.n_samples_train_per_fold)}, n_features={int(row.n_features)})"
                for row in max_sample_datasets.itertuples(index=False)
            )
        )

    if failed_datasets:
        failed_summary = (
            failed_rows.loc[:, ["dataset", "metric_error", "metric_error_val", "time_train_s", "time_infer_s"]]
            .drop_duplicates(subset=["dataset"])
            .merge(
                task_metadata.loc[:, ["dataset", "n_samples_train_per_fold", "n_features"]],
                on="dataset",
                how="left",
            )
            .sort_values("dataset")
        )
        print("  rows with NaN metric_error:")
        print(failed_summary.to_string(index=False))

    if missing_datasets:
        missing_summary = (
            task_metadata.loc[task_metadata["dataset"].isin(missing_datasets), ["dataset", "n_samples_train_per_fold", "n_features"]]
            .sort_values("dataset")
        )
        print("  missing datasets:")
        print(missing_summary.to_string(index=False))


def print_result_diagnostics(
    ablation_model_results: pd.DataFrame,
    ta_results: pd.DataFrame,
) -> None:
    task_metadata = load_task_metadata()
    prep_lr_method = resolve_method_name(ta_results, PREP_LR_METHOD_ALIASES)
    prep_lgb_method = resolve_method_name(ta_results, PREP_LGB_METHOD_ALIASES)
    ta_results_selected = ta_results.loc[ta_results.fold.isin(PLOT_FOLDS)].copy()
    print_successful_dataset_counts(ablation_model_results, ta_results_selected)
    print_missing_result_diagnostics(
        ablation_model_results,
        method_label="AutoFeat",
        method_names=["AutoFeatLinearModel_c1_BAG_L1"],
        task_metadata=task_metadata,
    )
    print_missing_result_diagnostics(
        ablation_model_results,
        method_label="OpenFE",
        method_names=["OpenFELGBModel_c1_BAG_L1"],
        task_metadata=task_metadata,
    )
    print_missing_result_diagnostics(
        ta_results,
        method_label="PrepLR",
        method_names=[prep_lr_method],
        task_metadata=task_metadata,
    )
    print_missing_result_diagnostics(
        ta_results,
        method_label="PrepLGB",
        method_names=[prep_lgb_method],
        task_metadata=task_metadata,
    )


def select_method_rows(
    df: pd.DataFrame,
    *,
    method_names: list[str],
    value_col: str = "metric_error",
) -> pd.DataFrame:
    out = df.loc[df.method.isin(method_names), ["dataset", "fold", value_col]].copy()
    out = out.drop_duplicates(subset=["dataset", "fold"], keep="first")
    return out


def build_foldwise_comparison_frame(
    df: pd.DataFrame,
    *,
    baseline_method_names: list[str],
    competitor_method_names: list[str],
    baseline_col: str,
    competitor_col: str,
) -> pd.DataFrame:
    baseline_df = select_method_rows(df, method_names=baseline_method_names).rename(
        columns={"metric_error": baseline_col}
    )
    competitor_df = select_method_rows(df, method_names=competitor_method_names).rename(
        columns={"metric_error": competitor_col}
    )
    return baseline_df.merge(competitor_df, on=["dataset", "fold"], how="outer")


def summarize_win_tie_lose_all_folds(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    competitor_col: str,
    expected_datasets: list[str],
    expected_folds: tuple[int, int, int] = (0, 1, 2),
    eps: float = 1e-12,
) -> dict[str, int]:
    """
    Summarize wins/ties/losses using all three folds.

    Rule:
    - If the competitor is missing any fold, count as a lose.
    - If the competitor has all folds and the baseline is missing at least one fold, count as a win by failure.
    - Otherwise, count as a win by comparison only if the competitor beats the baseline on all three folds.
    - All other complete cases count as a tie.
    """
    outcomes = {"win_comparison": 0, "win_failure": 0, "tie": 0, "lose": 0, "n": 0}
    expected_fold_set = set(expected_folds)

    for dataset in expected_datasets:
        dataset_df = df.loc[df["dataset"] == dataset, ["fold", baseline_col, competitor_col]].copy()
        dataset_df = dataset_df.drop_duplicates(subset=["fold"], keep="first")
        dataset_df = dataset_df.loc[dataset_df["fold"].isin(expected_fold_set)]
        outcomes["n"] += 1

        if dataset_df.empty or set(dataset_df["fold"].tolist()) != expected_fold_set:
            outcomes["lose"] += 1
            continue

        competitor_missing = dataset_df[competitor_col].isna().any()
        baseline_missing = dataset_df[baseline_col].isna().any()

        if competitor_missing:
            outcomes["lose"] += 1
            continue

        if baseline_missing:
            outcomes["win_failure"] += 1
            continue

        ordered = dataset_df.sort_values("fold")
        diff = ordered[competitor_col].to_numpy(dtype=float) - ordered[baseline_col].to_numpy(dtype=float)
        if np.all(diff < -eps):
            outcomes["win_comparison"] += 1
        else:
            outcomes["tie"] += 1

    return outcomes


def plot_win_tie_lose_bars(
    comparisons: list[tuple[str, pd.DataFrame, str, str]],
    *,
    expected_datasets: list[str],
    save_path: str,
    title: str,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    transparent: bool = True,
) -> pd.DataFrame:
    """
    Plot normalized win/tie/lose bars for several competitor-vs-baseline comparisons.

    Bars are normalized to 100% of common datasets. Segment labels show raw counts.
    """
    rows = []
    for label, df, baseline_col, competitor_col in comparisons:
        summary = summarize_win_tie_lose_all_folds(
            df,
            baseline_col=baseline_col,
            competitor_col=competitor_col,
            expected_datasets=expected_datasets,
        )
        rows.append({
            "comparison": label,
            "baseline_col": baseline_col,
            "competitor_col": competitor_col,
            **summary,
        })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    if figsize is None:
        figsize = (9.5, 0.75 * len(summary_df) + 1.6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    colors = {
        "win_comparison": "#2ca02c",
        "win_failure": "#1f77b4",
        "tie": "#9e9e9e",
        "lose": "#d62728",
    }

    y = np.arange(len(summary_df))
    left = np.zeros(len(summary_df), dtype=float)
    total = summary_df["n"].replace(0, np.nan).to_numpy(dtype=float)

    for key, label in [
        ("win_comparison", "Win by comparison"),
        ("win_failure", "Win by failure"),
        ("tie", "Tie"),
        ("lose", "Lose"),
    ]:
        counts = summary_df[key].to_numpy(dtype=float)
        widths = np.where(np.isnan(total), 0.0, np.divide(counts, total, out=np.zeros_like(counts), where=total > 0) * 100.0)
        bars = ax.barh(
            y,
            widths,
            left=left,
            color=colors[key],
            edgecolor="white",
            linewidth=1.0,
            label=label,
        )
        for bar, count in zip(bars, summary_df[key].to_numpy(dtype=int)):
            if count <= 0:
                continue
            width = bar.get_width()
            if width < 7:
                continue
            ax.text(
                bar.get_x() + width / 2.0,
                bar.get_y() + bar.get_height() / 2.0,
                f"{count}",
                ha="center",
                va="center",
                fontsize=11,
                color="black",
            )
        left += widths

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.comparison} (n={row.n})" for row in summary_df.itertuples(index=False)])
    ax.set_xlabel("Share of common datasets")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(save_path, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    return summary_df


def plot_prep_win_tie_lose(
    ta_results: pd.DataFrame,
    task_metadata: pd.DataFrame,
    *,
    suffix: str = "",
) -> None:
    suffix_text = f"_{suffix}" if suffix else ""
    expected_datasets = task_metadata["dataset"].dropna().tolist()
    prep_lr_method = resolve_method_name(ta_results, PREP_LR_METHOD_ALIASES)
    prep_lgb_method = resolve_method_name(ta_results, PREP_LGB_METHOD_ALIASES)
    prep_lr_frame = build_foldwise_comparison_frame(
        ta_results,
        baseline_method_names=["LR (default)"],
        competitor_method_names=[prep_lr_method],
        baseline_col="LinearModel",
        competitor_col="PrepLR",
    )
    prep_lgb_frame = build_foldwise_comparison_frame(
        ta_results,
        baseline_method_names=["GBM (default)"],
        competitor_method_names=[prep_lgb_method],
        baseline_col="LightGBM",
        competitor_col="PrepLGB",
    )
    comparisons = [
        ("PrepLR vs LinearModel", prep_lr_frame, "LinearModel", "PrepLR"),
        ("PrepLGB vs LightGBM", prep_lgb_frame, "LightGBM", "PrepLGB"),
    ]

    summary_df = plot_win_tie_lose_bars(
        comparisons,
        expected_datasets=expected_datasets,
        save_path=f"{SAVE_PATH}/prep_win_tie_lose{suffix_text}.pdf",
        title="Prep models vs library baselines (all 3 folds required for win)",
    )

    if not summary_df.empty:
        latex_table = summary_df.loc[:, ["comparison", "win_comparison", "win_failure", "tie", "lose", "n"]].copy()
        latex_table = latex_table.rename(
            columns={
                "comparison": "Comparison",
                "win_comparison": "Win comparison",
                "win_failure": "Win failure",
                "tie": "Tie",
                "lose": "Lose",
                "n": "N",
            }
        )
        latex_path = SAVE_PATH / f"prep_win_tie_lose{suffix_text}.tex"
        latex_path.write_text(
            latex_table.to_latex(index=False, escape=True, column_format="lrrrrr"),
            encoding="utf-8",
        )
        print(f"\nWrote LaTeX table to: {latex_path}")
        print(latex_table.to_latex(index=False, escape=True, column_format="lrrrrr"))


def select_best_by_validation(
    df: pd.DataFrame,
    *,
    out_col: str,
    candidate_col: str,
    candidate_val_col: str,
    fallback_col: str,
    fallback_val_col: str,
) -> None:
    """
    Pick the lower-validation-error option between two columns.

    The selected value is taken from `candidate_col` when its validation error
    is available and no worse than the fallback; otherwise we use fallback.
    """
    choose_candidate = (
        df[candidate_val_col].notna()
        & (df[fallback_val_col].isna() | (df[candidate_val_col] <= df[fallback_val_col]))
    )
    df[out_col] = np.where(choose_candidate, df[candidate_col], df[fallback_col])


def select_best_by_validation_across_folds(
    df: pd.DataFrame,
    *,
    out_col: str,
    candidate_col: str,
    candidate_val_col: str,
    fallback_col: str,
    fallback_val_col: str,
) -> None:
    """
    Select the better method per dataset using mean validation error across folds.

    The dataset-level decision is then broadcast back to every fold row so that
    downstream plots can still average the fold-wise improvements.
    """
    dataset_scores = (
        df.loc[:, ["dataset", candidate_val_col, fallback_val_col]]
        .groupby("dataset", as_index=False)
        .mean(numeric_only=True)
    )
    choose_candidate = (
        dataset_scores[candidate_val_col].notna()
        & (dataset_scores[fallback_val_col].isna() | (dataset_scores[candidate_val_col] <= dataset_scores[fallback_val_col]))
    )
    choice_map = dict(zip(dataset_scores["dataset"], choose_candidate))
    df[out_col] = np.where(
        df["dataset"].map(choice_map).fillna(False),
        df[candidate_col],
        df[fallback_col],
    )


def build_autofeat_frame(
    ablation_model_results: pd.DataFrame,
    ta_results: pd.DataFrame,
) -> pd.DataFrame:
    prep_lr_method = resolve_method_name(ta_results, PREP_LR_METHOD_ALIASES)
    autofeat_comp_df = ablation_model_results.loc[
        ablation_model_results.method == "AutoFeatLinearModel_c1_BAG_L1",
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].rename(columns={"metric_error": "AutoFeat", "metric_error_val": "AutoFeat_val"})

    base_lr_df = ablation_model_results.loc[
        ablation_model_results.method == "LinearModel_c1_BAG_L1",
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].rename(columns={"metric_error": "LinearModel", "metric_error_val": "LinearModel_val"})

    autofeat_comp_df = autofeat_comp_df.merge(base_lr_df, on=["dataset", "fold"], how="outer")

    only_order2 = ablation_model_results.loc[
        ablation_model_results.method == "AutoFeatLinearModel_c2_BAG_L1",
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].rename(columns={
        "metric_error": "AutoFeat (2-order)",
        "metric_error_val": "AutoFeat (2-order)_val",
    })

    autofeat_comp_df = autofeat_comp_df.merge(only_order2, on=["dataset", "fold"], how="outer")

    prep_lr = ta_results.loc[
        np.logical_and(
            ta_results.method == prep_lr_method,
            ta_results.fold.isin(PLOT_FOLDS),
        ),
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].copy()
    prep_lr = prep_lr.rename(columns={"metric_error": "PrepLinearModel", "metric_error_val": "PrepLinearModel_val"})

    autofeat_comp_df = autofeat_comp_df.merge(prep_lr, on=["dataset", "fold"], how="outer")
    return autofeat_comp_df


def build_openfe_frame(
    ablation_model_results: pd.DataFrame,
    ta_results: pd.DataFrame,
) -> pd.DataFrame:
    prep_lgb_method = resolve_method_name(ta_results, PREP_LGB_METHOD_ALIASES)
    openfe_comp_df = ablation_model_results.loc[
        ablation_model_results.method == "OpenFELGBModel_c1_BAG_L1",
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].rename(columns={"metric_error": "OpenFE", "metric_error_val": "OpenFE_val"})

    base_lgb_df = ta_results.loc[
        np.logical_and(
            ta_results.method == "GBM (default)",
            ta_results.fold.isin(PLOT_FOLDS),
        ),
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].copy()
    base_lgb_df = base_lgb_df.rename(columns={"metric_error": "LightGBM", "metric_error_val": "LightGBM_val"})

    openfe_comp_df = openfe_comp_df.merge(base_lgb_df, on=["dataset", "fold"], how="outer")

    prep_lgb_df = ta_results.loc[
        np.logical_and(
            ta_results.method == prep_lgb_method,
            ta_results.fold.isin(PLOT_FOLDS),
        ),
        ["dataset", "fold", "metric_error", "metric_error_val"],
    ].copy()
    prep_lgb_df = prep_lgb_df.rename(columns={"metric_error": "PrepLightGBM", "metric_error_val": "PrepLightGBM_val"})

    openfe_comp_df = openfe_comp_df.merge(prep_lgb_df, on=["dataset", "fold"], how="outer")
    return openfe_comp_df


def add_best_of_columns(
    autofeat_comp_df: pd.DataFrame,
    openfe_comp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str, str]:
    """
    Add per-dataset best-of columns selected by validation error, then
    use the chosen method's test error for plotting.
    """
    autofeat_best_col = "Best AutoFeat"
    prep_lr_best_col = "Best PrepLR"
    openfe_best_col = "Best OpenFE"
    prep_lgb_best_col = "Best PrepLGB"

    autofeat_best_df = autofeat_comp_df.copy()
    if should_average_folds(PLOT_FOLDS):
        select_best_by_validation_across_folds(
            autofeat_best_df,
            out_col=autofeat_best_col,
            candidate_col="AutoFeat",
            candidate_val_col="AutoFeat_val",
            fallback_col="LinearModel",
            fallback_val_col="LinearModel_val",
        )
        select_best_by_validation_across_folds(
            autofeat_best_df,
            out_col=prep_lr_best_col,
            candidate_col="PrepLinearModel",
            candidate_val_col="PrepLinearModel_val",
            fallback_col="LinearModel",
            fallback_val_col="LinearModel_val",
        )
    else:
        select_best_by_validation(
            autofeat_best_df,
            out_col=autofeat_best_col,
            candidate_col="AutoFeat",
            candidate_val_col="AutoFeat_val",
            fallback_col="LinearModel",
            fallback_val_col="LinearModel_val",
        )
        select_best_by_validation(
            autofeat_best_df,
            out_col=prep_lr_best_col,
            candidate_col="PrepLinearModel",
            candidate_val_col="PrepLinearModel_val",
            fallback_col="LinearModel",
            fallback_val_col="LinearModel_val",
        )

    openfe_best_df = openfe_comp_df.copy()
    if should_average_folds(PLOT_FOLDS):
        select_best_by_validation_across_folds(
            openfe_best_df,
            out_col=openfe_best_col,
            candidate_col="OpenFE",
            candidate_val_col="OpenFE_val",
            fallback_col="LightGBM",
            fallback_val_col="LightGBM_val",
        )
        select_best_by_validation_across_folds(
            openfe_best_df,
            out_col=prep_lgb_best_col,
            candidate_col="PrepLightGBM",
            candidate_val_col="PrepLightGBM_val",
            fallback_col="LightGBM",
            fallback_val_col="LightGBM_val",
        )
    else:
        select_best_by_validation(
            openfe_best_df,
            out_col=openfe_best_col,
            candidate_col="OpenFE",
            candidate_val_col="OpenFE_val",
            fallback_col="LightGBM",
            fallback_val_col="LightGBM_val",
        )
        select_best_by_validation(
            openfe_best_df,
            out_col=prep_lgb_best_col,
            candidate_col="PrepLightGBM",
            candidate_val_col="PrepLightGBM_val",
            fallback_col="LightGBM",
            fallback_val_col="LightGBM_val",
        )

    return (
        autofeat_best_df,
        openfe_best_df,
        autofeat_best_col,
        prep_lr_best_col,
        openfe_best_col,
        prep_lgb_best_col,
    )


def plot_fe_boxplots(
    autofeat_comp_df: pd.DataFrame,
    openfe_comp_df: pd.DataFrame,
    *,
    suffix: str = "",
    autofeat_col: str = "AutoFeat",
    openfe_col: str = "OpenFE",
    prep_lr_col: str = "PrepLinearModel",
    prep_lgb_col: str = "PrepLightGBM",
    autofeat_label: str = "Autofeat",
    openfe_label: str = "OpenFE",
    subset_figsize: tuple[float, float] = (6.8, 2.6),
    full_figsize: tuple[float, float] = (12, 6),
) -> None:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    suffix_text = f"_{suffix}" if suffix else ""
    plot_fold_text = f"_{fold_suffix(PLOT_FOLDS)}"

    if should_average_folds(PLOT_FOLDS):
        autofeat_comp_df = average_relative_improvements_by_dataset(
            autofeat_comp_df,
            baseline_col="LinearModel",
            competitor_cols=[autofeat_col, prep_lr_col],
        )
        openfe_comp_df = average_relative_improvements_by_dataset(
            openfe_comp_df,
            baseline_col="LightGBM",
            competitor_cols=[openfe_col, prep_lgb_col],
        )

    boxplot_two_dataframes_pubready(
        df_left=autofeat_comp_df,
        left_baseline_col="LinearModel",
        left_competitor_cols=[autofeat_col, prep_lr_col],
        df_right=openfe_comp_df,
        right_baseline_col="LightGBM",
        right_competitor_cols=[openfe_col, prep_lgb_col],
        left_labels=[autofeat_label, "PrepLR"],
        right_labels=[openfe_label, "PrepLGB"],
        mode="relative",
        cap_left=[-0.25, 1],
        cap_right=[-0.1, 0.25],
        horizontal=True,
        share_scale=False,
        save_path=f"{SAVE_PATH}/autoFE_boxplots_subset{suffix_text}{plot_fold_text}.pdf",
        dpi=300,
        transparent=True,
        font_size=14.0,
        title_size=14.0,
        tick_size=10.0,
        figsize=subset_figsize,
    )

    boxplot_two_dataframes_pubready(
        dropna=False,
        df_left=autofeat_comp_df,
        left_baseline_col="LinearModel",
        left_competitor_cols=[autofeat_col, prep_lr_col],
        df_right=openfe_comp_df,
        right_baseline_col="LightGBM",
        right_competitor_cols=[openfe_col, prep_lgb_col],
        left_labels=[autofeat_label, "PrepLR"],
        right_labels=[openfe_label, "PrepLGB"],
        mode="relative",
        cap_left=[-0.5, 1],
        cap_right=[-0.5, 0.25],
        horizontal=True,
        share_scale=False,
        save_path=f"{SAVE_PATH}/autoFE_boxplots_full{suffix_text}{plot_fold_text}.pdf",
        dpi=300,
        figsize=full_figsize,
        font_size=14.0,
        title_size=14.0,
        tick_size=12.0,
    )


def main() -> None:
    ta_results = load_tabarena_hpo_results()
    ablation_model_results = load_ablation_results()

    print_result_diagnostics(ablation_model_results, ta_results)

    autofeat_comp_df = build_autofeat_frame(ablation_model_results, ta_results)
    openfe_comp_df = build_openfe_frame(ablation_model_results, ta_results)

    plot_fe_boxplots(autofeat_comp_df, openfe_comp_df)
    task_metadata = load_task_metadata()
    plot_prep_win_tie_lose(ta_results, task_metadata)

    (
        autofeat_best_df,
        openfe_best_df,
        autofeat_best_col,
        prep_lr_best_col,
        openfe_best_col,
        prep_lgb_best_col,
    ) = add_best_of_columns(autofeat_comp_df, openfe_comp_df)
    plot_fe_boxplots(
        autofeat_best_df,
        openfe_best_df,
        suffix="best_of_default_and_fe",
        autofeat_col=autofeat_best_col,
        prep_lr_col=prep_lr_best_col,
        openfe_col=openfe_best_col,
        prep_lgb_col=prep_lgb_best_col,
        autofeat_label="Autofeat",
        openfe_label="OpenFE",
        subset_figsize=(8.8, 2.6),
        full_figsize=(13.5, 6),
    )


if __name__ == "__main__":
    main()
