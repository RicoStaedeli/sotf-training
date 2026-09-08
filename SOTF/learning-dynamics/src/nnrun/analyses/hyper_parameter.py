from __future__ import annotations

from typing import Any, Dict, List

import logging

from joblib import parallel_backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from nnrun.analyses.base import AnalysisResult
from nnrun.zoo_info import zoo_plot_name

log = logging.getLogger(__name__)


def discover_hyper_cols(
    df: pd.DataFrame,
    include_prefixes: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    if include_prefixes is None:
        include_prefixes = [
            "config.training::",
            "config.model::",
            "config.optim::",
            "config.seed",
        ]
    if exclude is None:
        exclude = set()

    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(c.startswith(p) for p in include_prefixes):
            cols.append(c)
    return sorted(cols)


def _plot_importance(df_imp: pd.DataFrame, title: str, dpi: int = 200) -> plt.Figure:
    show = df_imp.sort_values("importance_mean", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(show))), dpi=dpi)
    ax.barh(show["feature"], show["importance_mean"], xerr=show["importance_std"])
    ax.set_title(title, pad=10)
    ax.set_xlabel("Permutation importance")
    fig.tight_layout()
    return fig


def _plot_numeric_assoc(df_num: pd.DataFrame, title: str, dpi: int = 200) -> plt.Figure:
    show = df_num.sort_values("abs_spearman_rho", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(show))), dpi=dpi)
    ax.barh(show["feature"], show["spearman_rho"])
    ax.set_title(title, pad=10)
    ax.set_xlabel("Spearman rho")
    fig.tight_layout()
    return fig


def run_hyper_analysis(df: pd.DataFrame, config: Dict[str, Any], registry: Dict[str, Any]) -> AnalysisResult:
    """
    Per dataset:
    - keep the last epoch per model
    - identify hyperparameter columns
    - compute univariate associations:
        * numeric features -> Spearman correlation
        * categorical features -> Kruskal-Wallis test
    - compute multivariate feature contribution with permutation importance
    """

    log.info("Running hyperparameter analysis")

    metric = config["metric"]
    epoch_col = config.get("epoch_col", "training_iteration")
    model_col = config.get("model_col", "model_name")
    epoch_range = tuple(config.get("epoch_range", [])) or None
    png_dpi = int(config.get("png_dpi", 200))
    higher_is_better = bool(config.get("higher_is_better", True))

    rf_n_estimators = int(config.get("rf_n_estimators", 300))
    rf_random_state = int(config.get("rf_random_state", 42))
    perm_n_repeats = int(config.get("perm_n_repeats", 30))
    rf_n_jobs = int(config.get("rf_n_jobs", 1))
    perm_n_jobs = int(config.get("perm_n_jobs", 1))
    parallel_backend_name = str(config.get("parallel_backend", "threading"))
    min_group_size = int(config.get("min_group_size", 2))

    figures: List[Dict[str, plt.Figure]] = []
    tables: Dict[str, pd.DataFrame] = {}

    required = ["dataset_name", model_col, epoch_col, metric]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"hyper_analysis: missing required column '{c}'")

    d = df.copy()
    d[epoch_col] = pd.to_numeric(d[epoch_col], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")

    if epoch_range is not None and len(epoch_range) == 2:
        e0, e1 = epoch_range
        d = d[(d[epoch_col] >= e0) & (d[epoch_col] <= e1)].copy()

    d = d.dropna(subset=[epoch_col, metric])
    datasets = sorted(d["dataset_name"].dropna().unique().tolist())

    all_final_tables: list[pd.DataFrame] = []
    all_numeric_assoc: list[pd.DataFrame] = []
    all_categorical_assoc: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []

    force_cat_suffixes = (
        "::optimizer",
        "::nlin",
        "::init_type",
        "::type",
        "::scheduler",
        "::dataloader",
        "::loss",
    )

    for ds in datasets:
        log.info("Hyper analysis for dataset: %s", ds)

        dsd = d[d["dataset_name"] == ds].copy()
        dsd = dsd.sort_values([model_col, epoch_col])

        idx = dsd.groupby(model_col, sort=False)[epoch_col].idxmax()
        d_last = dsd.loc[idx].copy()

        hyper_cols = discover_hyper_cols(d_last, include_prefixes=config.get("hyper_prefixes"))
        if not hyper_cols:
            log.warning("No hyperparameter columns found for dataset=%s", ds)
            continue

        keep_cols = ["dataset_name", model_col, epoch_col, metric] + hyper_cols
        final_tbl = d_last[keep_cols].copy()
        final_tbl = final_tbl.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)

        tables[f"hyper_final_table__{ds}"] = final_tbl
        all_final_tables.append(final_tbl)

        X = final_tbl[hyper_cols].copy()
        y = pd.to_numeric(final_tbl[metric], errors="coerce").astype(float)

        cat_cols = [c for c in X.columns if X[c].dtype == "object" or c.endswith(force_cat_suffixes)]
        num_cols = [c for c in X.columns if c not in cat_cols]

        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        # Drop all-NaN columns
        drop_cols = [c for c in X.columns if X[c].isna().all()]

        # Drop constant columns (including all-missing treated as one value)
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]

        cols_to_drop = sorted(set(drop_cols) | set(constant_cols))
        if cols_to_drop:
            log.info(
                "Dropping non-informative hyperparameter columns for dataset=%s: %s",
                ds,
                cols_to_drop,
            )
            X = X.drop(columns=cols_to_drop)

        cat_cols = [c for c in X.columns if X[c].dtype == "object" or c.endswith(force_cat_suffixes)]
        num_cols = [c for c in X.columns if c not in cat_cols]

        # ------------------------------------------------------------------
        # 1) Numeric features: Spearman
        # ------------------------------------------------------------------
        num_rows: list[dict[str, Any]] = []
        for c in num_cols:
            x = pd.to_numeric(X[c], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                continue

            rho, pval = spearmanr(x[mask], y[mask])
            num_rows.append(
                {
                    "dataset_name": ds,
                    "feature": c,
                    "n": int(mask.sum()),
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "abs_spearman_rho": abs(float(rho)) if pd.notna(rho) else np.nan,
                    "p_value": float(pval) if pd.notna(pval) else np.nan,
                }
            )

        df_num = pd.DataFrame(num_rows)
        if not df_num.empty:
            df_num = df_num.sort_values(["abs_spearman_rho", "p_value"], ascending=[False, True]).reset_index(drop=True)

        tables[f"{ds}_hyper_numeric_assoc"] = df_num
        all_numeric_assoc.append(df_num)

        if not df_num.empty:
            figures.append(
                {
                    f"{ds}_hyper_numeric_assoc_plot": _plot_numeric_assoc(
                        df_num, f"{zoo_plot_name(ds)} | Numeric hyperparameter association", dpi=png_dpi
                    )
                }
            )

        # ------------------------------------------------------------------
        # 2) Categorical features: Kruskal-Wallis
        # ------------------------------------------------------------------
        cat_rows: list[dict[str, Any]] = []
        for c in cat_cols:
            tmp = pd.DataFrame(
                {
                    "feature_value": X[c].astype("object").fillna("__MISSING__"),
                    "metric": y,
                }
            ).dropna(subset=["metric"])

            counts = tmp["feature_value"].value_counts()
            valid_levels = counts[counts >= min_group_size].index.tolist()
            tmp = tmp[tmp["feature_value"].isin(valid_levels)]

            if tmp["feature_value"].nunique() < 2:
                continue

            grouped = [g["metric"].values for _, g in tmp.groupby("feature_value")]
            if len(grouped) < 2:
                continue

            try:
                stat, pval = kruskal(*grouped)
            except Exception:
                stat, pval = np.nan, np.nan

            group_means = (
                tmp.groupby("feature_value", dropna=False)["metric"]
                .mean()
                .sort_values(ascending=False)
                .to_dict()
            )

            cat_rows.append(
                {
                    "dataset_name": ds,
                    "feature": c,
                    "n": int(len(tmp)),
                    "n_levels": int(tmp["feature_value"].nunique()),
                    "kruskal_stat": float(stat) if pd.notna(stat) else np.nan,
                    "p_value": float(pval) if pd.notna(pval) else np.nan,
                    "best_level_mean": float(max(group_means.values())) if group_means else np.nan,
                    "worst_level_mean": float(min(group_means.values())) if group_means else np.nan,
                    "range_of_level_means": (
                        float(max(group_means.values()) - min(group_means.values()))
                        if group_means
                        else np.nan
                    ),
                    "level_means": group_means,
                }
            )

        df_cat = pd.DataFrame(cat_rows)
        if not df_cat.empty:
            df_cat = df_cat.sort_values(
                ["p_value", "range_of_level_means"],
                ascending=[True, False],
            ).reset_index(drop=True)

        tables[f"{ds}_hyper_categorical_assoc"] = df_cat
        all_categorical_assoc.append(df_cat)

        # ------------------------------------------------------------------
        # 3) Joint contribution: random forest + permutation importance
        # ------------------------------------------------------------------
        if X.shape[1] > 0 and len(X) >= 5:
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                            ]
                        ),
                        num_cols,
                    ),
                    (
                        "cat",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        cat_cols,
                    ),
                ],
                remainder="drop",
            )

            model = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=rf_n_estimators,
                            random_state=rf_random_state,
                            n_jobs=rf_n_jobs,
                        ),
                    ),
                ]
            )

            valid = y.notna()
            X_fit = X.loc[valid].copy()
            y_fit = y.loc[valid].copy()

            if len(X_fit) >= 5:
                with parallel_backend(parallel_backend_name):
                    model.fit(X_fit, y_fit)

                    perm = permutation_importance(
                        model,
                        X_fit,
                        y_fit,
                        n_repeats=perm_n_repeats,
                        random_state=rf_random_state,
                        n_jobs=perm_n_jobs,
                    )

                df_imp = pd.DataFrame(
                    {
                        "dataset_name": ds,
                        "feature": X_fit.columns,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                    }
                ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

                tables[f"{ds}_hyper_permutation_importance"] = df_imp
                all_importances.append(df_imp)

                figures.append(
                    {
                        f"{ds}_hyper_permutation_importance_plot": _plot_importance(
                            df_imp, f"{zoo_plot_name(ds)} | Permutation importance", dpi=png_dpi
                        )
                    }
                )

    if all_final_tables:
        tables["hyper_final_table_all"] = pd.concat(all_final_tables, ignore_index=True)
    if all_numeric_assoc:
        tables["hyper_numeric_assoc_all"] = pd.concat(all_numeric_assoc, ignore_index=True)
    if all_categorical_assoc:
        tables["hyper_categorical_assoc_all"] = pd.concat(all_categorical_assoc, ignore_index=True)
    if all_importances:
        tables["hyper_permutation_importance_all"] = pd.concat(all_importances, ignore_index=True)

    return AnalysisResult(tables=tables, figures=figures)
