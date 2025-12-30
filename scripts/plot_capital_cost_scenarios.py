# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
Plot scenario comparison of BF-BOF capital cost differences.

Compares model BF-BOF nodal capital costs against computed annuitized
investment for existing integrated steelworks and plots the difference
across scenarios in a single figure.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts._helpers import (
    configure_logging,
    mock_snakemake,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

DEFAULT_MARKER_YEARS = [2020, 2025, 2030, 2035, 2040, 2045, 2050]


def _params_get(params, key, default=None):
    getter = getattr(params, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(params, key, default)


def _normalize_scenario_names(value: None | str | Sequence[str]) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    normalized = [str(v) for v in value if v not in (None, "", "None")]
    if not normalized:
        return None
    if normalized == ["all"]:
        return None
    return normalized


def _resolve_results_root(root_param: str | Path) -> Path:
    root = Path(root_param)
    if not root.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / root
    return root


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_plotting_default(project_root: Path) -> dict:
    default_path = project_root / "config" / "plotting.default.yaml"
    if not default_path.exists():
        logger.warning("plotting.default.yaml not found at %s", default_path)
        return {}
    with default_path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        logger.warning("plotting.default.yaml did not load as dict, got %s", type(loaded).__name__)
        return {}
    plotting = loaded.get("plotting")
    if isinstance(plotting, dict):
        return plotting
    return loaded


def _load_discount_rate(config_path: Path) -> tuple[float, list[str]]:
    discount_rate = 0.07
    countries = []
    if not config_path.exists():
        return discount_rate, countries
    try:
        with config_path.open("r") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Could not read discount rate from %s: %s", config_path, exc)
        return discount_rate, countries
    if isinstance(cfg, dict):
        fill_values = cfg.get("costs", {}).get("fill_values", {}) or {}
        if "discount rate" in fill_values:
            discount_rate = float(fill_values["discount rate"])
        elif "discount_rate" in fill_values:
            discount_rate = float(fill_values["discount_rate"])
        countries = cfg.get("countries", []) or []
    return discount_rate, countries


def _pick_cost_value(costs_bf: pd.DataFrame, param: str, year: int) -> float | None:
    rows = costs_bf[costs_bf["parameter"].str.lower() == param.lower()]
    if rows.empty:
        return None
    if "currency_year" in rows.columns:
        yrs = pd.to_numeric(rows["currency_year"], errors="coerce")
        exact = rows[yrs == float(year)]
        if not exact.empty:
            return float(exact.iloc[0]["value"])
        le = rows[yrs.notna() & (yrs <= float(year))]
        if not le.empty:
            return float(le.sort_values("currency_year", ascending=False).iloc[0]["value"])
    return float(rows.iloc[0]["value"])


def _compute_annuitized_costs(
    costs: pd.DataFrame,
    steel_plants: pd.DataFrame,
    discount_rate: float,
    years: Sequence[int],
) -> pd.DataFrame:
    bf_mask = costs["technology"].str.contains("blast furnace", case=False, na=False)
    costs_bf = costs[bf_mask].copy()
    inv_val = _pick_cost_value(costs_bf, "investment", 2020) or 7637406.0
    life = (
        _pick_cost_value(costs_bf, "lifetime", 2020)
        or _pick_cost_value(costs_bf, "economic_lifetime", 2020)
        or 40.0
    )
    fom = _pick_cost_value(costs_bf, "FOM", 2020) or 0.1418

    steel_plants = steel_plants.copy()
    steel_plants[["start_year", "retired_year"]] = steel_plants[
        ["start_year", "retired_year"]
    ].apply(pd.to_numeric, errors="coerce")

    if "process" in steel_plants.columns:
        is_int_mask = steel_plants["process"].str.contains(
            "Integrated steelworks", case=False, na=False
        )
        steel_plants = steel_plants[is_int_mask].copy()

    steel_plants["retired_year_calc"] = steel_plants["retired_year"]
    steel_plants.loc[
        steel_plants["retired_year_calc"].isna()
        & steel_plants["start_year"].notna(),
        "retired_year_calc",
    ] = steel_plants["start_year"] + life

    r = float(discount_rate)
    results = []
    for year in years:
        active = steel_plants[
            (steel_plants["start_year"].fillna(-1) <= year)
            & (
                steel_plants["retired_year_calc"].isna()
                | (steel_plants["retired_year_calc"] >= year)
            )
        ]
        total_ktpa = active["capacity"].sum()
        if year == years[0]:
            logger.info("Integrated steelworks active capacity in %s: %.2f ktpa", year, total_ktpa)
        total_t_per_year = total_ktpa * 1000.0
        total_t_per_h = total_t_per_year / 8760.0 if total_t_per_year > 0 else 0.0

        inv_y = _pick_cost_value(costs_bf, "investment", year) or inv_val
        life_y = (
            _pick_cost_value(costs_bf, "lifetime", year)
            or _pick_cost_value(costs_bf, "economic_lifetime", year)
            or life
        )
        fom_y = _pick_cost_value(costs_bf, "FOM", year) or fom

        overnight = float(inv_y) * total_t_per_h if total_t_per_h > 0 else 0.0
        n = float(life_y)
        annuity = r / (1 - (1 + r) ** (-n)) if (r > 0 and n > 0) else (1.0 / n if n > 0 else 0.0)
        fom_frac = float(fom_y) / 100.0 if float(fom_y) > 1 else float(fom_y)
        annualized = overnight * (annuity + fom_frac) if overnight > 0 else 0.0
        per_ton = annualized / total_t_per_year if total_t_per_year > 0 else np.nan

        results.append(
            {
                "year": year,
                "total_ktpa": total_ktpa,
                "overnight_EUR": overnight,
                "annualized_EUR_per_year": annualized,
                "annualized_EUR_per_ton": per_ton,
            }
        )
    return pd.DataFrame(results)


def _list_scenarios(results_root: Path, names: Sequence[str] | None) -> list[Tuple[str, Path]]:
    if not results_root.exists():
        logger.warning("Results root does not exist: %s", results_root)
        return []
    scenarios = []
    target_names = list(names) if names else None
    if target_names:
        iterate = target_names
    else:
        iterate = sorted(entry.name for entry in results_root.iterdir() if entry.is_dir())

    for scenario in iterate:
        nc_dir = results_root / scenario / "csvs" / "individual"
        if not nc_dir.exists():
            continue
        if any(nc_dir.glob("nodal_costs*___*.csv")):
            scenarios.append((scenario, nc_dir))
    return scenarios


def _normalize_marker_years(value: Sequence[int] | Sequence[str] | str | None) -> list[int]:
    if value is None:
        return list(DEFAULT_MARKER_YEARS)
    if isinstance(value, str):
        cleaned = value.replace(";", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if not parts:
            parts = [p for p in cleaned.split() if p]
        return [int(p) for p in parts]
    years = [int(v) for v in value]
    return years


def _load_nodal_dir(nc_dir: Path, countries_cfg: Sequence[str]) -> pd.DataFrame:
    if nc_dir is None or not nc_dir.exists():
        return pd.DataFrame(columns=["year", "BF_nodal_EUR_per_year", "file"])

    pattern = re.compile(r"nodal_costs.*?___(\\d{4}).*\\.csv$")
    rows = []
    loc_re = None
    if countries_cfg:
        loc_re = r"^(?:" + "|".join([re.escape(c) for c in countries_cfg]) + ")"

    for p in nc_dir.iterdir():
        if not p.is_file():
            continue
        match = pattern.search(p.name)
        if not match:
            continue
        year = int(match.group(1))
        try:
            df_nc = pd.read_csv(p)
        except Exception:
            try:
                df_nc = pd.read_csv(
                    p, header=None, names=["cost", "component", "location", "carrier", "value"]
                )
            except Exception as exc:
                logger.warning("Could not read %s: %s", p, exc)
                continue
        if "value" not in df_nc.columns:
            value_col = df_nc.columns[-1]
            df_nc = df_nc.rename(columns={value_col: "value"})
        df_nc["value"] = pd.to_numeric(df_nc["value"], errors="coerce")
        mask = (
            (df_nc["cost"] == "capital")
            & (df_nc["component"] == "Link")
            & (df_nc["carrier"].str.contains("BF-BO", na=False))
        )
        if loc_re:
            mask &= df_nc["location"].str.contains(loc_re, regex=True, na=False)
        val = float(df_nc.loc[mask, "value"].sum()) if mask.any() else np.nan
        rows.append({"year": year, "BF_nodal_EUR_per_year": val, "file": str(p)})

    if not rows:
        return pd.DataFrame(columns=["year", "BF_nodal_EUR_per_year", "file"])
    return pd.DataFrame(rows).drop_duplicates("year").sort_values("year")


def _get_bf_for_years(
    nc_dir: Path, years: Sequence[int], countries_cfg: Sequence[str]
) -> pd.DataFrame:
    df_all = _load_nodal_dir(nc_dir, countries_cfg)
    if df_all.empty:
        return pd.DataFrame(
            {"year": years, "BF_nodal_EUR_per_year": [np.nan] * len(years), "file": [None] * len(years)}
        )
    return pd.DataFrame({"year": years}).merge(
        df_all[["year", "BF_nodal_EUR_per_year", "file"]], on="year", how="left"
    )


def _collect_difference_data(
    results_root: Path,
    scenario_names: Sequence[str] | None,
    comp_df: pd.DataFrame,
    marker_years: Sequence[int],
    countries_cfg: Sequence[str],
) -> pd.DataFrame:
    frames = []
    for scenario_name, nc_dir in _list_scenarios(results_root, scenario_names):
        if "year" not in comp_df.columns:
            raise KeyError("year")
        df_sc = _get_bf_for_years(nc_dir, marker_years, countries_cfg)
        merged = pd.merge(df_sc, comp_df, on="year", how="left")
        merged["Difference_EUR_per_year"] = (
            merged["BF_nodal_EUR_per_year"] - merged["Computed_EUR_per_year"]
        )
        merged["scenario"] = scenario_name
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=[
                "year",
                "BF_nodal_EUR_per_year",
                "Computed_EUR_per_year",
                "Difference_EUR_per_year",
                "scenario",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _plot_differences(
    data: pd.DataFrame,
    scenario_order: Sequence[str],
    marker_years: Sequence[int],
    plotting: dict,
    output_path: Path,
) -> None:
    if data.empty:
        logger.warning("No scenario data available; skipping plot.")
        return

    scale = plotting.get("costs_scale", 1e9)
    unit_label = plotting.get("costs_unit", "bn EUR/a" if scale == 1e9 else "EUR/a")
    scenario_nice = plotting.get("scenario_nice_names", {}) or {}
    scenario_colors = plotting.get("scenario_colors", {}) or {}

    fig, ax = plt.subplots(figsize=(11, 6))
    has_points = False
    order = list(scenario_order) if scenario_order else list(dict.fromkeys(data["scenario"]))
    for scenario in order:
        subset = data[data["scenario"] == scenario].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("year")
        if np.isfinite(subset["Difference_EUR_per_year"]).any():
            has_points = True
        label = scenario_nice.get(scenario, scenario)
        color = scenario_colors.get(scenario)
        ax.plot(
            subset["year"],
            subset["Difference_EUR_per_year"] / scale,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )

    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Difference ({unit_label})")
    ax.set_xticks(marker_years)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved capital cost difference plot to %s", output_path)
    if not has_points:
        logger.warning("All scenario differences are NaN; check nodal costs and carrier filters.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_capital_cost_scenarios",
            configfiles=["config/config.steel.yaml"],
            run="regional_steel_demand_39",
            sector_opts="",
            opts="",
            planning_horizons="2020",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    project_root = Path(__file__).resolve().parents[1]
    plotting_default = _load_plotting_default(project_root)
    plotting_from_params = _params_get(snakemake.params, "plotting", None)
    if callable(plotting_from_params):
        wc = dict(getattr(snakemake, "wildcards", {}))
        plotting_user = plotting_from_params(wc)
    elif plotting_from_params:
        plotting_user = plotting_from_params
    else:
        plotting_user = snakemake.config.get("plotting", {}) or {}
    plotting = _deep_update(plotting_default, plotting_user or {})

    results_root_param = _params_get(snakemake.params, "results_root", "results")
    results_root = _resolve_results_root(results_root_param)
    show_scenarios = _normalize_scenario_names(_params_get(snakemake.params, "scenario_names"))
    marker_years = _normalize_marker_years(_params_get(snakemake.params, "marker_years"))
    config_horizons = snakemake.config.get("scenario", {}).get("planning_horizons")
    if config_horizons:
        marker_years = _normalize_marker_years(config_horizons)

    costs_path = project_root / "resources" / "regional_steel_demand_39" / "costs_2020.csv"
    steel_path = project_root / "resources" / "regional_steel_demand_39" / "steel_existing_plants_base_s_39.csv"
    config_path = project_root / "config" / "config.default.yaml"

    costs = pd.read_csv(costs_path)
    steel_plants = pd.read_csv(steel_path)
    discount_rate, countries_cfg = _load_discount_rate(config_path)

    years = range(2020, 2051)
    res_df = _compute_annuitized_costs(costs, steel_plants, discount_rate, years)
    comp_df = res_df[res_df["year"].isin(marker_years)][
        ["year", "annualized_EUR_per_year"]
    ].copy()
    comp_df = comp_df.rename(columns={"annualized_EUR_per_year": "Computed_EUR_per_year"})

    data = _collect_difference_data(
        results_root=results_root,
        scenario_names=show_scenarios,
        comp_df=comp_df,
        marker_years=marker_years,
        countries_cfg=countries_cfg,
    )

    available = list(dict.fromkeys(data["scenario"]))
    scenario_order = [s for s in (show_scenarios or available) if s in available]

    thesis_dir = results_root / "thesis_plots"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    output_path = thesis_dir / Path(snakemake.output[0]).name

    annuity_path = thesis_dir / "capital_cost_annuity.csv"
    comp_df.to_csv(annuity_path, index=False)
    logger.info("Saved annuitized capital costs to %s", annuity_path)
    diff_path = thesis_dir / "capital_cost_difference.csv"
    data.to_csv(diff_path, index=False)
    logger.info("Saved scenario differences to %s", diff_path)

    _plot_differences(data, scenario_order, marker_years, plotting, output_path)
