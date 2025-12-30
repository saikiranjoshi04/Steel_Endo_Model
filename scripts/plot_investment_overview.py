# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
Plot per-scenario investment overview (capex/opex) for selected technologies.

Currently focuses on steel production technologies and mirrors the scenario +
planning horizon layout from plot_balances_overview.py.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from scripts._helpers import (
    configure_logging,
    mock_snakemake,
    rename_techs,
    set_scenario_config,
)
from scripts.plot_summary import preferred_order

logger = logging.getLogger(__name__)


STEEL_TECHS = [
    "DRI-HBI-HYBRID",
    "DRI-HBI-NG CC",
    "HBI-EAF",
    "BF-BOF",
    "BF-BOF CC",
    "EAF-SCRAP",
]
H2_ELECTROLYSIS_TECHS = ["H2 Electrolysis"]


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


def _list_scenarios(results_root: Path, names: Sequence[str] | None) -> list[Tuple[str, Path]]:
    summaries = []
    target_names = list(names) if names else None
    if target_names:
        iterate = target_names
    else:
        iterate = sorted(entry.name for entry in results_root.iterdir() if entry.is_dir())

    for scenario in iterate:
        csv_path = results_root / scenario / "csvs" / "nodal_costs.csv"
        if csv_path.exists():
            summaries.append((scenario, csv_path))
    return summaries


def _sort_horizons(horizons: Sequence[str]) -> list[str]:
    def key(value: str) -> tuple[float, str]:
        try:
            return float(value), ""
        except (TypeError, ValueError):
            return float("inf"), value

    unique = list(dict.fromkeys(horizons))
    return sorted(unique, key=key)


_norm_re = re.compile(r"[^a-z0-9]+")


def _norm_key(x: Any) -> str:
    if x is None:
        return ""
    return _norm_re.sub("", str(x).strip().lower())


def _has_multiheader(path: Path) -> bool:
    try:
        with path.open("r") as f:
            lines = [next(f, "").strip().lower() for _ in range(3)]
    except OSError:
        return False
    if len(lines) < 3:
        return False
    return "cluster" in lines[0] and "planning_horizon" in lines[2]


def _read_summary_csv(path: Path, index_cols: int) -> pd.DataFrame:
    if _has_multiheader(path):
        return pd.read_csv(path, index_col=list(range(index_cols)), header=[0, 1, 2])
    return pd.read_csv(path, index_col=list(range(index_cols)))


def _stack_summary(df: pd.DataFrame) -> pd.DataFrame:
    index_names = [
        name if name is not None else f"level_{idx}"
        for idx, name in enumerate(df.index.names)
    ]
    df = df.copy()
    df.index.names = index_names
    if isinstance(df.columns, pd.MultiIndex):
        if "planning_horizon" in df.columns.names:
            horizons = df.columns.get_level_values("planning_horizon")
        else:
            horizons = df.columns.get_level_values(-1)
        df.columns = horizons
    else:
        df.columns = [str(c) for c in df.columns]

    melted = df.reset_index().melt(
        id_vars=index_names,
        var_name="planning_horizon",
        value_name="value",
    )
    melted["planning_horizon"] = melted["planning_horizon"].astype(str)
    return melted


def _steel_techs_for_focus(focus: str) -> list[str]:
    if str(focus).strip().lower() == "steel":
        return STEEL_TECHS
    if str(focus).strip().lower() in {"h2_electrolysis", "h2-electrolysis", "h2 electrolysis"}:
        return H2_ELECTROLYSIS_TECHS
    return []


def _prepare_investment_summary(
    costs_path: Path,
    balance_path: Path,
    prices_path: Path,
    scenario_name: str,
    focus: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    techs = _steel_techs_for_focus(focus)
    if not techs:
        empty = pd.DataFrame(
            columns=["planning_horizon", "carrier_plot", "carrier_raw", "cost_kind", "value", "scenario"]
        )
        detail = pd.DataFrame(
            columns=[
                "scenario",
                "planning_horizon",
                "carrier_plot",
                "carrier_raw",
                "cost_component",
                "fuel_carrier",
                "value",
            ]
        )
        return empty, detail

    costs_raw = _read_summary_csv(costs_path, index_cols=4)
    costs = _stack_summary(costs_raw)
    costs["value"] = pd.to_numeric(costs["value"], errors="coerce").fillna(0.0)
    costs["carrier_raw"] = costs["carrier"].astype(str)
    costs["carrier_plot"] = costs["carrier_raw"].map(rename_techs)
    costs = costs[costs["carrier_raw"].isin(techs)]

    if costs.empty:
        empty = pd.DataFrame(
            columns=["planning_horizon", "carrier_plot", "carrier_raw", "cost_kind", "value", "scenario"]
        )
        detail = pd.DataFrame(
            columns=[
                "scenario",
                "planning_horizon",
                "carrier_plot",
                "carrier_raw",
                "cost_component",
                "fuel_carrier",
                "value",
            ]
        )
        return empty, detail

    rep = (
        costs.assign(abs_value=costs["value"].abs())
        .sort_values("abs_value", ascending=False)
        .groupby("carrier_plot", observed=True)
        .head(1)
        .set_index("carrier_plot")["carrier_raw"]
        .to_dict()
    )

    capex = (
        costs[costs["cost"].str.lower() == "capital"]
        .groupby(["planning_horizon", "carrier_plot"], observed=True, as_index=False)["value"]
        .sum()
    )
    capex["cost_kind"] = "capex"

    opex_base = (
        costs[costs["cost"].str.lower() == "marginal"]
        .groupby(["planning_horizon", "carrier_plot"], observed=True, as_index=False)["value"]
        .sum()
    )

    fuel_costs = pd.DataFrame(columns=["planning_horizon", "carrier_plot", "fuel_cost"])
    fuel_costs_split = pd.DataFrame(
        columns=["planning_horizon", "carrier_plot", "fuel_carrier", "fuel_cost"]
    )
    if balance_path.exists() and prices_path.exists():
        balances_raw = _read_summary_csv(balance_path, index_cols=4)
        balances = _stack_summary(balances_raw)
        balances["value"] = pd.to_numeric(balances["value"], errors="coerce").fillna(0.0)

        fuel_use = balances[
            (balances["component"].str.lower() == "link")
            & (balances["carrier"].isin(techs))
            & (balances["value"] < 0)
        ].copy()

        if not fuel_use.empty:
            fuel_use["energy_used"] = -fuel_use["value"]
            fuel_use["carrier_plot"] = fuel_use["carrier"].map(rename_techs)
            fuel_use["bus_carrier_key"] = fuel_use["bus_carrier"].str.lower()
            fuel_use["location"] = fuel_use["location"].astype(str)

            # Exclude HBI and material inputs that are already captured via marginal costs.
            # HBI is an output/product, and iron/scrap steel are accounted for in link marginal costs.
            bus_carrier_lower = fuel_use["bus_carrier"].astype(str).str.lower()
            fuel_use = fuel_use[
                ~bus_carrier_lower.str.contains("hbi")
                & ~bus_carrier_lower.str.contains("iron")
                & ~bus_carrier_lower.str.contains("scrap")
            ].copy()

            prices_raw = _read_summary_csv(prices_path, index_cols=2)
            prices = _stack_summary(prices_raw)
            prices["value"] = pd.to_numeric(prices["value"], errors="coerce").fillna(0.0)
            if "carrier" not in prices.columns:
                index_name = prices.columns[0]
                prices = prices.rename(columns={index_name: "carrier"})
            if "location" not in prices.columns:
                index_name = prices.columns[0]
                prices = prices.rename(columns={index_name: "location"})
            prices["carrier_key"] = prices["carrier"].str.lower()
            prices["location"] = prices["location"].astype(str)
            prices = prices.rename(columns={"value": "price"})

            fuel_use = fuel_use.merge(
                prices[["location", "carrier_key", "planning_horizon", "price"]],
                left_on=["location", "bus_carrier_key", "planning_horizon"],
                right_on=["location", "carrier_key", "planning_horizon"],
                how="left",
            )
            missing_prices = fuel_use["price"].isna().sum()
            if missing_prices:
                logger.warning(
                    "Missing nodal weighted prices for %d fuel entries in %s; treating as zero.",
                    missing_prices,
                    scenario_name,
                )
            fuel_use["price"] = fuel_use["price"].fillna(0.0)
            fuel_use["fuel_cost"] = fuel_use["energy_used"] * fuel_use["price"]

            fuel_costs = (
                fuel_use.groupby(["planning_horizon", "carrier_plot"], observed=True, as_index=False)["fuel_cost"]
                .sum()
            )
            fuel_costs_split = (
                fuel_use.groupby(
                    ["planning_horizon", "carrier_plot", "bus_carrier"],
                    observed=True,
                    as_index=False,
                )["fuel_cost"]
                .sum()
                .rename(columns={"bus_carrier": "fuel_carrier"})
            )

    opex_detail = opex_base.merge(
        fuel_costs,
        on=["planning_horizon", "carrier_plot"],
        how="left",
    )
    opex_detail["fuel_cost"] = opex_detail["fuel_cost"].fillna(0.0)
    opex_detail = opex_detail.rename(columns={"value": "marginal_cost"})
    opex_detail["opex_total"] = opex_detail["marginal_cost"] + opex_detail["fuel_cost"]
    opex_detail["carrier_raw"] = opex_detail["carrier_plot"].map(rep.get)
    opex_detail["scenario"] = scenario_name

    opex = opex_detail.rename(columns={"opex_total": "value"})[
        ["planning_horizon", "carrier_plot", "value"]
    ].copy()
    opex["cost_kind"] = "opex"

    summary = pd.concat([capex, opex], ignore_index=True)
    summary["carrier_raw"] = summary["carrier_plot"].map(rep.get)
    summary["scenario"] = scenario_name
    marginal_detail = opex_base.rename(columns={"value": "value"}).copy()
    marginal_detail["cost_component"] = "marginal"
    marginal_detail["fuel_carrier"] = ""
    fuel_detail = fuel_costs_split.rename(columns={"fuel_cost": "value"}).copy()
    fuel_detail["cost_component"] = "fuel"
    detail = pd.concat([marginal_detail, fuel_detail], ignore_index=True)
    detail["carrier_raw"] = detail["carrier_plot"].map(rep.get)
    detail["scenario"] = scenario_name

    detail = detail[
        ["scenario", "planning_horizon", "carrier_plot", "carrier_raw", "cost_component", "fuel_carrier", "value"]
    ]
    return summary, detail


def _collect_investment_data(
    results_root: Path,
    focus: str,
    scenario_names: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = _list_scenarios(results_root, scenario_names)
    frames = []
    details = []
    for scenario_name, costs_path in scenarios:
        base_dir = costs_path.parent
        balance_path = base_dir / "nodal_energy_balance.csv"
        prices_path = base_dir / "nodal_weighted_prices.csv"
        frame, detail = _prepare_investment_summary(
            costs_path,
            balance_path,
            prices_path,
            scenario_name,
            focus,
        )
        if not frame.empty:
            frames.append(frame)
        if not detail.empty:
            details.append(detail)
    if not frames:
        empty = pd.DataFrame(
            columns=["planning_horizon", "carrier_plot", "carrier_raw", "cost_kind", "value", "scenario"]
        )
        detail = pd.DataFrame(
            columns=[
                "scenario",
                "planning_horizon",
                "carrier_plot",
                "carrier_raw",
                "cost_component",
                "fuel_carrier",
                "value",
            ]
        )
        return empty, detail
    summary = pd.concat(frames, ignore_index=True)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame(
        columns=[
            "scenario",
            "planning_horizon",
            "carrier_plot",
            "carrier_raw",
            "cost_component",
            "fuel_carrier",
            "value",
        ]
    )
    return summary, detail


def _order_scenarios(available: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    if requested:
        return [s for s in requested if s in available]
    return list(available)


def _plot_investment_overview(
    data: pd.DataFrame,
    opex_detail: pd.DataFrame,
    focus: str,
    scenario_order: list[str],
    plotting: dict,
    output_path: Path,
) -> None:
    if data.empty:
        logger.warning("No investment data available for %s; skipping plot.", focus)
        return

    focus_key = str(focus).strip().lower()
    is_steel = focus_key == "steel"
    is_h2 = focus_key in {"h2_electrolysis", "h2-electrolysis", "h2 electrolysis"}
    ordered_horizons = _sort_horizons(data["planning_horizon"].unique())

    unique_carriers = pd.Index(data["carrier_plot"].unique())
    carrier_order_index = preferred_order.intersection(unique_carriers).append(
        unique_carriers.difference(preferred_order)
    )
    carrier_order = list(dict.fromkeys(carrier_order_index))
    if str(focus).strip().lower() == "steel":
        eaf_label = rename_techs("EAF-SCRAP")
        if eaf_label in carrier_order:
            carrier_order = [c for c in carrier_order if c != eaf_label] + [eaf_label]

    rep = (
        data.assign(abs_value=data["value"].abs())
        .sort_values("abs_value", ascending=False)
        .groupby("carrier_plot", observed=True)
        .head(1)
        .set_index("carrier_plot")["carrier_raw"]
        .to_dict()
    )

    tech_colors = plotting.get("tech_colors", {}) or {}
    nice_names = plotting.get("nice_names", {}) or {}
    scenario_nice = plotting.get("scenario_nice_names", {}) or {}

    fallback_color = "#999999"
    nice_names_norm = {_norm_key(k): v for k, v in nice_names.items()}
    tech_colors_norm = {_norm_key(k): v for k, v in tech_colors.items() if _norm_key(k)}

    scale = plotting.get("costs_scale", 1e9)
    unit_label = plotting.get("costs_unit", "bn EUR/a" if scale == 1e9 else "EUR/a")
    threshold = plotting.get("investment_threshold", 0.5)
    if threshold:
        threshold_raw = threshold * scale
        carrier_max = (
            data.groupby("carrier_plot", observed=True)["value"]
            .sum()
            .abs()
        )
        to_drop = [c for c, v in carrier_max.items() if v < threshold_raw]
        if to_drop:
            data = data.loc[~data["carrier_plot"].isin(to_drop)].copy()
            carrier_order = [c for c in carrier_order if c not in to_drop]

    def _pick_color(cp: str | None, raw_key: str | None) -> str | None:
        for candidate in (cp, raw_key, rename_techs(raw_key) if raw_key else None):
            if not candidate:
                continue
            color = tech_colors.get(candidate)
            if color:
                return color
            normalized = _norm_key(candidate)
            if normalized:
                color = tech_colors_norm.get(normalized)
                if color:
                    return color
        return None

    def _pick_color_from_candidates(candidates: Sequence[str]) -> str | None:
        for candidate in candidates:
            if not candidate:
                continue
            color = tech_colors.get(candidate)
            if color:
                return color
            normalized = _norm_key(candidate)
            if normalized:
                color = tech_colors_norm.get(normalized)
                if color:
                    return color
            renamed = rename_techs(candidate)
            if renamed and renamed != candidate:
                color = tech_colors.get(renamed)
                if color:
                    return color
                normalized = _norm_key(renamed)
                if normalized:
                    color = tech_colors_norm.get(normalized)
                    if color:
                        return color
        return None

    n_scenarios = len(scenario_order)
    group_positions = np.arange(len(ordered_horizons))
    bar_width = min(0.8, 0.9 / n_scenarios)
    scenario_offsets = (np.arange(n_scenarios) - (n_scenarios - 1) / 2) * bar_width

    sns.set_theme(style="white")
    nrows = 3 if is_steel else 2
    fig_height = 10 if is_steel else 8
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(max(9, len(ordered_horizons) * 0.9), fig_height),
        sharex=True,
    )

    tick_positions, tick_labels = [], []
    for cost_kind, ax in zip(["capex", "opex"], axes[:2]):
        subset = data.loc[data["cost_kind"] == cost_kind]
        values = (
            subset.groupby(["scenario", "planning_horizon", "carrier_plot"], observed=True, as_index=True)["value"]
            .sum()
        )
        values_lookup = (values / scale).to_dict()

        seen = set()
        bottoms = np.zeros((n_scenarios, len(ordered_horizons)))

        for sidx, scen in enumerate(scenario_order):
            scen_label = scenario_nice.get(scen, scen)
            for hidx, horizon in enumerate(ordered_horizons):
                x = group_positions[hidx] + scenario_offsets[sidx]
                if cost_kind == "capex":
                    tick_positions.append(x)
                    tick_labels.append(scen_label)

                for cp in carrier_order:
                    v = values_lookup.get((scen, horizon, cp), 0.0)
                    if v == 0.0:
                        continue

                    raw_key = rep.get(cp, cp)
                    color = _pick_color(cp, raw_key) or fallback_color

                    label_name = nice_names.get(cp, nice_names.get(raw_key, str(cp)))
                    label = label_name if cp not in seen else None

                    b = bottoms[sidx, hidx]
                    ax.bar(
                        x,
                        v,
                        bottom=b,
                        width=bar_width,
                        color=color,
                        label=label,
                    )
                    bottoms[sidx, hidx] += v

                    if label:
                        seen.add(cp)

        ax.set_ylabel(f"{cost_kind.upper()} [{unit_label}]")
        ax.set_facecolor("white")
        ax.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    # OPEX fuel-type breakdown subplot (steel only)
    if is_steel:
        breakdown_ax = axes[2]
        # Exclude HBI from the OPEX fuel breakdown (HBI is a product, not an input fuel)
        allowed_fuels = {"coal", "electricity", "gas", "hydrogen", "iron", "scrap steel"}
        fuel_color_candidates = {
            "coal": ["coal", "lignite"],
            "gas": ["gas", "methane", "natural gas"],
            "hydrogen": ["H2", "hydrogen"],
            "electricity": ["AC", "electricity", "electricity supply"],
            "iron": ["iron", "iron ore"],
            "scrap steel": ["scrap", "scrap_steel", "scrap steel"],
        }
        if opex_detail.empty:
            breakdown_ax.set_visible(False)
        else:
            detail = opex_detail.copy()
            detail["fuel_key"] = detail["fuel_carrier"].astype(str).str.lower()
            marginal = detail[detail["cost_component"] == "marginal"].copy()
            if not marginal.empty:
                raw_upper = marginal["carrier_raw"].astype(str).str.upper()
                marginal["fuel_type"] = np.select(
                    [
                        raw_upper.str.contains("EAF-SCRAP"),
                        raw_upper.str.contains("HBI"),
                        raw_upper.str.contains("DRI"),
                        raw_upper.str.contains("BF-BOF"),
                    ],
                    ["scrap steel", "iron", "iron", "iron"],
                    default="",
                )
            fuel = detail[detail["cost_component"] == "fuel"].copy()
            if not fuel.empty:
                fuel_map = {
                    "electricity": "electricity",
                    "low voltage": "electricity",
                    "ac": "electricity",
                    "coking coal": "coal",
                    "coal": "coal",
                    "lignite": "coal",
                    "gas": "gas",
                    "methane": "gas",
                    "natural gas": "gas",
                    "h2": "hydrogen",
                    "hydrogen": "hydrogen",
                    "scrap_steel": "scrap steel",
                }
                fuel["fuel_type"] = fuel["fuel_key"].replace(fuel_map)
            breakdown = pd.concat([marginal, fuel], ignore_index=True)
            breakdown = breakdown[breakdown["fuel_type"].isin(allowed_fuels)]

            # Diagnostics: export residuals between total OPEX and breakdown
            try:
                opex_total = data.loc[data["cost_kind"] == "opex"].groupby(
                    ["scenario", "planning_horizon"],
                    observed=True,
                    as_index=True,
                )["value"].sum()
                breakdown_total = breakdown.groupby(
                    ["scenario", "planning_horizon"],
                    observed=True,
                    as_index=True,
                )["value"].sum()
                residual = (opex_total - breakdown_total).rename("residual")
                if (residual.abs() > 0).any():
                    diag = residual.reset_index()
                    diag_path = output_path.parent / f"opex_breakdown_residuals_{focus}.csv"
                    diag.to_csv(diag_path, index=False)
                    logger.warning("OPEX breakdown residuals written to %s", diag_path)

                    diag_detail = detail.copy()
                    diag_detail["fuel_key"] = diag_detail["fuel_carrier"].astype(str).str.lower()
                    diag_marginal = diag_detail[diag_detail["cost_component"] == "marginal"].copy()
                    if not diag_marginal.empty:
                        raw_upper = diag_marginal["carrier_raw"].astype(str).str.upper()
                        diag_marginal["fuel_type"] = np.select(
                            [
                                raw_upper.str.contains("EAF-SCRAP"),
                                raw_upper.str.contains("HBI"),
                                raw_upper.str.contains("DRI"),
                                raw_upper.str.contains("BF-BOF"),
                            ],
                            ["scrap steel", "iron", "iron", "iron"],
                            default="",
                        )
                    diag_fuel = diag_detail[diag_detail["cost_component"] == "fuel"].copy()
                    if not diag_fuel.empty:
                        fuel_map = {
                            "electricity": "electricity",
                            "low voltage": "electricity",
                            "ac": "electricity",
                            "coking coal": "coal",
                            "coal": "coal",
                            "lignite": "coal",
                            "gas": "gas",
                            "methane": "gas",
                            "natural gas": "gas",
                            "h2": "hydrogen",
                            "hydrogen": "hydrogen",
                            "scrap_steel": "scrap steel",
                        }
                        diag_fuel["fuel_type"] = diag_fuel["fuel_key"].replace(fuel_map)

                    missing_marginal = diag_marginal[
                        (diag_marginal["cost_component"] == "marginal")
                        & (~diag_marginal["fuel_type"].isin(allowed_fuels))
                    ]
                    missing_fuel = diag_fuel[
                        (diag_fuel["cost_component"] == "fuel")
                        & (~diag_fuel["fuel_type"].isin(allowed_fuels))
                    ]
                    if not missing_marginal.empty or not missing_fuel.empty:
                        missing_rows = []
                        if not missing_marginal.empty:
                            mm = (
                                missing_marginal.groupby(
                                    ["scenario", "planning_horizon", "carrier_raw"],
                                    observed=True,
                                    as_index=False,
                                )["value"]
                                .sum()
                            )
                            mm["reason"] = "unmapped_marginal_carrier"
                            mm["fuel_carrier"] = ""
                            missing_rows.append(mm.rename(columns={"carrier_raw": "source"}))
                        if not missing_fuel.empty:
                            mf = (
                                missing_fuel.groupby(
                                    ["scenario", "planning_horizon", "fuel_carrier"],
                                    observed=True,
                                    as_index=False,
                                )["value"]
                                .sum()
                            )
                            mf["reason"] = "unmapped_fuel_carrier"
                            mf["source"] = mf["fuel_carrier"]
                            missing_rows.append(mf[["scenario", "planning_horizon", "source", "value", "reason"]])
                        missing = pd.concat(missing_rows, ignore_index=True)
                        missing_path = output_path.parent / f"opex_breakdown_missing_{focus}.csv"
                        missing.to_csv(missing_path, index=False)
                        logger.warning("OPEX breakdown missing items written to %s", missing_path)
            except Exception:
                logger.debug("Could not compute OPEX breakdown residuals", exc_info=True)

            values = (
                breakdown.groupby(
                    ["scenario", "planning_horizon", "fuel_type"],
                    observed=True,
                    as_index=True,
                )["value"]
                .sum()
            )
            values_lookup = (values / scale).to_dict()

            fuel_order = [
                f
                for f in ["coal", "electricity", "gas", "hydrogen",  "iron", "scrap steel"]
                if f in allowed_fuels
            ]

            seen = set()
            bottoms = np.zeros((n_scenarios, len(ordered_horizons)))

            for sidx, scen in enumerate(scenario_order):
                for hidx, horizon in enumerate(ordered_horizons):
                    x = group_positions[hidx] + scenario_offsets[sidx]
                    for fuel_type in fuel_order:
                        v = values_lookup.get((scen, horizon, fuel_type), 0.0)
                        if v == 0.0:
                            continue
                        candidates = fuel_color_candidates.get(fuel_type, [fuel_type])
                        color = _pick_color_from_candidates(candidates) or fallback_color
                        label = fuel_type.lower() if fuel_type not in seen else None
                        b = bottoms[sidx, hidx]
                        breakdown_ax.bar(
                            x,
                            v,
                            bottom=b,
                            width=bar_width,
                            color=color,
                            label=label,
                        )
                        bottoms[sidx, hidx] += v
                        if label:
                            seen.add(fuel_type)

            breakdown_ax.set_ylabel(f"OPEX fuels [{unit_label}]")
            breakdown_ax.set_facecolor("white")
            breakdown_ax.grid(False)
            handles, labels = breakdown_ax.get_legend_handles_labels()
            if handles:
                breakdown_ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                )

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=90, fontsize=7)
    axes[-1].tick_params(axis="x", which="major", pad=6)
    axes[-1].set_xlabel("")

    # Draw framed boxes around each planning-horizon group (publication style)
    try:
        group_width = n_scenarios * bar_width
        pad = 0.06 * group_width
        for ax in axes:
            y0, y1 = ax.get_ylim()
            for hidx, center in enumerate(group_positions):
                left = center - group_width / 2 - pad
                width = group_width + 2 * pad
                rect = Rectangle(
                    (left, y0),
                    width,
                    y1 - y0,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=2,
                )
                ax.add_patch(rect)
            left_edge = group_positions[0] - group_width / 2 - pad
            right_edge = group_positions[-1] + group_width / 2 + pad
            ax.set_xlim(left_edge, right_edge)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
    except Exception:
        logger.debug("Could not draw horizon boxes", exc_info=True)

    # Horizon labels on secondary axis, placed below scenario labels
    try:
        ax2 = axes[-1].twiny()
        ax2.set_xticks(group_positions)
        ax2.set_xticklabels(ordered_horizons)
        ax2.set_xlim(axes[-1].get_xlim())
        ax2.xaxis.set_ticks_position("bottom")
        ax2.spines["bottom"].set_position(("axes", -0.34))
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", pad=8)
        ax2.annotate(
            "",
            xy=(1.02, -0.34),
            xytext=(1.0, -0.34),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
            clip_on=False,
        )
    except Exception:
        logger.debug("Could not draw horizon axis", exc_info=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved investment overview plot to %s", output_path)

    if is_h2:
        efficiency = {}
        for horizon in ordered_horizons:
            costs_path = Path("resources") / "reference" / f"costs_{horizon}.csv"
            if not costs_path.exists():
                efficiency[horizon] = np.nan
                continue
            try:
                costs_df = pd.read_csv(costs_path)
                tech_col = costs_df.get("technology")
                param_col = costs_df.get("parameter")
                value_col = costs_df.get("value")
                if tech_col is None or param_col is None or value_col is None:
                    efficiency[horizon] = np.nan
                    continue
                mask = tech_col.astype(str).str.lower().eq("electrolysis") & param_col.astype(str).str.lower().eq(
                    "efficiency"
                )
                if not mask.any():
                    mask = tech_col.astype(str).str.lower().str.contains("electrolysis") & param_col.astype(
                        str
                    ).str.lower().eq("efficiency")
                if mask.any():
                    efficiency[horizon] = pd.to_numeric(value_col[mask].iloc[0], errors="coerce")
                else:
                    efficiency[horizon] = np.nan
            except Exception:
                efficiency[horizon] = np.nan

        eff_values = [efficiency.get(h, np.nan) for h in ordered_horizons]
        eff_fig, eff_ax = plt.subplots(
            figsize=(max(9, len(ordered_horizons) * 0.9), 3.5),
        )
        eff_ax.plot(
            group_positions,
            eff_values,
            marker="o",
            linewidth=1.6,
            label="electrolysis efficiency",
        )
        eff_ax.set_ylabel("Electrolysis efficiency")
        eff_ax.set_facecolor("white")
        eff_ax.grid(False)
        eff_ax.set_xticks(group_positions)
        eff_ax.set_xticklabels(ordered_horizons)
        handles, labels = eff_ax.get_legend_handles_labels()
        if handles:
            eff_ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
            )
        eff_fig.tight_layout()
        eff_output = output_path.with_name(f"{output_path.stem}_efficiency{output_path.suffix}")
        eff_fig.savefig(eff_output, dpi=150, bbox_inches="tight")
        logger.info("Saved electrolysis efficiency plot to %s", eff_output)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_investment_overview",
            configfiles=["config/config.steel.yaml"],
            # carrier="h2_electrolysis",
            carrier="steel",
            run="regional_steel_demand_39",
            sector_opts="",
            opts="",
            planning_horizons="2020",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    focus_carrier = snakemake.wildcards.carrier
    show_scenarios = _normalize_scenario_names(_params_get(snakemake.params, "scenario_names"))
    results_root_param = _params_get(snakemake.params, "results_root", "results")

    plotting_from_params = _params_get(snakemake.params, "plotting", None)
    if callable(plotting_from_params):
        wc = dict(getattr(snakemake, "wildcards", {}))
        plotting_user = plotting_from_params(wc)
    elif plotting_from_params:
        plotting_user = plotting_from_params
    else:
        plotting_user = snakemake.config.get("plotting", {}) or {}

    project_root = Path(__file__).resolve().parents[1]
    plotting_default = _load_plotting_default(project_root)

    if isinstance(plotting_user, dict):
        plotting_user = plotting_user.copy()
        default_colors = plotting_default.get("tech_colors")
        user_colors = plotting_user.get("tech_colors")
        if isinstance(default_colors, dict) and isinstance(user_colors, dict):
            merged_colors = default_colors.copy()
            merged_colors.update(user_colors)
            plotting_user["tech_colors"] = merged_colors
    else:
        plotting_user = {}

    plotting = _deep_update(plotting_default, plotting_user)

    results_root = _resolve_results_root(results_root_param)
    data, opex_detail = _collect_investment_data(
        results_root=results_root,
        focus=focus_carrier,
        scenario_names=show_scenarios,
    )

    available = list(dict.fromkeys(data["scenario"]))
    scenario_order = _order_scenarios(available, show_scenarios)

    thesis_dir = results_root / "thesis_plots"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    output_path = thesis_dir / Path(snakemake.output[0]).name
    legacy_opex_path = thesis_dir / f"opex_breakdown_{focus_carrier}.csv"

    _plot_investment_overview(
        data,
        opex_detail,
        focus_carrier,
        scenario_order,
        plotting,
        output_path,
    )
    if legacy_opex_path.exists():
        legacy_opex_path.unlink()
        logger.info("Removed legacy OPEX breakdown CSV at %s", legacy_opex_path)
    if not opex_detail.empty:
        for scenario_name, df in opex_detail.groupby("scenario", sort=False):
            scenario_slug = str(scenario_name).replace(" ", "_")
            opex_path = thesis_dir / f"opex_breakdown_{focus_carrier}_{scenario_slug}.csv"
            df.to_csv(opex_path, index=False)
            logger.info("Saved OPEX breakdown CSV to %s", opex_path)
