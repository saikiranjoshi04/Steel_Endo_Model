# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
Plot per-scenario energy balance stacks (TWh) for one bus carrier (e.g. H2)
over planning horizons, with consistent PyPSA-Eur plotting colors.

Key fix:
- Keep raw carrier names for tech_colors lookup
- Use rename_techs / nice_names only for grouping/labels
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
    """Deep-merge updates into base (recursively)."""
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
    # plotting.default.yaml stores everything under a top-level `plotting:` key.
    # This script expects the already-unwrapped plotting dict (same shape as
    # `snakemake.config["plotting"]`).
    plotting = loaded.get("plotting")
    if isinstance(plotting, dict):
        return plotting
    # Backwards/alternative format: file is already the plotting dict.
    return loaded


def _list_scenarios(results_root: Path, names: Sequence[str] | None) -> list[Tuple[str, Path]]:
    balances = []
    target_names = list(names) if names else None
    if target_names:
        iterate = target_names
    else:
        iterate = sorted(entry.name for entry in results_root.iterdir() if entry.is_dir())

    for scenario in iterate:
        csv_path = results_root / scenario / "csvs" / "energy_balance.csv"
        if csv_path.exists():
            balances.append((scenario, csv_path))
    return balances


def _read_balance_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col=[0, 1, 2], header=[0, 1, 2])
    cols = raw.columns
    if "planning_horizon" in cols.names:
        planning_horizons = cols.get_level_values("planning_horizon")
    else:
        planning_horizons = cols.get_level_values(-1)
    raw.columns = planning_horizons

    melted = raw.reset_index().melt(
        id_vars=["component", "carrier", "bus_carrier"],
        var_name="planning_horizon",
        value_name="value",
    )
    melted["planning_horizon"] = melted["planning_horizon"].astype(str)
    return melted


def _sort_horizons(horizons: Sequence[str]) -> list[str]:
    def key(value: str) -> tuple[float, str]:
        try:
            return float(value), ""
        except (TypeError, ValueError):
            return float("inf"), value

    unique = list(dict.fromkeys(horizons))
    return sorted(unique, key=key)


def _summarize_values_by_kind(values: pd.Series, kind: str | None) -> pd.Series:
    kind_value = (kind or "net").lower()
    if kind_value == "supply":
        return values[values > 0]
    if kind_value == "demand":
        return values[values < 0].abs()
    if kind_value == "absolute":
        return values.abs()
    return values


# --- robust normalization for matching keys ---
_norm_re = re.compile(r"[^a-z0-9]+")


def _norm_key(x: Any) -> str:
    if x is None:
        return ""
    return _norm_re.sub("", str(x).strip().lower())


def _prepare_carrier_summary(
    csv_path: Path, bus_carrier: str, scenario_name: str, balance_kind: str
) -> pd.DataFrame:
    # Special case: for the reference scenario and steel carrier, construct balances
    # from `resources/reference/steel_production_base_s_39_<year>.csv` files
    if str(bus_carrier).strip().lower() == "steel" and str(scenario_name).strip().lower() == "reference":
        return _prepare_steel_reference_summary(bus_carrier, scenario_name)

    data = _read_balance_csv(csv_path)
    carrier_mask = data["bus_carrier"].str.lower() == bus_carrier.lower()
    subset = data.loc[carrier_mask]
    if subset.empty:
        return pd.DataFrame(columns=["planning_horizon", "carrier_plot", "carrier_color_key", "value", "scenario"])

    subset = subset.copy()
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce").fillna(0.0) / 1e6  # TWh
    filtered = _summarize_values_by_kind(subset["value"], balance_kind)
    if filtered.empty:
        return pd.DataFrame(columns=["planning_horizon", "carrier_plot", "carrier_color_key", "value", "scenario"])

    subset = subset.loc[filtered.index, ["planning_horizon", "carrier"]].copy()
    subset["value"] = filtered.values

    # Keep raw carrier keys for color lookup and create plotting name
    subset["carrier_raw"] = subset["carrier"].astype(str)
    subset["carrier_plot"] = subset["carrier_raw"].map(rename_techs)

    # For each (horizon, carrier_plot) pick a representative raw carrier
    # (weighted by abs(value) so the dominant contributor defines the color key)
    subset["abs_value"] = subset["value"].abs()
    idx = (
        subset.sort_values("abs_value", ascending=False)
        .groupby(["planning_horizon", "carrier_plot"], observed=True)
        .head(1)
        .set_index(["planning_horizon", "carrier_plot"])["carrier_raw"]
    )

    grouped = (
        subset.groupby(["planning_horizon", "carrier_plot"], observed=True, as_index=False)
        .agg(value=("value", "sum"))
    )

    # add representative raw carrier (for color)
    grouped["carrier_color_key"] = grouped.set_index(["planning_horizon", "carrier_plot"]).index.map(idx.get)

    grouped["scenario"] = scenario_name

    # For steel bus carrier (non-reference scenarios) try to split HBI-EAF into
    # H2-based, NG-based, and NG-CC-based routes using link energy inputs and
    # HBI outputs. This follows the approach used by steel_validation_plots.
    try:
        if str(bus_carrier).strip().lower() == "steel" and str(scenario_name).strip().lower() != "reference":
            value_series = pd.to_numeric(data["value"], errors="coerce").fillna(0.0)
            link_mask = data["component"].str.lower() == "link"
            carrier_lower = data["carrier"].str.lower()
            hybrid_mask = carrier_lower.str.contains("dri-hbi-hybrid")
            ngcc_mask = carrier_lower.str.contains("dri-hbi-ng cc")
            h2_mask = link_mask & hybrid_mask & data["bus_carrier"].str.lower().eq("h2") & value_series.lt(0)
            gas_hybrid_mask = link_mask & hybrid_mask & data["bus_carrier"].str.lower().eq("gas") & value_series.lt(0)
            hbi_hybrid_mask = link_mask & hybrid_mask & data["bus_carrier"].str.lower().eq("hbi") & value_series.gt(0)
            hbi_ngcc_mask = link_mask & ngcc_mask & data["bus_carrier"].str.lower().eq("hbi") & value_series.gt(0)

            h2_energy = value_series.loc[h2_mask].groupby(data.loc[h2_mask, "planning_horizon"]).sum().abs()
            gas_hybrid_energy = value_series.loc[gas_hybrid_mask].groupby(
                data.loc[gas_hybrid_mask, "planning_horizon"]
            ).sum().abs()
            hbi_hybrid_out = value_series.loc[hbi_hybrid_mask].groupby(
                data.loc[hbi_hybrid_mask, "planning_horizon"]
            ).sum()
            hbi_ngcc_out = value_series.loc[hbi_ngcc_mask].groupby(
                data.loc[hbi_ngcc_mask, "planning_horizon"]
            ).sum()

            hydrogen_input = 2.10
            gas_input = 2.78

            split_shares: dict[str, tuple[float, float, float]] = {}
            horizons = set(h2_energy.index) | set(gas_hybrid_energy.index) | set(hbi_hybrid_out.index) | set(
                hbi_ngcc_out.index
            )
            for horizon in horizons:
                h2_in = float(h2_energy.get(horizon, 0.0))
                gas_in = float(gas_hybrid_energy.get(horizon, 0.0))
                hbi_out = float(hbi_hybrid_out.get(horizon, 0.0))
                hbi_ngcc = float(hbi_ngcc_out.get(horizon, 0.0))
                m_h2 = h2_in / hydrogen_input if hydrogen_input else 0.0
                m_ng = gas_in / gas_input if gas_input else 0.0
                m_total = m_h2 + m_ng
                if hbi_out > 0 and m_total > 0:
                    hbi_h2 = hbi_out * (m_h2 / m_total)
                    hbi_ng = hbi_out * (m_ng / m_total)
                else:
                    hbi_h2 = 0.0
                    hbi_ng = 0.0
                total_hbi = hbi_h2 + hbi_ng + hbi_ngcc
                if total_hbi > 0:
                    split_shares[horizon] = (hbi_h2 / total_hbi, hbi_ng / total_hbi, hbi_ngcc / total_hbi)

            # Identify HBI-EAF grouped rows and split them per horizon
            hbi_plot = rename_techs("HBI-EAF")
            new_rows = []
            drop_idx = []
            for idx, row in grouped.iterrows():
                if row["carrier_plot"] == hbi_plot:
                    h = row["planning_horizon"]
                    shares = split_shares.get(h)
                    if shares:
                        frac_h2, frac_ng, frac_ngcc = shares
                        val = row["value"]
                        if frac_h2 > 0:
                            new_rows.append({
                                "scenario": row["scenario"],
                                "planning_horizon": h,
                                "carrier_plot": "HBI-EAF H2-based",
                                "carrier_color_key": "HBI-EAF H2-based",
                                "value": val * frac_h2,
                            })
                        if frac_ng > 0:
                            new_rows.append({
                                "scenario": row["scenario"],
                                "planning_horizon": h,
                                "carrier_plot": "HBI-EAF NG-based",
                                "carrier_color_key": "HBI-EAF NG-based",
                                "value": val * frac_ng,
                            })
                        if frac_ngcc > 0:
                            new_rows.append({
                                "scenario": row["scenario"],
                                "planning_horizon": h,
                                "carrier_plot": "DRI-NG-CC-HBI-EAF",
                                "carrier_color_key": "DRI-NG-CC-HBI-EAF",
                                "value": val * frac_ngcc,
                            })
                        drop_idx.append(idx)
            if new_rows:
                grouped = pd.concat(
                    [grouped.drop(index=drop_idx), pd.DataFrame(new_rows)],
                    ignore_index=True,
                )
    except Exception:
        logger.debug("Could not split HBI-EAF into NG/H2 by energy inputs", exc_info=True)

    # For steel bus carrier, keep only supply side (clip negatives to zero) mirroring plot_balances behavior
    if str(bus_carrier).strip().lower() == "steel":
        grouped["value"] = grouped["value"].clip(lower=0)
    return grouped[["scenario", "planning_horizon", "carrier_plot", "carrier_color_key", "value"]]


def _prepare_steel_reference_summary(bus_carrier: str, scenario_name: str) -> pd.DataFrame:
    """Construct steel balances (Mt) for planning horizons found in resources/reference.

    Sums primary and secondary steel production across regions for each available year
    file (`steel_production_base_s_39_<year>.csv`) and returns a DataFrame with the same
    columns as `_prepare_carrier_summary` so plotting code can consume it transparently.
    """
    project_root = Path(__file__).resolve().parents[1]
    res_dir = project_root / "resources" / "reference"

    # Prefer industrial production files which contain the requested categories
    files = sorted(res_dir.glob("industrial_production_base_s_39_*.csv"))
    source = "industrial_production"
    if not files:
        # Fallback to steel production files (older format)
        files = sorted(res_dir.glob("steel_production_base_s_39_*.csv"))
        source = "steel_production"

    if not files:
        return pd.DataFrame(columns=["scenario", "planning_horizon", "carrier_plot", "carrier_color_key", "value"])

    rows = []
    # mapping of human-readable reference columns to canonical raw keys
    col_map = {
        "electric arc": "EAF-SCRAP",
        "integrated steelworks": "BF-BOF",
        "dri + electric arc": "HBI-EAF",
    }

    for f in files:
        m = re.search(r"(\d{4})", f.name)
        if not m:
            continue
        year = m.group(1)
        try:
            df = pd.read_csv(f, index_col=0)
        except Exception:
            logger.debug("Could not read production file %s", f, exc_info=True)
            continue

        # If industrial_production files: use the three named columns if present
        if source == "industrial_production":
            requested_cols = ["Electric arc", "Integrated steelworks", "DRI + Electric arc"]
            present = [c for c in requested_cols if c in df.columns]
            if present:
                for col in present:
                    total_col = df[col].sum()
                    value_mt = float(total_col) / 1000.0  # convert kt -> Mt
                    mapped_key = col_map.get(col.strip().lower(), col.strip())
                    rows.append({
                        "planning_horizon": str(year),
                        "carrier_raw": mapped_key,
                        "carrier_plot": rename_techs(mapped_key),
                        "carrier_color_key": mapped_key,
                        "value": value_mt,
                    })
                continue
            # if none of the requested columns are present, fall back to any steel-like columns

        # generic handling (either fallback files or industrial files without explicit cols)
        steel_cols = [c for c in df.columns if "primary" in c.lower() or "secondary" in c.lower() or "steel" in c.lower()]
        if steel_cols:
            for col in steel_cols:
                total_col = df[col].sum()
                value_mt = float(total_col) / 1000.0
                mapped_key = col_map.get(col.strip().lower(), col.strip())
                rows.append({
                    "planning_horizon": str(year),
                    "carrier_raw": mapped_key,
                    "carrier_plot": rename_techs(mapped_key),
                    "carrier_color_key": mapped_key,
                    "value": value_mt,
                })
        else:
            # Fallback: sum all numeric columns and present as aggregated steel
            total = df.select_dtypes(include=[float, int]).sum().sum()
            value_mt = float(total) / 1000.0
            rows.append({"planning_horizon": str(year), "carrier_raw": "steel", "carrier_plot": rename_techs("steel"), "carrier_color_key": "steel", "value": value_mt})

    if not rows:
        return pd.DataFrame(columns=["scenario", "planning_horizon", "carrier_plot", "carrier_color_key", "value"])

    grouped = pd.DataFrame(rows)
    grouped["scenario"] = scenario_name
    # Apply the same steel rules as `plot_balances`: drop negative parts (clip lower to 0)
    grouped.loc[:, "value"] = grouped["value"].clip(lower=0)
    return grouped[["scenario", "planning_horizon", "carrier_plot", "carrier_color_key", "value"]]


def _collect_balance_data(
    results_root: Path,
    bus_carrier: str,
    scenario_names: Sequence[str] | None,
    balance_kind: str,
) -> pd.DataFrame:
    entries = _list_scenarios(results_root, scenario_names)
    if not entries:
        raise FileNotFoundError("No energy_balance.csv files were found for the requested scenarios.")

    frames = []
    for scenario_name, csv_path in entries:
        df = _prepare_carrier_summary(csv_path, bus_carrier, scenario_name, balance_kind)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"No balance data found for bus carrier '{bus_carrier}'.")
    return pd.concat(frames, ignore_index=True)


def _order_scenarios(available: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    ordered, seen = [], set()
    for s in requested or []:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    for s in available:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _plot_balance_overview(
    data: pd.DataFrame,
    bus_carrier: str,
    scenario_order: list[str],
    plotting: dict,
    output_path: Path,
) -> None:
    ordered_horizons = _sort_horizons(data["planning_horizon"].unique())

    unique_carriers = pd.Index(data["carrier_plot"].unique())
    carrier_order_index = preferred_order.intersection(unique_carriers).append(
        unique_carriers.difference(preferred_order)
    )
    carrier_order = list(dict.fromkeys(carrier_order_index))
    if str(bus_carrier).strip().lower() == "steel":
        eaf_label = rename_techs("EAF-SCRAP")
        if eaf_label in carrier_order:
            carrier_order = [c for c in carrier_order if c != eaf_label] + [eaf_label]

    # Build value lookup
    values = (
        data.groupby(["scenario", "planning_horizon", "carrier_plot"], observed=True, as_index=True)["value"]
        .sum()
    )
    values_lookup = values.to_dict()

    # Apply energy threshold (same approach as `plot_balances`) to drop very
    # small carriers so they don't clutter legend or stacks.
    try:
        threshold = plotting.get("energy_threshold", 50.0) / 100.0
        carrier_max = values.abs().groupby(level=2).max()
        to_drop = [c for c, v in carrier_max.items() if v < threshold and c != "steel"]
        if to_drop:
            logger.info("Dropping small carriers below energy_threshold %s: %s", threshold, to_drop)
            data = data.loc[~data["carrier_plot"].isin(to_drop)].copy()
            # Recompute aggregated values and lookup after dropping
            values = (
                data.groupby(["scenario", "planning_horizon", "carrier_plot"], observed=True, as_index=True)["value"]
                .sum()
            )
            values_lookup = values.to_dict()
    except Exception:
        logger.debug("Could not apply energy threshold", exc_info=True)

    # One representative raw key per carrier_plot (across all scenarios/horizons)
    # choose the one with max abs contribution
    rep = (
        data.assign(abs_value=data["value"].abs())
        .sort_values("abs_value", ascending=False)
        .groupby("carrier_plot", observed=True)
        .head(1)
        .set_index("carrier_plot")["carrier_color_key"]
        .to_dict()
    )

    # Use plotting tech_colors and nice_names directly (same as other plotters)
    tech_colors = plotting.get("tech_colors", {}) or {}
    nice_names = plotting.get("nice_names", {}) or {}
    scenario_nice = plotting.get("scenario_nice_names", {}) or {}

    fallback_color = "#999999"
    logger.info("✓ tech_colors loaded with %d entries", len(tech_colors))

    # normalized mapping of nice names for consistent sorting/lookups
    nice_names_norm = {_norm_key(k): v for k, v in nice_names.items()}
    tech_colors_norm = {_norm_key(k): v for k, v in tech_colors.items() if _norm_key(k)}

    # Diagnostics: which plotted carriers still miss a color entry
    missing = []
    # Map renamed group -> list of original tech keys present in tech_colors
    renamed_to_keys: dict[str, list[str]] = {}
    for k in tech_colors.keys():
        rk = rename_techs(k)
        renamed_to_keys.setdefault(rk, []).append(k)

    def _gather_candidates(cp: str | None, raw_key: str | None) -> list[str]:
        candidates: list[str] = []
        if cp not in (None, ""):
            candidates.append(cp)
        if raw_key not in (None, "", cp):
            candidates.append(raw_key)
        candidates.extend(renamed_to_keys.get(cp, []))
        normalized_raw = rename_techs(raw_key) if isinstance(raw_key, str) else None
        if normalized_raw:
            candidates.extend(renamed_to_keys.get(normalized_raw, []))
        return candidates

    def _pick_color(cp: str | None, raw_key: str | None) -> str | None:
        for candidate in _gather_candidates(cp, raw_key):
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

    for cp in carrier_order:
        raw_key = rep.get(cp, cp)
        if _pick_color(cp, raw_key) is None:
            missing.append((cp, raw_key))
    if missing:
        logger.warning("Still missing colors for (carrier_plot -> raw_key) first 30: %s", missing[:30])

    n_scenarios = len(scenario_order)
    group_positions = np.arange(len(ordered_horizons))
    bar_width = min(0.8, 0.9 / n_scenarios)
    offsets = (np.arange(n_scenarios) - (n_scenarios - 1) / 2) * bar_width

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(max(9, len(ordered_horizons) * 0.9), 6))

    seen = set()
    pos_bottoms = np.zeros((n_scenarios, len(ordered_horizons)))
    neg_bottoms = np.zeros((n_scenarios, len(ordered_horizons)))

    tick_positions, tick_labels = [], []

    for sidx, scen in enumerate(scenario_order):
        scen_label = scenario_nice.get(scen, scen)
        for hidx, horizon in enumerate(ordered_horizons):
            x = group_positions[hidx] + offsets[sidx]
            tick_positions.append(x)
            tick_labels.append(scen_label)

            for cp in carrier_order:
                v = values_lookup.get((scen, horizon, cp), 0.0)
                if v == 0.0:
                    continue

                raw_key = rep.get(cp, cp)
                color = _pick_color(cp, raw_key) or fallback_color

                # label: nice_names should map display, otherwise carrier_plot
                label_name = nice_names.get(cp, str(cp))
                label = label_name if cp not in seen else None

                if v >= 0:
                    b = pos_bottoms[sidx, hidx]
                    ax.bar(
                        x,
                        v,
                        bottom=b,
                        width=bar_width,
                        color=color,
                        label=label,
                    )
                    pos_bottoms[sidx, hidx] += v
                else:
                    b = neg_bottoms[sidx, hidx]
                    ax.bar(x, v, bottom=b, width=bar_width, color=color, label=label)
                    neg_bottoms[sidx, hidx] += v

                if label:
                    seen.add(cp)

    # Choose units based on carrier category similar to `plot_balances`
    co2_carriers = {"co2", "co2 stored", "process emissions", "steel process emissions"}
    steel_carriers = {"steel", "hbi", "iron", "scrap_steel"}
    bc_lc = str(bus_carrier).lower()
    if bc_lc in co2_carriers:
        units = "MtCO2"
    elif bc_lc in steel_carriers:
        units = "Mt"
    else:
        units = "TWh"

    ax.set_ylabel(f"{bus_carrier} Balance ({units})")
    # Title removed to avoid redundant labeling across overview plots
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_xlabel("")
    ax.set_facecolor("white")
    ax.grid(False)

    # Annotate positive totals above each stacked positive bar and ensure space
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else abs(ymax) + 1.0
    text_offset = 0.025 * yrange

    # compute the required top for annotations and expand ylim if necessary
    max_annotation_y = ymax
    for sidx in range(n_scenarios):
        for hidx in range(len(ordered_horizons)):
            total_pos = pos_bottoms[sidx, hidx]
            if total_pos > 0:
                annot_y = total_pos + text_offset
                if annot_y > max_annotation_y:
                    max_annotation_y = annot_y

    if max_annotation_y > ymax:
        # add a slightly larger top margin to make room for rotated labels
        ax.set_ylim(ymin, max_annotation_y + 0.06 * yrange)

        # If annotations pushed the ylim, increase figure height and reduce
        # the subplot top so the annotations have space and don't overlap
        # the figure border. Scale height by the relative overflow.
        overflow = max_annotation_y - ymax
        extra_frac = max(0.0, overflow / yrange)
        try:
            # increase figure height up to 50% depending on overflow
            cur_w, cur_h = fig.get_size_inches()
            new_h = cur_h * (1.0 + min(0.5, extra_frac * 1.5))
            fig.set_size_inches(cur_w, new_h)
            # move subplot area down by increasing bottom/top margins
            extra_margin = min(0.30, 0.06 + extra_frac * 0.4)
            # top should be smaller to create space above axes
            new_top = max(0.60, 0.95 - extra_margin)
            fig.subplots_adjust(top=new_top)
        except Exception:
            pass

    # Annotate per-scenario per-horizon positive totals (rotated 90 degrees)
    try:
        prod_totals = values[values > 0].groupby(level=(0, 1)).sum()
        prod_totals_lookup = {k: float(v) for k, v in prod_totals.items()}
    except Exception:
        prod_totals_lookup = {}

    texts: list[plt.Text] = []
    for sidx, scen in enumerate(scenario_order):
        for hidx, horizon in enumerate(ordered_horizons):
            x = group_positions[hidx] + offsets[sidx]
            total_pos = prod_totals_lookup.get((scen, horizon), float(pos_bottoms[sidx, hidx]))
            if total_pos <= 0:
                continue
            # keep label at the computed top of the positive stack
            y = total_pos + text_offset
            if abs(total_pos) >= 10:
                txt = f"{total_pos:.0f}"
            else:
                txt = f"{total_pos:.1f}"
            t = ax.text(
                x,
                y,
                txt,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                fontweight="bold",
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.95},
                zorder=10,
            )
            texts.append(t)

    # Ensure ylim includes highest annotation with extra padding and grow the
    # figure height (box) so labels don't touch the figure border.
    if texts:
        y_positions = [t.get_position()[1] for t in texts]
        highest = max(y_positions)
        cur_ymin, cur_ymax = ax.get_ylim()
        # padding fraction to leave above labels
        pad_fraction = 0.10
        padding = pad_fraction * yrange

        required_top = highest + padding
        if required_top > cur_ymax:
            ax.set_ylim(cur_ymin, required_top)
            # grow figure height proportionally to the overflow so the added
            # space looks natural and labels stay away from the frame
            try:
                overflow = required_top - cur_ymax
                extra_frac = overflow / yrange
                cur_w, cur_h = fig.get_size_inches()
                # increase height by up to 50% depending on overflow
                new_h = cur_h * (1.0 + min(0.6, extra_frac * 1.8))
                fig.set_size_inches(cur_w, new_h)
                # move subplot area down by increasing bottom/top margins
                extra_margin = min(0.30, 0.06 + extra_frac * 0.5)
                new_top = max(0.55, 0.95 - extra_margin)
                fig.subplots_adjust(top=new_top)
            except Exception:
                pass
        else:
            # Ensure a minimum visual margin even if labels already fit
            min_margin = 0.06 * yrange
            if cur_ymax - highest < min_margin:
                ax.set_ylim(cur_ymin, highest + min_margin)
                try:
                    cur_w, cur_h = fig.get_size_inches()
                    fig.set_size_inches(cur_w, cur_h * 1.08)
                    fig.subplots_adjust(top=min(0.9, 0.92))
                except Exception:
                    pass
    # Draw framed boxes around each planning-horizon group (publication style)
    try:
        y0, y1 = ax.get_ylim()
        group_width = n_scenarios * bar_width
        pad = 0.06 * group_width  # horizontal padding so bars don't touch box edges
        for hidx, center in enumerate(group_positions):
            left = center - group_width / 2 - pad
            width = group_width + 2 * pad
            rect = Rectangle(
                (left, y0), width, y1 - y0,
                facecolor="none", edgecolor="black", linewidth=0.9, zorder=2
            )
            ax.add_patch(rect)
        # Set x-limits tightly to the outer edges of the boxes
        left_edge = group_positions[0] - group_width / 2 - pad
        right_edge = group_positions[-1] + group_width / 2 + pad
        ax.set_xlim(left_edge, right_edge)
        # Make axis spines subtly thicker and consistent for publication
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
    except Exception:
        logger.debug("Could not draw horizon boxes", exc_info=True)

    # horizon labels on secondary axis
    ax2 = ax.twiny()
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(ordered_horizons)
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.28))
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", pad=8)
    try:
        ax2.annotate(
            "",
            xy=(1.02, -0.28),
            xytext=(1.0, -0.28),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
            clip_on=False,
        )
    except Exception:
        logger.debug("Could not draw horizon arrow", exc_info=True)

    # Legends split by sign as of the 2050 planning horizon (fallback: overall totals)
    carrier_totals = values.groupby("carrier_plot").sum()
    carrier_totals_2050 = {}
    try:
        if "2050" in data["planning_horizon"].values:
            # sum across scenarios for the 2050 horizon and aggregate per carrier_plot
            vals_2050 = values.groupby(level=("planning_horizon", "carrier_plot")).sum()
            if "2050" in vals_2050.index.get_level_values("planning_horizon"):
                # select 2050 entries and sum over scenarios to get a single total per carrier
                vals_2050_cp = vals_2050.xs("2050", level="planning_horizon")
                carrier_totals_2050 = vals_2050_cp.groupby(level="carrier_plot").sum()
    except Exception:
        logger.debug("Could not compute 2050 totals for legend classification", exc_info=True)

    # Build production/consumption sets using 2050 sign where available, otherwise fall back to overall sign
    prod, cons = set(), set()
    for c, overall in carrier_totals.items():
        v2050 = float(carrier_totals_2050.get(c, overall))
        if v2050 > 0:
            prod.add(c)
        elif v2050 < 0:
            cons.add(c)
        # zero totals are omitted from both sets

    # Ensure steel-related carriers show only production (no consumption entries)
    steel_carriers = {"steel", "hbi", "iron", "scrap_steel"}
    steel_plots = {rename_techs(s) for s in steel_carriers}
    # remove steel-like plots from consumption set
    cons = cons.difference(steel_plots)
    # if steel has positive overall total, ensure it's in production
    for s in steel_plots:
        if s in carrier_totals and float(carrier_totals.get(s, 0.0)) > 0:
            prod.add(s)

    def _legend_handles(keys: Sequence[str]):
        handles = []
        seen = set()
        # For Production (positive carriers) show legend in visual top-to-bottom
        # order so the legend reads the same way as the stacked bars. For this
        # we reverse the carrier_order when handling production carriers.
        use_reversed = keys == prod
        ordered_iter = reversed(carrier_order) if use_reversed else carrier_order
        for cp in ordered_iter:
            if cp in keys and cp not in seen:
                raw_key = rep.get(cp, cp)
                color = _pick_color(cp, raw_key) or fallback_color
                label = nice_names.get(cp, nice_names.get(raw_key, nice_names.get(rename_techs(raw_key), str(cp))))
                handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))
                seen.add(cp)
        # Append any remaining keys (not found in carrier_order) in a stable, readable order
        remaining = [k for k in keys if k not in seen]
        for cp in sorted(remaining, key=lambda x: nice_names_norm.get(_norm_key(x), str(x))):
            raw_key = rep.get(cp, cp)
            color = _pick_color(cp, raw_key) or fallback_color
            label = nice_names.get(cp, nice_names.get(raw_key, nice_names.get(rename_techs(raw_key), str(cp))))
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))
        return handles

    # Place legends outside plot on the right without overlap. Compute
    # vertical anchors based on number of items so legends don't overlap.
    prod_handles = _legend_handles(prod) if prod else []
    cons_handles = _legend_handles(cons) if cons else []
    n_prod = len(prod_handles)
    n_cons = len(cons_handles)

    # approximate height per entry (in axes fraction)
    entry_height = 0.035
    top_anchor = 0.95
    left_anchor = 1.02

    # ensure some room on the right for legends
    try:
        fig.subplots_adjust(right=0.72)
    except Exception:
        pass

    if prod_handles:
        prod_anchor_y = top_anchor
        leg1 = fig.legend(
            handles=prod_handles,
            title="Production",
            loc="upper left",
            bbox_to_anchor=(left_anchor, prod_anchor_y),
            frameon=False,
            labelspacing=0.4,
            handletextpad=0.6,
        )
        ax.add_artist(leg1)

    if cons_handles:
        # place consumption legend below production legend (or near top if no prod)
        cons_anchor_start = top_anchor - (n_prod * entry_height if n_prod else 0.0) - 0.02
        cons_anchor_y = max(0.08, cons_anchor_start - (n_cons * entry_height) / 2)
        leg2 = fig.legend(
            handles=cons_handles,
            title="Consumption",
            loc="upper left",
            bbox_to_anchor=(left_anchor, cons_anchor_y),
            frameon=False,
            labelspacing=0.4,
            handletextpad=0.6,
        )
        ax.add_artist(leg2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved balance overview plot to %s", output_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        
        snakemake = mock_snakemake(
            "plot_balances_overview",
            configfiles=["config/config.steel.yaml"],
            carrier="gas",
            run="reference",
            sector_opts="",
            opts="",
            planning_horizons="2020",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    bus_carrier = snakemake.wildcards.carrier
    show_scenarios = _normalize_scenario_names(_params_get(snakemake.params, "scenario_names"))
    results_root_param = _params_get(snakemake.params, "results_root", "results")
    balance_kind = _params_get(snakemake.params, "balance_kind", "net")

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

    tech_colors = plotting.get("tech_colors", {}) or {}
    nice_names = plotting.get("nice_names", {}) or {}
    globals()["tech_colors"] = tech_colors
    globals()["nice_names"] = nice_names

    logger.info(
        "Plotting loaded: default=%s, user=%s, merged_keys=%d",
        "yes" if plotting_default else "no",
        "yes" if plotting_user else "no",
        len(plotting.keys()) if isinstance(plotting, dict) else 0,
    )

    results_root = _resolve_results_root(results_root_param)
    balances = _collect_balance_data(
        results_root=results_root,
        bus_carrier=bus_carrier,
        scenario_names=show_scenarios,
        balance_kind=balance_kind,
    )

    available = list(dict.fromkeys(balances["scenario"]))
    scenario_order = _order_scenarios(available, show_scenarios)

    # Ensure thesis plots directory exists under results root and use basename of provided output
    thesis_dir = results_root / "thesis_plots"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    output_path = thesis_dir / Path(snakemake.output[0]).name

    _plot_balance_overview(
        balances,
        bus_carrier,
        scenario_order,
        plotting,
        output_path,
    )
