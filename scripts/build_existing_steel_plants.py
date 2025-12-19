#!/usr/bin/env python3
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
Extract and aggregate existing steel plant capacities and build years from the
Global Steel Plant Tracker (GEM) and optional manual sources, mapping them to model nodes (buses).

Description
-----------
This script processes steel plant data for model initialization:

1. Reads GEM Excel file (sheet "Steel Plants").
2. Optionally appends manually curated EAF plants (<300 kt/a) from CSV.
3. Filters to European region (default) unless overridden.
4. Parses coordinates and creates geometry for spatial join.
5. Loads model regions and maps plants to regions (buses) via spatial join.
6. Filters plants to those existing before the base year and not fully retired before base year.
7. Classifies plants into three processes:
   - Electric Arc Furnace (EAF)
   - DRI + EAF (direct reduced iron + EAF hybrid)
   - Integrated steelworks (BF/BOF route)
8. For each process:
   - Filters by operating status.
   - For integrated steelworks, deduplicates by Plant ID and status, keeping latest retired year.
   - Selects and sums nominal capacities (kt/a) by bus.
   - Computes capacity-weighted average start year per bus and process, with fallbacks as needed.
   - Stores plant-level details for optional output.
9. Ensures all model regions are present in outputs, filling missing with zeros.
10. Writes outputs:
    - capacities: CSV (index = bus, columns = processes; values = kt/a)
    - start_dates: CSV (index = bus, columns = processes; values = year (int))
    - steel_plants (optional): plant-level CSV (index = bus) with selected columns

Intended downstream usage: feeding `add_existing_baseyear` to initialize existing steel production assets.
"""


import logging
from typing import Iterable

import geopandas as gpd
import pandas as pd
import numpy as np

from scripts._helpers import configure_logging, set_scenario_config

# Manual-add CSV path (relative to project root). Maintainers can update this file
# to include additional European EAF plants (<300 kt/a) that are missing in GEM.

# Option to enable/disable consideration of manual EAF CSV
USE_MANUAL_EAF_CSV = False  # Set to False to ignore manual EAF CSV
MANUAL_EAF_CSV = "saikiran/manual_eaf_steel_plants.csv" # https://www.eurofer.eu/assets/Uploads/Map-20191113_Eurofer_SteelIndustry_Rev3-has-stainless.pdf

logger = logging.getLogger(__name__)

# Column/output order requested: Integrated steelworks, DRI + EAF, EAF
PROCESSES = ["Integrated steelworks", "DRI + EAF", "EAF"]

# Central status list mapping so user can easily adjust later.
# Current requirement: only consider plants that are 'operating'.
# If you later want to broaden, just edit the lists below (e.g. add 'construction').
STEEL_STATUS = {
    "Integrated steelworks": ["operating","operating pre-retirement"],
    "DRI + EAF": ["operating"],
    "EAF": ["operating","operating pre-retirement"],
}

def _parse_year(series: pd.Series) -> pd.Series:
    """Extract 4-digit year from strings like '1998' or '1998-01' or return NA."""
    return (
        series.astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )


def load_regions(path: str) -> gpd.GeoDataFrame:
    regions = gpd.read_file(path).set_index("name")
    if regions.crs is None:
        regions = regions.set_crs("EPSG:4326")
    else:
        regions = regions.to_crs("EPSG:4326")
    return regions


def load_gem(gem_path: str, sheet: str = "Steel Plants", region_filter: str | None = "Europe") -> pd.DataFrame:
    df = pd.read_excel(
        gem_path,
        sheet_name=sheet,
        na_values=["N/A", "unknown", ">0"],
    )
    if region_filter:
        df = df.query("Region == @region_filter")
    # unify column names we rely on
    df.rename(columns={
        "Start date": "Start date",
        "Retired Date": "Retired Date",
        "Idled Date": "Idled Date",
    }, inplace=True)

    # Retirement: if retired date missing use idled date as fallback
    df["Retired Date"] = pd.to_numeric(
        df["Retired Date"].combine_first(df.get("Idled Date")), errors="coerce"
    )
    df["Start date"] = pd.to_numeric(df["Start date"].astype(str).str.split("-").str[0], errors="coerce")

    # Coordinates column "lat, lon"
    latlon = df["Coordinates"].str.split(", ", expand=True)
    latlon = latlon.rename(columns={0: "lat", 1: "lon"})
    geometry = gpd.points_from_xy(latlon["lon"], latlon["lat"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def load_manual_eaf(path: str) -> pd.DataFrame:
    """Load manually curated EAF plants CSV.

    Expected columns:
        plant_id, plant_name, country, process, capacity_ttpA, start_year, latitude, longitude
    NOTE: capacity_ttpA must already be in *thousand tonnes per annum* (ttpa) to
    match GEM's Nominal ... capacity columns. We do NOT convert units here.
    Only rows with process == 'EAF' and capacity_ttpA < 300 are considered.
    If file missing or empty, returns empty DataFrame.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.info("Manual EAF CSV %s not found; skipping manual additions", path)
        return pd.DataFrame(columns=["plant_id","plant_name","country","process","capacity","start_year","geometry"])

    if df.empty:
        logger.info("Manual EAF CSV %s is empty; skipping manual additions", path)
        return pd.DataFrame(columns=["plant_id","plant_name","country","process","capacity","start_year","geometry"])

    # Basic validation
    # Accept legacy column name capacity_tpa (tonnes per annum) and convert to ttpA if present
    if "capacity_ttpA" not in df.columns and "capacity_tpa" in df.columns:
        # convert t/a -> ttpA (divide by 1000)
        df["capacity_ttpA"] = df["capacity_tpa"] / 1000.0
    required = {"plant_id","plant_name","country","process","capacity_ttpA","start_year","latitude","longitude"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Manual EAF CSV %s missing columns %s; skipping manual additions", path, ", ".join(sorted(missing)))
        return pd.DataFrame(columns=["plant_id","plant_name","country","process","capacity","start_year","geometry"])

    # Filter to EAF only (no capacity threshold)
    df = df[df["process"].str.upper() == "EAF"].copy()
    if df.empty:
        logger.info("Manual EAF CSV %s has no EAF rows; nothing to add", path)
        return pd.DataFrame(columns=["plant_id","plant_name","country","process","capacity","start_year","geometry"])

    # Align with GEM naming convention for capacity columns (Nominal EAF steel capacity (ttpa))
    df["Nominal EAF steel capacity (ttpa)"] = df["capacity_ttpA"].astype(float)
    df["process"] = "EAF"  # normalize

    # Build geometry for spatial join with regions
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def map_to_regions(gdf: gpd.GeoDataFrame, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")
    gdf = gdf.rename(columns={"name": "bus"})
    gdf["country"] = gdf.bus.str[:2]
    return gdf

def classify_and_filter(gdf: gpd.GeoDataFrame, baseyear: int = 2022) -> gpd.GeoDataFrame:
    """Filter to plants existing before baseyear (Start <= baseyear-1) and not fully retired before baseyear.

    Retained if:
        Start date <= 2019 (for baseyear=2020) OR unknown start but status suggests existing
        AND (Retired Date is NA or Retired Date >= baseyear)
    """
    start_ok = (gdf["Start date"].isna()) | (gdf["Start date"] <= baseyear - 1)
    not_retired = gdf["Retired Date"].isna() | (gdf["Retired Date"] >= baseyear)
    gdf = gdf[start_ok & not_retired].copy()
    return gdf


def select_capacity_series(plants: pd.DataFrame, process: str) -> pd.Series:
    """Return capacity series (in kt/a) for a process from plant-level GEM columns.

    GEM columns are in thousand tonnes per annum (ttpa). We keep kt/a (numeric) here.
    """
    if process == "EAF":
        return plants["Nominal EAF steel capacity (ttpa)"].rename(process)
    if process == "DRI + EAF":
        return plants["Nominal DRI capacity (ttpa)"].rename(process)
    if process == "Integrated steelworks":
        # Combine BOF + OHF per row; if multiple rows share a Plant ID (e.g. different
        # statuses like operating vs operating pre-retirement) we KEEP ALL rows.
        # Downstream aggregation sums by bus, effectively adding capacities across
        # those rows (updated requirement replacing previous max-per-Plant-ID logic).
        bof_col = "Nominal BOF steel capacity (ttpa)"
        ohf_col = "Nominal OHF steel capacity (ttpa)"
        cap = plants[[c for c in [bof_col, ohf_col] if c in plants.columns]].fillna(0).sum(axis=1)
        return cap.rename(process)
    raise ValueError(f"Unknown process {process}")

def aggregate_by_bus(
    gdf: gpd.GeoDataFrame,
    countries: Iterable[str],
    *,
    # lifetime_years: int = 20,
    fallback_start_year: int = 2010,
    integrated_lookback_years: int = 40,
    integrated_fallback_start_year: int = 2005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate capacities (kt/a) and capacity-weighted start year per bus and process.

    Returns
    -------
    capacities : DataFrame (index=bus, columns=processes)
    start_dates : DataFrame (index=bus, columns=processes, Int64)
    plants_detail : DataFrame (index=bus, columns=plant-level attributes)
    """
    capacities = pd.DataFrame(index=gdf.bus.unique(), columns=PROCESSES, dtype=float)
    start_dates = pd.DataFrame(index=gdf.bus.unique(), columns=PROCESSES, dtype="Int64")
    plant_rows = []

    for process in PROCESSES:
        # Filter by operating status according to central mapping (if column present)
        if "Capacity operating status" in gdf.columns:
            allowed = STEEL_STATUS.get(process, None)
            if allowed is not None:
                gdf_proc = gdf[gdf["Capacity operating status"].isin(allowed)].copy()
            else:
                gdf_proc = gdf.copy()
        else:
            gdf_proc = gdf.copy()

        if gdf_proc.empty:
            capacities.loc[:, process] = 0.0
            start_dates.loc[:, process] = 0
            continue
        # For Integrated steelworks, deduplicate rows that refer to the same
        # physical plant (same Plant ID and same Capacity operating status).
        # Keep the record with the LATEST retired year (higher numeric value).
        if process == "Integrated steelworks" and "Plant ID" in gdf_proc.columns:
            group_cols = ["Plant ID"]
            if "Capacity operating status" in gdf_proc.columns:
                group_cols.append("Capacity operating status")

            # If Nominal BF capacity (ttpa) is NA for a Plant ID, assign Nominal BOF steel capacity (ttpa) to that row
            if "Nominal BF capacity (ttpa)" in gdf_proc.columns and "Nominal BOF steel capacity (ttpa)" in gdf_proc.columns:
                bf_na = gdf_proc["Nominal BF capacity (ttpa)"].isna()
                gdf_proc.loc[bf_na, "Nominal BF capacity (ttpa)"] = gdf_proc.loc[bf_na, "Nominal BOF steel capacity (ttpa)"]
            if "Nominal BF capacity (ttpa)" in gdf_proc.columns:
                group_cols.append("Nominal BF capacity (ttpa)")

            if group_cols:
                before = len(gdf_proc)
                retired = pd.to_numeric(gdf_proc.get("Retired Date", pd.Series(np.nan, index=gdf_proc.index)), errors="coerce")
                retired_filled = retired.fillna(np.NINF)
                gdf_proc = gdf_proc.assign(_ret=retired_filled)
                idx_keep = gdf_proc.groupby(group_cols)["_ret"].idxmax()
                keep_idx = pd.Index(idx_keep.values)
                gdf_proc = gdf_proc.loc[keep_idx].copy()
                gdf_proc = gdf_proc.drop(columns=["_ret"])
                dropped = before - len(gdf_proc)
                if dropped:
                    logger.info("Integrated steelworks: dropped %d duplicate rows keeping latest retired year (grouped by Plant ID, Capacity operating status, Nominal BF/BOF capacity)", int(dropped))
        
        cap_series = select_capacity_series(gdf_proc, process).fillna(0.0)
        positive_idx = cap_series[cap_series > 0].index
        if positive_idx.empty:
            capacities.loc[:, process] = 0.0
            start_dates.loc[:, process] = 0
            continue

        # store plant-level detail before aggregation
        df_plants = pd.DataFrame(index=positive_idx)
        df_plants["plant_id"] = gdf_proc.loc[positive_idx, "Plant ID"].astype(str)
        df_plants["plant_name"] = gdf_proc.loc[positive_idx, "Plant name (English)"].astype(str)
        df_plants["country"] = gdf_proc.loc[positive_idx, "country"].astype(str)
        df_plants["bus"] = gdf_proc.loc[positive_idx, "bus"].astype(str)
        df_plants["process"] = process
        df_plants["capacity"] = cap_series.loc[positive_idx].astype(float)
        df_plants["retired_year"] = _parse_year(gdf_proc.loc[positive_idx, "Retired Date"])
        original_start = _parse_year(gdf_proc.loc[positive_idx, "Start date"])
        if process == "Integrated steelworks":
            # Use only the row's own retired year minus lookback, or fallback if missing
            df_plants["start_year"] = (df_plants["retired_year"] - integrated_lookback_years).where(df_plants["retired_year"].notna(), integrated_fallback_start_year)
        elif process == "DRI + EAF":
            # Always override start year for DRI + EAF with fallback
            df_plants["start_year"] = fallback_start_year
        elif process == "EAF":
            # Always override start year for EAF with fallback
            df_plants["start_year"] = fallback_start_year
        else:
            df_plants["start_year"] = original_start
        # Fallback for any missing start years
        #df_plants["start_year"] = df_plants["start_year"].fillna(fallback_start_year)
        plant_rows.append(df_plants)

        by_bus = cap_series.groupby(gdf_proc.bus).sum()
        capacities.loc[by_bus.index, process] = by_bus

        # capacity-weighted average start year per bus using computed start_year
        weight_df = df_plants[["bus", "capacity", "start_year"]].dropna(subset=["start_year"])
        if weight_df.empty:
            start_dates.loc[by_bus.index, process] = 0
        else:
            wavg = (weight_df.capacity * weight_df.start_year).groupby(weight_df.bus).sum() / weight_df.capacity.groupby(weight_df.bus).sum()
            start_dates.loc[wavg.index, process] = wavg.round().astype("Int64")


    # restrict to given countries order (buses already embed country code)
    capacities = capacities.reindex(sorted(capacities.index))
    start_dates = start_dates.reindex(capacities.index)

    if plant_rows:
        _plants = pd.concat(plant_rows, ignore_index=True)
        # ensure deterministic ordering: alphabetic by bus, then plant name, then process
        sort_cols = [c for c in ["bus", "plant_name", "process"] if c in _plants.columns]
        _plants = _plants.sort_values(sort_cols, na_position="last")
        plants_detail = _plants.set_index("bus").sort_index()
    else:
        plants_detail = pd.DataFrame(
            columns=["plant_id","plant_name","country","process","capacity","start_year","retired_year"],
            index=pd.Index([], name="bus"),
        )

    return capacities, start_dates, plants_detail

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake(
            "build_existing_steel_plants",
            clusters=5,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    baseyear = getattr(snakemake.params, "baseyear", 2020)
    countries = snakemake.params.countries

    regions = load_regions(snakemake.input.regions_onshore)

    # GEM steel plants (Europe filter inside loader)
    gem = load_gem(snakemake.input.gem_gspt)


    # Optionally load manual EAF additions (already filtered to <300 kt/a) and append before mapping.
    if USE_MANUAL_EAF_CSV:
        manual_eaf = load_manual_eaf(MANUAL_EAF_CSV)
        if not manual_eaf.empty:
            # Align schema with GEM for downstream functions: need columns
            # Plant ID, Plant name (English), country, Coordinates, Start date, Retired Date, etc.
            manual_eaf = manual_eaf.rename(columns={
                "plant_id": "Plant ID",
                "plant_name": "Plant name (English)",
                "start_year": "Start date",
            })
            # Construct Coordinates column matching GEM style "lat, lon"
            manual_eaf["Coordinates"] = manual_eaf["latitude"].astype(str) + ", " + manual_eaf["longitude"].astype(str)
            # Add empty retired date columns to satisfy classify_and_filter expectations
            manual_eaf["Retired Date"] = pd.NA
            manual_eaf["Idled Date"] = pd.NA
            # Provide GEM capacity column for EAF
            # Capacity already in thousand tonnes per annum; ensure correct dtype
            manual_eaf["Nominal EAF steel capacity (ttpa)"] = manual_eaf["Nominal EAF steel capacity (ttpa)"].astype(float)
            # Also maintain operating status so they are counted
            manual_eaf["Capacity operating status"] = "operating"
            # Append to GEM dataframe (geometry already EPSG:4326)
            gem = pd.concat([gem, manual_eaf], ignore_index=True)
            # logger.info("Appended %d manual EAF plants (<300 kt/a) to GEM dataset", len(manual_eaf))

    gem = map_to_regions(gem, regions)
    gem = classify_and_filter(gem, baseyear=baseyear)

    # Steel-specific config (supports both top-level `existing_steel_plants` and nested `sector.endo_industry.existing_steel_plants`)
    steel_cfg = (
        snakemake.config.get("sector", {})
        .get("endo_industry", {})
        .get("existing_steel_plants", {})
    ) or snakemake.config.get("existing_steel_plants", {})
    # lifetime_years = int(steel_cfg.get("lifetime_years", 25))
    fallback_start_year = int(steel_cfg.get("fallback_start_year", 2010))
    integrated_lookback_years = int(steel_cfg.get("integrated_lookback_years", 40))
    integrated_fallback_start_year = int(steel_cfg.get("integrated_fallback_start_year", 2005))

    capacities, start_dates, plants_detail = aggregate_by_bus(
        gem,
        countries,
        # lifetime_years=lifetime_years,
        fallback_start_year=fallback_start_year,
        integrated_lookback_years=integrated_lookback_years,
        integrated_fallback_start_year=integrated_fallback_start_year,
    )

    # ensure all model regions present
    capacities = capacities.reindex(regions.index, fill_value=0.0)
    start_dates = start_dates.reindex(regions.index, fill_value=0)
    # fill any remaining NaNs (e.g. buses with capacity but missing start years) with 0
    start_dates = start_dates.fillna(0).astype("Int64")
    
    # If capacity > 0 and start year still 0 (unknown after aggregation), set fallback
    fallback_year = fallback_start_year
    mask_positive = capacities > 0
    mask_missing_year = start_dates.eq(0) | start_dates.isna()
    update_mask = mask_positive & mask_missing_year
    if update_mask.any().any():
        start_dates[update_mask] = fallback_year
    # Save outputs
    capacities.to_csv(snakemake.output.capacities)
    start_dates.to_csv(snakemake.output.start_dates)
    if hasattr(snakemake.output, "steel_plants"):
        plants_detail.to_csv(snakemake.output.steel_plants)
        