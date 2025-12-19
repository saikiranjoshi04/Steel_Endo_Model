# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build future industrial production per country.

Description
-------

This rule uses the ``industrial_production_per_country.csv`` file and the expected recycling rates to calculate the future production of the industrial sectors.

**St_primary_fraction**
The fraction of steel that is coming from primary production. This is more energy intensive than recycling steel (secondary production).

**DRI_fraction**
The fraction of primary steel that is produced in DRI plants.

**Al_primary_fraction**
The fraction of aluminium that is coming from primary production. This is more energy intensive than recycling aluminium (secondary production).

**HVC_primary_fraction**
The fraction of high value chemicals that are coming from primary production (crude oil or Fischer Tropsch).

**HVC_mechanical_recycling_fraction**
The fraction of high value chemicals that are coming from mechanical recycling.

**HVC_chemical_recycling_fraction**
The fraction of high value chemicals that are coming from chemical recycling.

If not already present, the information is added as new column in the output file.

The unit of the production is kt/a.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.prepare_sector_network import get

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_industrial_production_per_country_tomorrow")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.industry

    investment_year = int(snakemake.wildcards.planning_horizons)

    fn = snakemake.input.industrial_production_per_country
    production = pd.read_csv(fn, index_col=0)

    # --- Steel (legacy logic + EAF-only growth) -----------------------------
    keys = ["Integrated steelworks", "Electric arc"]

    growth = params.get("EAF_only_scrap_growth_factor", 1.0)
    if isinstance(growth, (dict, list)):
        growth = get(growth, investment_year)
    if growth != 1.0:
        eaf_only_mask = production["Integrated steelworks"] <= 0
        production.loc[eaf_only_mask, "Electric arc"] *= float(growth)

    total_steel = production[keys].sum(axis=1)

    st_primary_fraction = get(params["St_primary_fraction"], investment_year)
    dri_flag = params.get("dri_eaf_plants", params.get("dri_eaf_plants", False))
    if dri_flag:
        dri_fraction = get(params["DRI_fraction"], investment_year)
    else:
        dri_fraction = 0.0

    # Generic exception: countries with zero Electric arc (and some Integrated steelworks) keep all primary integrated
    # and skip the DRI split (treated as no DRI/EAF evolution yet)
    no_eaf_countries = production.index[(production["Electric arc"] <= 0) & (production["Integrated steelworks"] > 0)]

    
    int_steel = production["Integrated steelworks"].sum() or 1
    fraction_persistent_primary = st_primary_fraction * total_steel.sum() / int_steel

    if "DRI + Electric arc" in production.columns:
        production.drop(columns=["DRI + Electric arc"], inplace=True)

    if len(no_eaf_countries) > 0:
        # Initialize new column with zeros if not already inserted
        production.insert(2, "DRI + Electric arc", 0.0)
        # Apply split only to countries that DO have some existing Electric arc capacity
        mask_other = ~production.index.isin(no_eaf_countries)
        if mask_other.any():
            dri_other = (
                dri_fraction * fraction_persistent_primary * production.loc[mask_other, "Integrated steelworks"]
            )
            production.loc[mask_other, "DRI + Electric arc"] = dri_other
            not_dri = 1 - dri_fraction
            production.loc[mask_other, "Integrated steelworks"] = (
                not_dri * fraction_persistent_primary * production.loc[mask_other, "Integrated steelworks"]
            )
            production.loc[mask_other, "Electric arc"] = (
                total_steel.loc[mask_other]
                - production.loc[mask_other, "DRI + Electric arc"]
                - production.loc[mask_other, "Integrated steelworks"]
            )
        # Countries in no_eaf_countries retain original 'Integrated steelworks' and 'Electric arc' (0), no DRI portion.
    else:
        # Normal path (all countries)
        dri = (
            dri_fraction * fraction_persistent_primary * production["Integrated steelworks"]
        )
        production.insert(2, "DRI + Electric arc", dri)
        not_dri = 1 - dri_fraction
        production["Integrated steelworks"] = (
            not_dri * fraction_persistent_primary * production["Integrated steelworks"]
        )
        production["Electric arc"] = (
            total_steel
            - production["DRI + Electric arc"]
            - production["Integrated steelworks"]
        )

    keys = ["Aluminium - primary production", "Aluminium - secondary production"]
    total_aluminium = production[keys].sum(axis=1)

    key_pri = "Aluminium - primary production"
    key_sec = "Aluminium - secondary production"

    al_primary_fraction = get(params["Al_primary_fraction"], investment_year)
    fraction_persistent_primary = (
        al_primary_fraction * total_aluminium.sum() / (production[key_pri].sum() or 1)
    )

    production[key_pri] = fraction_persistent_primary * production[key_pri]
    production[key_sec] = total_aluminium - production[key_pri]

    production["HVC (mechanical recycling)"] = (
        get(params["HVC_mechanical_recycling_fraction"], investment_year)
        * production["HVC"]
    )
    production["HVC (chemical recycling)"] = (
        get(params["HVC_chemical_recycling_fraction"], investment_year)
        * production["HVC"]
    )

    production["HVC"] *= get(params["HVC_primary_fraction"], investment_year)

    fn = snakemake.output.industrial_production_per_country_tomorrow
    production.to_csv(fn, float_format="%.2f")
