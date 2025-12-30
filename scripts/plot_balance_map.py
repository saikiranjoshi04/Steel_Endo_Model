# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create energy balance maps for the defined carriers.
"""

import cartopy.crs as ccrs
import geopandas as gpd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    rename_techs,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import sanitize_carriers
from scripts.plot_power_network import load_projection

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_balance_map",
            configfiles=["config/config.steel.yaml"],
            clusters="39",
            run="regional_steel_demand_no_h2_network_39",
            # run="reference",
            opts="",
            sector_opts="",
            planning_horizons="2050",
            carrier="H2",
        )


    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)
    sanitize_carriers(n, snakemake.config)
    pypsa.set_option("params.statistics.round", 3)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    config = snakemake.params.plotting
    carrier = snakemake.wildcards.carrier

    # fill empty colors or "" with light grey
    mask = n.carriers.color.isna() | n.carriers.color.eq("")
    n.carriers["color"] = n.carriers.color.mask(mask, "lightgrey")

    # set EU location with location from config
    eu_location = config["eu_node_location"]
    n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]

    # get balance map plotting parameters
    boundaries = config["map"]["boundaries"]
    config = config["balance_map"][carrier]
    conversion = config["unit_conversion"]

    if carrier not in n.buses.carrier.unique():
        raise ValueError(
            f"Carrier {carrier} is not in the network. Remove from configuration `plotting: balance_map: bus_carriers`."
        )

    # for plotting change bus to location
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

    # set location of buses to EU if location is empty and set x and y coordinates to bus location
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)

    # bus_sizes according to energy balance of bus carrier
    eb = n.statistics.energy_balance(bus_carrier=carrier, groupby=["bus", "carrier"])

    # remove energy balance of transmission carriers which relate to losses
    transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
        {"name": "carrier"}
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")

    # only carriers that are also in the energy balance
    carriers_in_eb = carriers[carriers.isin(eb.index.get_level_values("carrier"))]

    eb.loc[components] = eb.loc[components].drop(index=carriers_in_eb, level="carrier")
    eb = eb.dropna()
    bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(conversion)
    bus_sizes = bus_sizes.sort_values(ascending=False)

    # If steel is aggregated to a single bus, allocate production by electricity-use location
    if carrier == "steel":
        steel_buses = bus_sizes.index.get_level_values("bus").unique()
        if len(steel_buses) == 1:
            steel_carriers = bus_sizes.index.get_level_values("carrier").unique()
            production_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
            demand_part = bus_sizes[bus_sizes < 0]
            elec_use = pd.Series(dtype=float)

            # Prefer allocation using AC energy balance (captures electricity use by steel carriers)
            if "AC" in n.buses.carrier.unique():
                ac_balance = n.statistics.energy_balance(
                    bus_carrier="AC", groupby=["bus", "carrier"]
                )
                ac_balance = ac_balance.loc[
                    ac_balance.index.get_level_values("carrier").isin(production_carriers)
                ]
                if not ac_balance.empty:
                    ac_use = ac_balance[ac_balance < 0].abs()
                    elec_use = ac_use.groupby(level=["bus", "carrier"]).sum()

            # Fallback: infer electricity use from link power on AC side
            if elec_use.empty:
                steel_links = n.links[n.links.carrier.isin(production_carriers)]
                if not steel_links.empty:
                    bus0_carrier = n.buses.carrier.reindex(steel_links.bus0)
                    bus1_carrier = n.buses.carrier.reindex(steel_links.bus1)
                    elec_on_bus0 = bus0_carrier.eq("AC")
                    elec_on_bus1 = bus1_carrier.eq("AC")
                    weights = n.snapshot_weightings.generators

                    elec_use_frames = []
                    if elec_on_bus0.any():
                        link_ids = steel_links.index[elec_on_bus0]
                        p0 = n.links_t.p0[link_ids]
                        use = (-p0).clip(lower=0).mul(weights, axis=0).sum()
                        use = use.to_frame("value")
                        use["bus"] = steel_links.loc[link_ids, "bus0"].values
                        use["carrier"] = steel_links.loc[link_ids, "carrier"].values
                        elec_use_frames.append(use)

                    if elec_on_bus1.any():
                        link_ids = steel_links.index[elec_on_bus1]
                        p1 = n.links_t.p1[link_ids]
                        use = (-p1).clip(lower=0).mul(weights, axis=0).sum()
                        use = use.to_frame("value")
                        use["bus"] = steel_links.loc[link_ids, "bus1"].values
                        use["carrier"] = steel_links.loc[link_ids, "carrier"].values
                        elec_use_frames.append(use)

                    if elec_use_frames:
                        elec_use = pd.concat(elec_use_frames, axis=0)
                        elec_use = (
                            elec_use.groupby(["bus", "carrier"])["value"]
                            .sum()
                            .loc[lambda s: s > 0]
                        )

            if not elec_use.empty:
                carrier_totals = elec_use.groupby(level="carrier").sum()
                shares = elec_use.div(carrier_totals, level="carrier")
                steel_totals = bus_sizes.groupby(level="carrier").sum()
                prod_sizes = shares.mul(steel_totals, level="carrier")

                demand_alloc = demand_part
                demand_totals = demand_part.groupby(level="carrier").sum()
                if not demand_totals.empty:
                    bus_totals = elec_use.groupby(level="bus").sum()
                    bus_shares = bus_totals / bus_totals.sum()
                    if not bus_shares.empty:
                        demand_alloc_list = []
                        for carrier_name, total in demand_totals.items():
                            alloc = bus_shares * total
                            alloc.index = pd.MultiIndex.from_product(
                                [bus_shares.index, [carrier_name]],
                                names=["bus", "carrier"],
                            )
                            demand_alloc_list.append(alloc)
                        if demand_alloc_list:
                            demand_alloc = pd.concat(demand_alloc_list)

                bus_sizes = pd.concat([prod_sizes, demand_alloc], axis=0)

    # Get colors for carriers
    n.carriers.update({"color": snakemake.params.plotting["tech_colors"]})
    carrier_colors = n.carriers.color.copy().replace("", "grey")

    colors = (
        bus_sizes.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(carrier_colors)
    )

    # line and links widths according to optimal capacity
    flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(conversion)

    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    # if there are not lines or links for the bus carrier, use fallback for plotting
    fallback = pd.Series()
    line_widths = flow.get("Line", fallback).abs()
    link_widths = flow.get("Link", fallback).abs()

    # define maximal size of buses and branch width
    bus_size_factor = config["bus_factor"]
    branch_width_factor = config["branch_factor"]
    flow_size_factor = config["flow_factor"]

    # get demand-weighted marginal prices per region as colormap
    buses = n.buses.query("carrier in @carrier").index
    demand = (
        n.statistics.energy_balance(
            bus_carrier=carrier,
            aggregate_time=False,
            groupby=["bus", "carrier"],
        )
        .clip(upper=0)
        .abs()
        .groupby("bus")
        .sum()
        .reindex(buses)
        .rename(n.buses.location)
        .T
    )
    price = n.buses_t.marginal_price.reindex(buses, axis=1).rename(
        n.buses.location, axis=1
    )
    weighted_prices = (demand * price).sum() / demand.sum()

    # Add CO2 atmosphere price
    if carrier == "co2 stored":
        emission_price = n.buses_t.marginal_price.T.loc["co2 atmosphere"].values[0]
        weighted_prices = weighted_prices - emission_price

    # if only one price is available, use this price for all regions
    if weighted_prices.size == 1:
        regions["price"] = weighted_prices.values[0]
        shift = round(weighted_prices.values[0] / 20, 0)
    else:
        regions["price"] = weighted_prices.reindex(regions.index).fillna(0)
        shift = 0

    # Hide price shading where there is no (or negligible) production
    prod_by_bus = bus_sizes[bus_sizes > 0].groupby(level="bus").sum()
    prod_by_region = prod_by_bus.rename(n.buses.location).groupby(level=0).sum()
    price_mask_min = config.get("price_mask_min")
    price_mask_share = config.get("price_mask_share")
    prod_by_region = prod_by_region.reindex(regions.index).fillna(0)
    if price_mask_share is not None:
        share = prod_by_region / prod_by_region.sum()
        regions["price"] = regions["price"].where(share >= price_mask_share)
    else:
        if price_mask_min is None:
            price_mask_min = 0
        regions["price"] = regions["price"].where(prod_by_region >= price_mask_min)

    vmin, vmax = regions.price.min() - shift, regions.price.max() + shift
    if config["vmin"] is not None:
        vmin = config["vmin"]
    if config["vmax"] is not None:
        vmax = config["vmax"]

    crs = load_projection(snakemake.params.plotting)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

    n.plot(
        bus_sizes=bus_sizes * bus_size_factor,
        bus_colors=colors,
        bus_split_circles=True,
        line_widths=line_widths * branch_width_factor,
        link_widths=link_widths * branch_width_factor,
        ax=ax,
        margin=0.2,
        geomap_colors={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        boundaries=boundaries,
    )

    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="price",
        cmap=config["cmap"],
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
    )

    # Add gridlines with latitude/longitude labels
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle=":",
    )
    gl.top_labels = False
    gl.left_labels = False
    gl.bottom_labels = True
    gl.right_labels = True
    gl.xpadding = -1
    gl.ypadding = -1
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 5))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 5))

    ax.set_title(carrier)

    # Add colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=config["cmap"], norm=norm)
    price_unit = config["region_unit"]
    cbr = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.95,
        pad=0.02,
        aspect=50,
        orientation="horizontal",
    )
    # smaller colorbar label and tick sizes (local change)
    cbr.set_label(f"Demand-weighted price ({price_unit})", fontsize=8)
    cbr.ax.tick_params(labelsize=10)
    cbr.outline.set_edgecolor("None")

    # add legend
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
        "columnspacing": 1.0,
        "labelspacing": 0.4,
        "handlelength": 1.2,
    }

    pad = 0.12
    n.carriers.loc["", "color"] = "None"

    # Get lists for production and consumption carriers (optionally apply legend threshold)
    legend_min = config.get("legend_min")
    if legend_min is None:
        unit = config.get("unit")
        if unit == "TWh":
            legend_min = 1
        elif unit == "Mt":
            legend_min = 0.1
        else:
            legend_min = 0
    legend_exclude = set(config.get("legend_exclude", []))
    legend_sizes = bus_sizes[bus_sizes.abs() >= legend_min]
    pos_carriers = legend_sizes[legend_sizes > 0].index.unique("carrier")
    neg_carriers = legend_sizes[legend_sizes < 0].index.unique("carrier")
    if legend_exclude:
        pos_carriers = pos_carriers.difference(legend_exclude)
        neg_carriers = neg_carriers.difference(legend_exclude)

    # Determine larger total absolute value for supply and consumption for a carrier if carrier exists as both supply and consumption
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(carrier, sign):
        values = bus_sizes.loc[:, carrier]
        return values[values * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    # Ensure industry loads show under consumption
    industry_carriers = [c for c in supp_carriers if "industry" in c.lower()]
    if industry_carriers:
        supp_carriers = [c for c in supp_carriers if c not in industry_carriers]
        cons_carriers = cons_carriers + [c for c in industry_carriers if c not in cons_carriers]

    total_legend_items = len(supp_carriers) + len(cons_carriers)
    legend_fontsize = 8 #if total_legend_items <= 8 else 8 if total_legend_items <= 16 else 8
    legend_kwargs["fontsize"] = legend_fontsize
    legend_kwargs["title_fontproperties"] = {"weight": "bold", "size": legend_fontsize}

    ncol_supp = 2 if len(supp_carriers) > 4 else 1
    ncol_cons = 2 if len(cons_carriers) > 4 else 1
    if total_legend_items > 12:
        ncol_supp = max(ncol_supp, 2)
        ncol_cons = max(ncol_cons, 2)
    if total_legend_items > 20:
        ncol_supp = max(ncol_supp, 3)
        ncol_cons = max(ncol_cons, 3)
    x_anchor_supp = 0.0
    x_anchor_cons = 0.6 if ncol_supp == 1 else 0.75
    pad_cons = pad
    legend_break_threshold = config.get("legend_break_threshold", 12)
    if carrier == "H2":
        ncol_supp = 1
        ncol_cons = 2
        x_anchor_cons = 0.33
    elif (len(supp_carriers) + len(cons_carriers)) > legend_break_threshold:
        x_anchor_cons = 0.0
        ncol_cons = max(ncol_cons, 2)
        rows_supp = max(1, math.ceil(len(supp_carriers) / ncol_supp))
        pad_cons = pad + 0.07 * rows_supp

    nice_names = n.carriers.nice_name.to_dict()
    steel_label_keys = ("DRI", "HBI", "EAF", "BF-BOF", "steel")

    def _label_for_carrier(name: str) -> str:
        base = nice_names.get(name, name)
        if any(key in name for key in steel_label_keys):
            return rename_techs(base)
        return base

    supp_labels = [_label_for_carrier(c) for c in supp_carriers]
    cons_labels = [_label_for_carrier(c) for c in cons_carriers]

    # Add production carriers
    add_legend_patches(
        ax,
        n.carriers.color[supp_carriers],
        supp_labels,
        legend_kw={
            "bbox_to_anchor": (x_anchor_supp, -pad),
            "ncol": ncol_supp,
            "title": "Production" if carrier == "steel" else ("Production" if carrier == "H2" else "Supply"),
            **legend_kwargs,
        },
    )

    # Add demand carriers
    add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        cons_labels,
        legend_kw={
            "bbox_to_anchor": (x_anchor_cons, -pad_cons),
            "ncol": ncol_cons,
            "title": "Demand" if carrier == "steel" else ("Consumption" if carrier == "H2" else "Consumption"),
            **legend_kwargs,
        },
    )

    # Add bus legend
    legend_bus_sizes = config["bus_sizes"]
    carrier_unit = config["unit"]
    if legend_bus_sizes is not None:
        add_legend_semicircles(
            ax,
            [
                s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_sizes
            ],
            [f"{s} {carrier_unit}" for s in legend_bus_sizes],
            patch_kw={"color": "#666"},
            legend_kw={
                "bbox_to_anchor": (0, 1),
                **legend_kwargs,
            },
        )

    # Add branch legend
    legend_branch_sizes = config["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {carrier_unit}" for s in legend_branch_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
        )

    fig.savefig(
        snakemake.output[0],
        dpi=400,
        bbox_inches="tight",
    )
