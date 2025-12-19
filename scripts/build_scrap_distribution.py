import logging
from pathlib import Path
import pandas as pd
from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

def build_scrap_distribution():
    # 1. Read planning horizon
    year = int(snakemake.params.baseyear)
    logger.info(f"Building scrap steel distribution for horizon: {year}")

    # 2. Load node-level steel loads (kt/year)
    node_prod = pd.read_csv(snakemake.input.industrial_production, index_col=0)

    # 2a. Sanity check expected columns
    if "Electric arc" not in node_prod.columns:
        raise KeyError(
            f"'Electric arc' not found in {snakemake.input.industrial_production}. "
            f"Available columns: {list(node_prod.columns)}"
        )

    # 3. Compute scrap availability (kt/a) — EAF assumed 100% scrap
    scrap_by_node = node_prod["Electric arc"].clip(lower=0).rename("scrap_steel")

    logger.info("First five node-level scrap steel (kt):")
    for node, val in scrap_by_node.head(5).items():
        logger.info(f"  {node}: {val:.2f} kt")

    # Ensure output dir exists and save
    out_path = Path(snakemake.output.scrap_steel_distribution)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scrap_by_node.to_csv(out_path)
    # logger.info(f"Saved node-level scrap steel distribution to {out_path}")

    # 5. Log total
    total_kt = float(scrap_by_node.sum())
    logger.info(f"Total scrap steel for {year}: {total_kt:.1f} kt ({total_kt/1000:.3f} Mt)")

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake(
            "build_scrap_steel_distribution",  # match the rule name for clarity
            configfiles=["config/config.steel.yaml"],
            clusters="39",
            opts="",
            sector_opts="",
            planning_horizons="2050",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    build_scrap_distribution()
    



