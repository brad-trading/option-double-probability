#!/usr/bin/env python3
import sys, json
from datetime import date, datetime
import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from odps.config import Settings
from odps.vol_surface import VolatilitySurface, build_mock_surface
from odps.mc import MonteCarlo
from odps.pricing import wilson_ci
from odps.ib import IBClient, find_strike_by_delta
from odps.utils import parse_expiry_to_years

console = Console()

@click.command()
@click.option("--symbol", required=True, help="Underlying symbol (e.g. AAPL)")
@click.option("--expiry", required=True, help="Expiry YYYY-MM-DD or days or years (e.g. 0.25)")
@click.option("--option-type", type=click.Choice(["call","put"]), default="call")
@click.option("--strike", type=float, help="Strike (omit to pick by delta)")
@click.option("--delta", type=float, default=0.25, show_default=True,
              help="Target absolute delta if strike not provided")
@click.option("--sims", type=int, default=25000, show_default=True, help="Number of Monte Carlo paths")
@click.option("--steps-per-day", type=int, default=48, show_default=True, help="Time steps per day")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for reproducibility")
@click.option("--live", is_flag=True, help="Use live IB if IB_GATEWAY_URL is set")
@click.option("--json-out", is_flag=True, help="Print raw JSON result")
def main(symbol, expiry, option_type, strike, delta, sims, steps_per_day, seed, live, json_out):
    """
    Option Doubling Probability — CLI
    """
    load_dotenv()
    cfg = Settings()  # env-first defaults

    # Resolve T (years)
    T = parse_expiry_to_years(expiry)

    # Live IB (optional)
    ib = IBClient(cfg.ib_url, cfg.ib_ca_bundle, cfg.ib_verify_ssl) if (live and cfg.ib_url) else None

    # Get spot, current option price, and vol surface
    if ib:
        contract = ib.get_contract(symbol)
        spot = contract["price"]
        if strike is None:
            strike = find_strike_by_delta(ib, contract["conid"], option_type, target_abs_delta=delta, expiry_years=T)
        opt = ib.get_option_snapshot(contract["conid"], strike, option_type, T)
        current_option_price = opt["last"] or (opt["bid"] + opt["ask"]) / 2.0
        surface = ib.build_surface(contract["conid"], spot)
        vol_surface = VolatilitySurface(surface)
    else:
        # Offline: mock surface + model price
        spot = 100.0
        if strike is None:
            strike = spot * (1 + (0.5 - delta)) if option_type == "call" else spot * (1 - (0.5 + delta))
        vol_surface = VolatilitySurface(build_mock_surface(spot))
        from odps.pricing import bs_price
        current_option_price = bs_price(spot, strike, T, cfg.risk_free, vol_surface.get_vol(strike, T), option_type)

    engine = MonteCarlo(
        num_paths=sims,
        steps_per_day=steps_per_day,
        seed=seed,
    )

    result = engine.doubling_probability(
        S0=spot, K=strike, T=T, r=cfg.risk_free, option_type=option_type,
        current_option_price=current_option_price,
        vol_fn=vol_surface.get_vol
    )

    out = {
        "symbol": symbol,
        "spot": spot,
        "strike": strike,
        "expiry_years": T,
        "option_type": option_type,
        "current_option_price": current_option_price,
        "num_paths": result["total_paths"],
        "doubling_probability": result["probability_double"],
        "wilson_ci_95": wilson_ci(result["probability_double"], result["total_paths"]),
        "paths_doubled": result["paths_doubled"],
        "expected_max_value": result["expected_max_value"],
        "p5_max_value": result["p5_max_value"],
        "p95_max_value": result["p95_max_value"],
        "avg_time_to_double_years": result["avg_time_to_double_years"],
        "seconds": result["seconds"]
    }

    if json_out:
        console.print_json(data=out)
    else:
        table = Table(title="Option Doubling Simulation")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        prob = out["doubling_probability"]
        ci = out["wilson_ci_95"]
        table.add_row("P(Double)", f"{prob:.2%} (95% CI {ci['lower']:.2%}–{ci['upper']:.2%})")
        table.add_row("Paths doubled", f"{out['paths_doubled']:,} / {out['num_paths']:,}")
        if out["avg_time_to_double_years"]:
            table.add_row("Avg time to double (cond.)", f"{out['avg_time_to_double_years']*365.25:.1f} days")
        table.add_row("Expected max value", f"{out['expected_max_value']:.4f}")
        table.add_row("5th–95th pct max value", f"{out['p5_max_value']:.4f} – {out['p95_max_value']:.4f}")
        table.add_row("Spot / Strike / Price", f"{out['spot']:.2f} / {out['strike']:.2f} / {out['current_option_price']:.2f}")
        table.add_row("Steps × Paths", f"{engine.N} × {out['num_paths']:,}")
        table.add_row("Runtime", f"{out['seconds']:.2f} s")

        console.print(table)

if __name__ == "__main__":
    sys.exit(main())
