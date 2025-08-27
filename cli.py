#!/usr/bin/env python3
import sys, json, os, time
from datetime import datetime
import click
from rich.console import Console
from rich.table import Table

from odps.config import Settings
from odps.vol_surface import VolatilitySurface, build_mock_surface
from odps.mc import MonteCarlo
from odps.pricing import wilson_ci, bs_price
from odps.ib import IBClient
from odps.utils import parse_expiry_to_years

console = Console()

def _fmt_small(x: float) -> str:
    return f"{x:.6g}"  # switches to sci-notation when tiny

@click.command()
@click.option("--symbol", required=True, help="Underlying symbol (e.g. AAPL)")
@click.option("--expiry", required=True, help="Expiry YYYY-MM-DD, days, or years (e.g. 2025-12-19 or 0.25)")
@click.option("--option-type", type=click.Choice(["call","put"]), default="call", show_default=True)
@click.option("--strike", type=float, help="Strike (omit to pick by delta)")
@click.option("--delta", "target_delta", type=float, default=0.25, show_default=True,
              help="Target absolute delta if strike not provided")
@click.option("--sims", type=int, default=25000, show_default=True, help="Number of Monte Carlo paths")
@click.option("--steps-per-day", type=int, default=48, show_default=True, help="Time steps per day")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for reproducibility")
@click.option("--live", is_flag=True, help="Use live IB if IB_GATEWAY_URL is set")
@click.option("--json-out", is_flag=True, help="Print raw JSON result")
def main(symbol, expiry, option_type, strike, target_delta, sims, steps_per_day, seed, live, json_out):
    """
    Option Doubling Probability — CLI
    """
    cfg = Settings()  # loads .env automatically (via load_dotenv in Settings)

    # Resolve T (years) and refuse past/tiny expiries
    T = parse_expiry_to_years(expiry)
    if T < 1 / (365.25 * 24):  # < ~1 hour
        raise SystemExit(f"Expiry resolves to ~{T*365.25*24:.2f} hours. Refusing. Check --expiry.")

    # Build IB client if requested & configured
    ib = IBClient(cfg.ib_url, cfg.ib_ca_bundle, cfg.ib_verify_ssl) if (live and cfg.ib_url) else None

    # Preflight IB (optional but friendly)
    if ib:
        try:
            st = ib.auth_status()
            if not (st.get("authenticated") and st.get("connected")):
                raise SystemExit("IB Gateway reachable but not authenticated/connected. Log in to Client Portal Gateway.")
        except Exception as e:
            raise SystemExit(str(e))

    # Get spot, current option price, and vol surface
    if ib:
        # Underlying contract & spot
        contract = ib.get_contract(symbol)
        if not contract.get("price"):
            raise SystemExit("Failed to retrieve live spot from IB. Check market data permissions/gateway status.")
        spot = float(contract["price"])

        # Strike selection (live delta) if not provided
        if strike is None:
            strike = ib.find_strike_by_delta(contract["conid"], option_type, target_abs_delta=target_delta, expiry_years=T)

        # Live option snapshot (includes IV if available) & live surface around ATM
        opt_snap = ib.get_option_snapshot(contract["conid"], strike, option_type, T)
        # pick current option price: last, or mid if available
        last = opt_snap.get("last")
        bid, ask = opt_snap.get("bid"), opt_snap.get("ask")
        if last is None and (bid is not None and ask is not None):
            current_option_price = (bid + ask) / 2.0
        else:
            current_option_price = last if last is not None else 0.0

        # If price is still zero-ish, warn but continue
        if current_option_price <= 0.0:
            console.print("[yellow]Warning:[/] option price is 0. Using 0 will make 'doubling' trivial. Consider a different strike/expiry.")

        surface_data = ib.build_surface(contract["conid"], spot)
        vol_surface = VolatilitySurface(surface_data)

    else:
        # Offline: mock surface + BS price
        spot = 100.0
        if strike is None:
            # simple heuristic around ATM for offline mode
            strike = spot
        vol_surface = VolatilitySurface(build_mock_surface(spot))
        current_option_price = bs_price(spot, strike, T, cfg.risk_free, vol_surface.get_vol(strike, T), option_type)

    # Run MC
    engine = MonteCarlo(num_paths=sims, steps_per_day=steps_per_day, seed=seed)
    result = engine.doubling_probability(
        S0=spot, K=strike, T=T, r=cfg.risk_free, option_type=option_type,
        current_option_price=current_option_price, vol_fn=vol_surface.get_vol
    )

    out = {
        "symbol": symbol,
        "spot": float(spot),
        "strike": float(strike),
        "expiry_years": float(T),
        "option_type": option_type,
        "current_option_price": float(current_option_price),
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
        return

    # Pretty table
    table = Table(title="Option Doubling Simulation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    prob = out["doubling_probability"]
    ci = out["wilson_ci_95"]
    table.add_row("P(Double)", f"{prob:.2%} (95% CI {ci['lower']:.2%}–{ci['upper']:.2%})")
    table.add_row("Paths doubled", f"{out['paths_doubled']:,} / {out['num_paths']:,}")
    if out["avg_time_to_double_years"]:
        table.add_row("Avg time to double (cond.)", f"{out['avg_time_to_double_years']*365.25:.1f} days")
    table.add_row("Expected max value", _fmt_small(out["expected_max_value"]))
    table.add_row("5th–95th pct max value", f"{_fmt_small(out['p5_max_value'])} – {_fmt_small(out['p95_max_value'])}")
    table.add_row("Spot / Strike / Price", f"{out['spot']:.2f} / {out['strike']:.2f} / {_fmt_small(out['current_option_price'])}")
    table.add_row("Steps × Paths", f"{engine.N} × {out['num_paths']:,}")
    table.add_row("Runtime", f"{out['seconds']:.2f} s")

    console.print(table)

if __name__ == "__main__":
    sys.exit(main())

