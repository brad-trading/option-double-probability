# option-double-probability
Strazza-inspired way to mathemtically check the probability of an option doubling at any point before expiry using monte-carlo simulation. Requires Interactive Brokers

# Option Doubling Probability (ODPS)

Monte Carlo engine with **Gamma-aware intrastep detection** to estimate the probability that an option's price **doubles before expiry**.

- ✅ Runs offline (no broker) by default with a realistic mock vol surface.
- ✅ Optional **Interactive Brokers Client Portal** integration via `IB_GATEWAY_URL`.
- ✅ CLI and FastAPI server.
- ✅ Accuracy-focused: Δ–Γ intrastep crossing, Wilson CI, fine time grid.

> Use at your own risk. This is **not financial advice**. Carefully validate before risking capital.

---

## Quickstart (offline, zero-config)

```bash
git clone <your-repo-url> option-doubling
cd option-doubling
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt```

# Example: 25k sims, 30-minute steps, mock data
```python cli.py --symbol AAPL --expiry 2025-12-19 --option-type call --strike 200 --sims 25000```

You’ll see a table with:

Doubling probability + Wilson 95% CI

Paths doubled / total

Avg time to double (conditional)

Percentiles of max option value

Enable IB (optional)

Copy .env.example to .env and set IB_GATEWAY_URL (e.g., https://localhost:5000/v1/api).

Ensure IB Client Portal Gateway is running and you’re authenticated.

Then:

```python cli.py --symbol AAPL --expiry 2025-12-19 --option-type call --delta 0.25 --live```


The system will:

Resolve contract, build vol surface from the chain (when available),

Fetch current option price (mapping numeric fields → friendly names),

Run the simulation with market vols.

If IB is unreachable, it falls back to offline mode.

API server (optional)
```uvicorn api.main:app --host 0.0.0.0 --port 8000```
# Swagger UI at http://localhost:8000/docs

Notes on accuracy

Intrastep detection: solves a quadratic for α∈(0,1) in V_t + Δ·(αΔS) + ½Γ·(αΔS)^2 = target.

Fine grid: default 48 steps/day (30-minute), configurable.

Expiry parsing: YYYY-MM-DD, integer days, or decimal years. Decimal < 3 ⇒ years; otherwise days/365.25.

Wilson CI: robust CI for the doubling probability.

N≥1 guard** and ceiling batch math prevent discretization/logic errors.

Safety

This is research software. Validate against known cases, test with small sizes, and verify live connectivity and price consistency. You are responsible for any trading decisions.


---
