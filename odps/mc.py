import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict
from .greeks import bs_price, delta as bs_delta, gamma as bs_gamma

@dataclass
class MCResult:
    probability_double: float
    expected_max_value: float
    p5_max_value: float
    p95_max_value: float
    avg_time_to_double_years: float | None
    paths_doubled: int
    total_paths: int
    seconds: float

class MonteCarlo:
    def __init__(self, num_paths: int = 25000, steps_per_day: int = 48, seed: int = 42, batch_size: int = 2048):
        self.num_paths = int(num_paths)
        self.steps_per_day = int(steps_per_day)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

    @property
    def N(self) -> int:
        # at least one step
        return max(1, int(self._T * 365 * self.steps_per_day))

    def doubling_probability(self, S0: float, K: float, T: float, r: float,
                         option_type: str, current_option_price: float,
                         vol_fn: Callable[[float,float], float]) -> Dict:
    t0 = time.time()
    self._T = float(T)
    N = self.N
    dt = self._T / N
    target = 2.0 * float(current_option_price)

    rng = np.random.default_rng(self.seed)
    # batches so we don't blow RAM
    num_batches = (self.num_paths + self.batch_size - 1) // self.batch_size
    paths_remaining = self.num_paths

    all_max = []
    all_doubled = []
    all_ttd = []

    # Precompute time grid & time-to-maturity
    time_grid = np.linspace(0.0, self._T, N + 1)
    Ttm_grid = (self._T - time_grid).clip(min=1e-12)

    # Precompute vols per time only once (σ depends on K, T; not on S)
    sig_t = np.array([max(0.01, float(vol_fn(K, Ttm))) for Ttm in Ttm_grid], dtype=float)

    for _ in range(num_batches):
        bs = min(self.batch_size, paths_remaining)
        paths_remaining -= bs

        # ----- simulate GBM for all paths (vectorized) -----
        S = np.empty((bs, N + 1), dtype=float)
        S[:, 0] = S0
        Z = rng.standard_normal((bs, N))
        # use per-time vol and drift
        drift = (r - 0.5 * sig_t[:-1] ** 2) * dt        # shape (N,)
        vol_step = sig_t[:-1] * np.sqrt(dt)             # shape (N,)
        for t in range(N):
            S[:, t + 1] = S[:, t] * np.exp(drift[t] + vol_step[t] * Z[:, t])

        # ----- price option for every (path, time) (vectorized per time) -----
        V = np.empty_like(S)
        # at expiry:
        if option_type == "call":
            V[:, -1] = (S[:, -1] - K).clip(min=0.0)
        else:
            V[:, -1] = (K - S[:, -1]).clip(min=0.0)

        # for t = N-1 .. 0 (but pricing is stateless; order doesn’t matter)
        from .greeks import bs_price as _bs
        for t in range(N):
            Ttm = Ttm_grid[t]
            sigma = sig_t[t]
            # vectorized BS via broadcasting over S[:,t]
            St = S[:, t]
            if Ttm <= 0:
                V[:, t] = V[:, -1]  # fallback
            else:
                # inline vectorized BS (avoids Python loop)
                sqrtT = np.sqrt(Ttm)
                with np.errstate(divide='ignore', invalid='ignore'):
                    d1 = (np.log(St / K) + (r + 0.5 * sigma**2) * Ttm) / (sigma * sqrtT)
                d2 = d1 - sigma * sqrtT
                from scipy.stats import norm
                if option_type == "call":
                    V[:, t] = St * norm.cdf(d1) - K * np.exp(-r * Ttm) * norm.cdf(d2)
                else:
                    V[:, t] = K * np.exp(-r * Ttm) * norm.cdf(-d2) - St * norm.cdf(-d1)

        # grid max & quick hits
        max_vals = V.max(axis=1)
        doubled = max_vals >= target
        ttd = np.full(bs, np.nan)
        if np.any(doubled):
            # first grid time crossing
            hit_mask = V[:, :-1] < target
            nxt_mask = V[:, 1:] >= target
            cross = hit_mask & nxt_mask
            hit_idx = np.argmax(cross, axis=1)  # returns 0 if all False; guard below
            has_hit = cross.any(axis=1)
            ttd[has_hit] = time_grid[hit_idx[has_hit] + 1]

        # ---- Γ-aware intrastep (only for those not yet marked doubled) ----
        remaining = ~doubled
        if np.any(remaining):
            from .greeks import delta as bs_delta, gamma as bs_gamma
            rem_idx = np.where(remaining)[0]
            # iterate over time once; operate on all remaining paths at that t
            for t in range(N):
                if not rem_idx.size:
                    break
                St = S[rem_idx, t]
                Stp1 = S[rem_idx, t + 1]
                dS = Stp1 - St
                Ttm = Ttm_grid[t]
                if Ttm <= 0:
                    continue
                sigma = sig_t[t]

                # Δ, Γ for all remaining paths at time t (vectorized)
                sqrtT = np.sqrt(Ttm)
                with np.errstate(divide='ignore', invalid='ignore'):
                    d1 = (np.log(St / K) + (r + 0.5 * sigma**2) * Ttm) / (sigma * sqrtT)
                from scipy.stats import norm
                Delta = (norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0)
                # gamma formula vectorized
                pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
                Gamma = pdf_d1 / (St * sigma * sqrtT)

                Vt = V[rem_idx, t]
                a = 0.5 * Gamma * dS * dS
                bq = Delta * dS
                c = Vt - target

                eps = 1e-14
                alpha_hit = np.full(rem_idx.size, np.nan)

                # quadratic case where |a| >= eps
                quad = np.abs(a) >= eps
                if np.any(quad):
                    disc = bq[quad] * bq[quad] - 4 * a[quad] * c[quad]
                    pos = disc >= 0
                    if np.any(pos):
                        sd = np.sqrt(disc[pos])
                        a2 = 2 * a[quad][pos]
                        roots = np.vstack(((-bq[quad][pos] - sd) / a2,
                                           (-bq[quad][pos] + sd) / a2)).T
                        # pick the first root in (0,1)
                        r = np.where((roots > 0) & (roots < 1), roots, np.nan)
                        alpha_sel = np.nanmin(r, axis=1)  # min ignores NaN if one valid
                        alpha_hit[np.where(quad)[0][np.where(pos)[0]]] = alpha_sel

                # linear case where |a| < eps
                lin = ~quad
                if np.any(lin):
                    nz = bq[lin] != 0
                    if np.any(nz):
                        alpha = -c[lin][nz] / bq[lin][nz]
                        good = (alpha > 0) & (alpha < 1)
                        if np.any(good):
                            idx_lin = np.where(lin)[0][np.where(nz)[0][np.where(good)[0]]]
                            alpha_hit[idx_lin] = alpha[good]

                got = ~np.isnan(alpha_hit)
                if np.any(got):
                    # mark these as doubled and assign ttd
                    sel = rem_idx[got]
                    doubled[sel] = True
                    ttd[sel] = time_grid[t] + alpha_hit[got] * dt
                    # update remaining set
                    rem_idx = rem_idx[~got]

        all_max.append(max_vals)
        all_doubled.append(doubled)
        all_ttd.append(ttd)

    # aggregate
    all_max = np.concatenate(all_max)
    all_doubled = np.concatenate(all_doubled)
    all_ttd = np.concatenate(all_ttd)

    paths_doubled = int(np.sum(all_doubled))
    total = int(len(all_doubled))
    p = paths_doubled / total if total > 0 else 0.0
    valid_t = all_ttd[~np.isnan(all_ttd)]
    avg_t = float(np.mean(valid_t)) if valid_t.size > 0 else None

    return {
        "probability_double": p,
        "expected_max_value": float(np.mean(all_max)),
        "p5_max_value": float(np.percentile(all_max, 5)),
        "p95_max_value": float(np.percentile(all_max, 95)),
        "avg_time_to_double_years": avg_t,
        "paths_doubled": paths_doubled,
        "total_paths": total,
        "seconds": time.time() - t0
    }
