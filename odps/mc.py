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
        """
        Simulate GBM paths with local vol σ(K,T) for pricing.
        Doubling target = 2 * current_option_price.
        Gamma-aware intrastep detection:
          Solve for α in (0,1): V_t + Δ*(αΔS) + ½Γ*(αΔS)^2 = target
        """
        t0 = time.time()
        self._T = float(T)
        N = self.N
        dt = self._T / N
        target = 2.0 * float(current_option_price)

        rng = np.random.default_rng(self.seed)

        # aggregation holders
        all_max = []
        all_doubled = []
        all_ttd = []

        # ceiling division for batches (no dropped remainder)
        num_batches = (self.num_paths + self.batch_size - 1) // self.batch_size
        paths_remaining = self.num_paths

        for b in range(num_batches):
            bs = min(self.batch_size, paths_remaining)
            paths_remaining -= bs

            # simulate GBM with local vol at spot (for drift we use r)
            S = np.empty((bs, N+1), dtype=float)
            S[:,0] = S0
            Z = rng.standard_normal((bs, N))
            for t in range(N):
                # local vol for spot strike at time remaining
                Ttm = self._T - t*dt
                vol = max(0.01, float(vol_fn(K, Ttm)))
                S[:, t+1] = S[:, t] * np.exp((r - 0.5*vol*vol)*dt + vol*np.sqrt(dt)*Z[:, t])

            # price path → option value path
            doubled = np.zeros(bs, dtype=bool)
            ttd = np.full(bs, np.nan)
            max_vals = np.zeros(bs, dtype=float)
            time_grid = np.linspace(0.0, self._T, N+1)

            for i in range(bs):
                V = np.empty(N+1, dtype=float)
                # Precompute vols along path at the option strike (consistent with surface)
                sig = np.empty(N+1, dtype=float)
                for t in range(N+1):
                    Ttm = self._T - time_grid[t]
                    if Ttm > 0:
                        sig[t] = float(vol_fn(K, Ttm))
                        V[t] = bs_price(S[i,t], K, Ttm, r, sig[t], option_type)
                    else:
                        sig[t] = sig[max(t-1,0)] if t>0 else float(vol_fn(K, 1e-6))
                        V[t] = max(S[i,t]-K, 0.0) if option_type=="call" else max(K-S[i,t], 0.0)

                max_vals[i] = float(np.max(V))
                # quick check (grid hit)
                if max_vals[i] >= target:
                    doubled[i] = True
                    idx = int(np.argmax(V >= target))
                    ttd[i] = time_grid[idx]
                else:
                    # Γ-aware intrastep crossing check
                    for t in range(N):
                        if V[t] >= target:
                            doubled[i] = True
                            ttd[i] = time_grid[t]
                            break
                        if V[t] < target and V[t+1] < target:
                            # only check if move is "toward" target (optional short-circuit)
                            pass
                        # Try to locate an intrastep α
                        Ttm = self._T - time_grid[t]
                        if Ttm <= 0: continue
                        dS = S[i,t+1] - S[i,t]
                        if dS == 0.0: continue
                        Delta = bs_delta(S[i,t], K, Ttm, r, sig[t], option_type)
                        Gamma = bs_gamma(S[i,t], K, Ttm, r, sig[t])

                        a = 0.5 * Gamma * dS * dS
                        bq = Delta * dS
                        c = V[t] - target

                        alpha_hit = None
                        eps = 1e-14
                        if abs(a) < eps:
                            # linear case
                            if bq != 0.0:
                                alpha = -c / bq
                                if 0.0 < alpha < 1.0:
                                    alpha_hit = alpha
                        else:
                            disc = bq*bq - 4*a*c
                            if disc >= 0.0:
                                sqrt_disc = np.sqrt(disc)
                                for alpha in ((-bq - sqrt_disc)/(2*a), (-bq + sqrt_disc)/(2*a)):
                                    if 0.0 < alpha < 1.0:
                                        alpha_hit = float(alpha)
                                        break

                        if alpha_hit is not None:
                            doubled[i] = True
                            ttd[i] = time_grid[t] + alpha_hit*dt
                            break

            all_max.append(max_vals)
            all_doubled.append(doubled)
            all_ttd.append(ttd)

        # aggregate
        all_max = np.concatenate(all_max)
        all_doubled = np.concatenate(all_doubled)
        all_ttd = np.concatenate(all_ttd)

        paths_doubled = int(np.sum(all_doubled))
        total = int(len(all_doubled))
        p = paths_doubled / total if total>0 else 0.0

        valid_t = all_ttd[~np.isnan(all_ttd)]
        avg_t = float(np.mean(valid_t)) if valid_t.size>0 else None

        out = MCResult(
            probability_double=p,
            expected_max_value=float(np.mean(all_max)),
            p5_max_value=float(np.percentile(all_max, 5)),
            p95_max_value=float(np.percentile(all_max, 95)),
            avg_time_to_double_years=avg_t,
            paths_doubled=paths_doubled,
            total_paths=total,
            seconds=time.time()-t0
        )
        return {
            "probability_double": out.probability_double,
            "expected_max_value": out.expected_max_value,
            "p5_max_value": out.p5_max_value,
            "p95_max_value": out.p95_max_value,
            "avg_time_to_double_years": out.avg_time_to_double_years,
            "paths_doubled": out.paths_doubled,
            "total_paths": out.total_paths,
            "seconds": out.seconds
        }
