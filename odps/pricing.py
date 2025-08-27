import numpy as np
from .greeks import bs_price

def wilson_ci(p: float, n: int, confidence: float = 0.95) -> dict:
    from scipy.stats import norm
    if n <= 0: return {"lower": 0.0, "upper": 0.0, "confidence": confidence}
    z = norm.ppf(1 - (1 - confidence)/2)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    margin = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin), "confidence": confidence}

# alias
bs_price = bs_price
