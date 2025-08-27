import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return d1, d2

def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(S-K, 0.0) if option_type=="call" else max(K-S, 0.0)
    d1, d2 = _d1_d2(S,K,T,r,sigma)
    if option_type=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def delta(S, K, T, r, sigma, option_type):
    if T <= 0:
        return 1.0 if (option_type=="call" and S>K) else (-1.0 if (option_type=="put" and S<K) else 0.0)
    d1, _ = _d1_d2(S,K,T,r,sigma)
    return norm.cdf(d1) if option_type=="call" else norm.cdf(d1)-1.0

def gamma(S, K, T, r, sigma):
    if T <= 0 or sigma<=0 or S<=0: return 0.0
    d1, _ = _d1_d2(S,K,T,r,sigma)
    return norm.pdf(d1)/(S*sigma*np.sqrt(T))
