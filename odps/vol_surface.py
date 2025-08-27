import numpy as np
from scipy.interpolate import griddata, Rbf

class VolatilitySurface:
    """
    Interpolates implied vol σ(K,T).
    - If (K,T) ∉ data: extrapolate flat to bounds.
    """
    def __init__(self, data: dict):
        # data = {"spot": float, "points": [{"K":..., "T":..., "iv":...}, ...]}
        pts = data.get("points", [])
        if not pts:
            # safe fallback
            self._Ks = np.array([data.get("spot", 100.0)])
            self._Ts = np.array([0.25])
            self._Vs = np.array([0.25])
        else:
            self._Ks = np.array([p["K"] for p in pts], dtype=float)
            self._Ts = np.array([p["T"] for p in pts], dtype=float)
            self._Vs = np.array([p["iv"] for p in pts], dtype=float)
        self._Kmin, self._Kmax = float(np.min(self._Ks)), float(np.max(self._Ks))
        self._Tmin, self._Tmax = float(np.min(self._Ts)), float(np.max(self._Ts))
        # RBF for scattered data + flat extrapolation
        self._rbf = Rbf(self._Ks, self._Ts, self._Vs, function="thin_plate", smooth=0.1)

    def get_vol(self, K: float, T: float) -> float:
        # clip for extrapolation flatness
        Kc = float(np.clip(K, self._Kmin, self._Kmax))
        Tc = float(np.clip(T, self._Tmin, self._Tmax))
        v = float(self._rbf(Kc, Tc))
        return float(np.clip(v, 0.01, 5.0))

def build_mock_surface(spot: float) -> dict:
    # Smooth smile + term structure
    Ks = np.linspace(0.7*spot, 1.3*spot, 11)
    Ts = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
    pts = []
    for T in Ts:
        for K in Ks:
            m = np.log(K/spot)
            iv = 0.20 + 0.10*m*m + 0.05*np.sqrt(T)
            pts.append({"K": float(K), "T": float(T), "iv": float(iv)})
    return {"spot": spot, "points": pts}
