import httpx
from dataclasses import dataclass

_FIELD_MAP = {
    "last": "31",
    "bid": "84",
    "ask": "86",
    "volume": "87",
    "implied_volatility": "7283",
    "delta": "7308",
    "gamma": "7309",
    "theta": "7310",
    "vega": "7311"
}

@dataclass
class IBClient:
    base_url: str | None
    ca_bundle: str | None = None

    def _client(self):
        if not self.base_url:
            raise RuntimeError("IB base_url not set")
        verify = True if not self.ca_bundle else self.ca_bundle
        return httpx.Client(base_url=self.base_url, timeout=10.0, verify=verify)

    def _get(self, path: str, params: dict | None = None):
        with self._client() as c:
            r = c.get(path, params=params)
            r.raise_for_status()
            return r.json()

    # --- Public helpers ---

    def get_contract(self, symbol: str) -> dict:
        # resolve conid via secdef search
        data = self._get("/iserver/secdef/search", {"symbol": symbol})
        if not data:
            raise RuntimeError(f"Symbol not found: {symbol}")
        c = data[0]  # better: filter by exchange/type
        # get snapshot for price (using conid from search detail when available)
        conid = c.get("conid") or c.get("contractId") or c.get("conId")
        price = self._spot_from_snapshot(conid) if conid else None
        return {"conid": conid, "price": price or 100.0}

    def _spot_from_snapshot(self, conid: int | str) -> float | None:
        js = self._get("/iserver/marketdata/snapshot", params={"conids": str(conid), "fields": _FIELD_MAP["last"]})
        if js and isinstance(js, list):
            v = js[0].get(_FIELD_MAP["last"])
            try:
                return float(v) if v is not None else None
            except Exception:
                return None
        return None

    def get_option_snapshot(self, underlying_conid: int | str, strike: float, option_type: str, T_years: float) -> dict:
        # In a production system you would resolve the exact option conid (expiry date, right, strike).
        # Here we assume you have it or use a placeholder mapping.
        # For this template, we just return a structure (user should replace with real chain resolution).
        fields = ",".join(_FIELD_MAP.values())
        # WARNING: real implementation must provide option conid; this is a simplified placeholder.
        js = self._get("/iserver/marketdata/snapshot", params={"conids": str(underlying_conid), "fields": fields})
        if not js:
            return {"last": None, "bid": None, "ask": None}
        row = js[0]
        out = {}
        for k, f in _FIELD_MAP.items():
            val = row.get(f)
            try:
                out[k] = float(val) if val is not None else None
            except Exception:
                out[k] = None
        return out

    def build_surface(self, underlying_conid: int, spot: float) -> dict:
        # Example: pull some strikes/expiries and map to IVs; here we return a mocked structure
        # Replace with real parsing of chain endpoints for production trading.
        from .vol_surface import build_mock_surface
        return build_mock_surface(spot)

def find_strike_by_delta(ib: IBClient, underlying_conid: int, option_type: str,
                         target_abs_delta: float, expiry_years: float) -> float:
    """
    Placeholder that returns ATM-ish strike (user should replace with real delta scan).
    """
    spot = ib._spot_from_snapshot(underlying_conid) or 100.0
    return spot  # ATM
