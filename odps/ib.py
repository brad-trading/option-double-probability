# odps/ib.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import httpx
from datetime import datetime, timezone

# IB market data field map (numeric IDs → friendly names)
_FIELD_MAP = {
    "last": "31",
    "bid": "84",
    "ask": "86",
    "volume": "87",
    "implied_volatility": "7283",
    "delta": "7308",
    "gamma": "7309",
    "theta": "7310",
    "vega": "7311",
}

def _to_years(expiry_yyyymmdd: str) -> float:
    """Convert 'YYYYMMDD' to year fraction from today (UTC)."""
    try:
        d = datetime.strptime(expiry_yyyymmdd, "%Y%m%d").date()
        today = datetime.now(timezone.utc).date()
        days = (d - today).days
        return max(days / 365.25, 1e-6)
    except Exception:
        return 0.25

@dataclass
class IBClient:
    """
    Thin synchronous client for the IB Client Portal API.

    SSL verification behavior (in order of precedence):
      - If ca_bundle is a path, it's passed to httpx verify=ca_bundle.
      - Else if verify_ssl is False, verification is disabled (dev only).
      - Else verify=True (default).
    """
    base_url: Optional[str]
    ca_bundle: Optional[str] = None
    verify_ssl: bool = True

    # ------------- low-level --------------

    def _client(self) -> httpx.Client:
        if not self.base_url:
            raise RuntimeError("IB base_url not set. Define IB_GATEWAY_URL or provide it to IBClient().")
        # httpx accepts bool or path for verify=
        verify = self.ca_bundle if self.ca_bundle else self.verify_ssl
        # trust_env=False to avoid proxies breaking localhost/127.0.0.1
        return httpx.Client(base_url=self.base_url, timeout=10.0, verify=verify, trust_env=False)

    def _get(self, path: str, params: Dict | None = None):
        try:
            with self._client() as c:
                r = c.get(path, params=params)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"IB GET {path} returned {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(
                f"IB Gateway unreachable at {self.base_url}. "
                f"Ensure it is running/authenticated and SSL settings are correct. Underlying error: {e}"
            ) from e

    # ------------- diagnostics ------------

    def auth_status(self) -> Dict:
        """Return authentication/connection status."""
        return self._get("/iserver/auth/status", params={})

    # ------------- core: underlying ----------

    def get_contract(self, symbol: str) -> Dict:
        """
        Look up an underlying contract by symbol and return conid + spot price.
        Chooses the first 'STK' result that matches the symbol (case-insensitive).
        """
        data = self._get("/iserver/secdef/search", {"symbol": symbol})
        if not data or not isinstance(data, list):
            raise RuntimeError(f"No contracts found for symbol '{symbol}'.")
        # Prefer 'STK' secType and exact symbol match, else fall back to first
        candidates = [x for x in data if str(x.get("secType", "")).upper() == "STK"]
        if not candidates:
            candidates = data
        sym_u = symbol.upper()
        # exact symbol match first
        exact = [x for x in candidates if str(x.get("symbol", "")).upper() == sym_u]
        chosen = exact[0] if exact else candidates[0]
        conid = chosen.get("conid") or chosen.get("conId") or chosen.get("contractId")
        if not conid:
            raise RuntimeError(f"Could not resolve conid for symbol '{symbol}'. Response: {chosen}")
        # snapshot for spot
        spot = self._spot_from_snapshot(conid)
        return {"conid": int(conid), "price": float(spot) if spot is not None else None}

    def _spot_from_snapshot(self, conid: int | str) -> Optional[float]:
        js = self._snapshot([conid], [_FIELD_MAP["last"]])
        if not js:
            return None
        try:
            v = js[0].get(_FIELD_MAP["last"])
            return float(v) if v is not None else None
        except Exception:
            return None

    def _snapshot(self, conids: List[int | str], fields: List[str]) -> List[Dict]:
        params = {"conids": ",".join(str(c) for c in conids), "fields": ",".join(fields)}
        js = self._get("/iserver/marketdata/snapshot", params)
        return js if isinstance(js, list) else [js]

    # ------------- options (safe placeholders for now) -------------

    def get_option_snapshot(self, underlying_conid: int | str, strike: float, option_type: str, T_years: float) -> Dict:
        """
        Placeholder: returns a snapshot structure using the underlying snapshot fields.

        NOTE: This does NOT yet resolve the actual option conid. It is intentionally
        conservative so your app runs end-to-end. To wire real option quotes/IV:
          1) Resolve option conid for (underlying, expiry, strike, right)
          2) Call /iserver/marketdata/snapshot on the OPTION conid (fields include 31,84,86,7283,7308,...)
        """
        # Using underlying as a surrogate so you don't get None/KeyErrors downstream.
        fields = list(_FIELD_MAP.values())
        row = self._snapshot([underlying_conid], fields)[0] if underlying_conid else {}
        out = {}
        for k, f in _FIELD_MAP.items():
            val = row.get(f)
            try:
                out[k] = float(val) if val is not None else None
            except Exception:
                out[k] = None
        return out

    def build_surface(self, underlying_conid: int | str, spot: float) -> Dict:
        """
        Placeholder vol surface: returns a smooth mock surface around spot.
        Keeps your pipeline working until we wire the real chain.
        """
        # local import to avoid circular
        from .vol_surface import build_mock_surface
        return build_mock_surface(float(spot) if spot is not None else 100.0)

    def find_strike_by_delta(
        self,
        underlying_conid: int | str,
        option_type: str,
        target_abs_delta: float = 0.25,
        expiry_years: float | None = None,
    ) -> float:
        """
        Placeholder delta selection: returns ATM strike (≈ spot).
        To implement real delta targeting we must:
          - choose an expiry (closest to requested)
          - build a mini-chain of conids around ATM
          - snapshot Greeks and select |Δ| closest to target_abs_delta
        """
        spot = self._spot_from_snapshot(underlying_conid) or 100.0
        # ATM
        return float(spot)

