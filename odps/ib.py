# odps/ib.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

# IB market data fields: friendly -> numeric ID strings
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

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def _third_friday(year: int, month: int) -> datetime.date:
    from calendar import monthcalendar, FRIDAY
    cal = monthcalendar(year, month)
    first_friday = cal[0][FRIDAY] != 0
    idx = 2 if first_friday else 3
    day = cal[idx][FRIDAY]
    return datetime(year, month, day, tzinfo=timezone.utc).date()

def _month_code_to_date(month_code: str) -> datetime.date:
    """
    Accepts 'YYYYMM', 'YYYYMMDD', or 'MONYY' (e.g., NOV25). Defaults to 3rd Friday when day absent.
    """
    s = str(month_code).strip().upper()
    # YYYYMM / YYYYMMDD
    if s.isdigit():
        if len(s) == 6:
            y, m = int(s[:4]), int(s[4:6])
            return _third_friday(y, m)
        if len(s) == 8:
            y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
            return datetime(y, m, d, tzinfo=timezone.utc).date()
    # MONYY
    if len(s) == 5 and s[:3] in _MONTHS:
        m = _MONTHS[s[:3]]
        yy = int(s[3:])
        y = 2000 + yy if yy < 80 else 1900 + yy
        return _third_friday(y, m)
    # fallback: 3 months ahead
    today = datetime.now(timezone.utc).date()
    m = (today.month + 2) % 12 + 1
    y = today.year + (1 if today.month > 10 else 0)
    return _third_friday(y, m)

def _years_between(d: datetime.date) -> float:
    today = datetime.now(timezone.utc).date()
    days = (d - today).days
    return max(days / 365.25, 1e-6)

def _to_month_codes(obj) -> List[str]:
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict) and "months" in obj:
        return [str(x) for x in obj["months"]]
    return []

def _pick_best_month(months: List[str], target_T_years: float) -> Optional[str]:
    if not months:
        return None
    target_days = target_T_years * 365.25
    scored: List[Tuple[str, float]] = []
    today = datetime.now(timezone.utc).date()
    for m in months:
        try:
            d = _month_code_to_date(m)
            diff = abs((d - today).days - target_days)
            scored.append((m, diff))
        except Exception:
            pass
    if not scored:
        return None
    scored.sort(key=lambda x: x[1])
    return scored[0][0]

def _fmt_month_variants(month_code: str) -> List[str]:
    """
    Return both 'MONYY' and 'YYYYMM' variants for robustness with /secdef/info.
    """
    d = _month_code_to_date(month_code)
    yyyymm = f"{d.year:04d}{d.month:02d}"
    inv = {v: k for k, v in _MONTHS.items()}
    monyy = f"{inv[d.month]}{str(d.year)[2:]}"
    return [monyy, yyyymm]

def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

@dataclass
class IBClient:
    """
    Synchronous client for IB Client Portal API.

    SSL verification precedence:
      - If ca_bundle is a path, use it (verify=<path>).
      - Else if verify_ssl is False, disable verification (dev only).
      - Else verify=True.
    """
    base_url: Optional[str]
    ca_bundle: Optional[str] = None
    verify_ssl: bool = True

    # --------- low-level HTTP ---------

    def _client(self) -> httpx.Client:
        if not self.base_url:
            raise RuntimeError("IB base_url not set. Define IB_GATEWAY_URL or pass to IBClient().")
        verify = self.ca_bundle if self.ca_bundle else self.verify_ssl  # bool or path
        # trust_env=False so proxies don't hijack localhost
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
                f"Ensure it is running/authenticated and SSL settings are correct. Underlying: {e}"
            ) from e

    # --------- diagnostics ---------

    def auth_status(self) -> Dict:
        return self._get("/iserver/auth/status", params={})

    # --------- snapshots ---------

    def _snapshot(self, conids: List[int | str], fields: List[str]) -> List[Dict]:
        params = {"conids": ",".join(str(c) for c in conids), "fields": ",".join(fields)}
        js = self._get("/iserver/marketdata/snapshot", params)
        return js if isinstance(js, list) else [js]

    def _spot_from_snapshot(self, conid: int | str) -> Optional[float]:
        # ask for last + bid/ask; use mid if last is missing
        fields = [_FIELD_MAP["last"], _FIELD_MAP["bid"], _FIELD_MAP["ask"]]
        rows = self._snapshot([conid], fields)
        if not rows:
            return None
        row = rows[0]
        last = _safe_float(row.get(_FIELD_MAP["last"]))
        bid = _safe_float(row.get(_FIELD_MAP["bid"]))
        ask = _safe_float(row.get(_FIELD_MAP["ask"]))
        if last is not None:
            return last
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        return None

    # --------- underlying ---------

    def get_contract(self, symbol: str) -> Dict:
        """
        Resolve underlying conid and return {'conid': int, 'price': float|None}
        where price is last or bid/ask mid.
        """
        data = self._get("/iserver/secdef/search", {"symbol": symbol})
        if not data or not isinstance(data, list):
            raise RuntimeError(f"No contracts found for symbol '{symbol}'.")
        candidates = [x for x in data if str(x.get("secType", "")).upper() == "STK"] or data
        sym_u = symbol.upper()
        exact = [x for x in candidates if str(x.get("symbol", "")).upper() == sym_u]
        chosen = exact[0] if exact else candidates[0]
        conid = chosen.get("conid") or chosen.get("conId") or chosen.get("contractId")
        if not conid:
            raise RuntimeError(f"Could not resolve conid for symbol '{symbol}'. Response: {chosen}")
        spot = self._spot_from_snapshot(conid)
        return {"conid": int(conid), "price": float(spot) if spot is not None else None}

    # --------- chain discovery ---------

    def _list_option_months(self, underlying_conid: int | str, exchange: str = "SMART") -> List[str]:
        """
        Query available months for the underlying's option chain via /iserver/secdef/strikes.
        """
        params = {"conid": str(underlying_conid), "secType": "OPT", "exchange": exchange}
        js = self._get("/iserver/secdef/strikes", params)
        months: List[str] = []
        if isinstance(js, dict) and "months" in js:
            months = _to_month_codes(js["months"])
        elif isinstance(js, list):
            for item in js:
                if isinstance(item, dict) and "months" in item:
                    months.extend(_to_month_codes(item["months"]))
        # uniq preserve order
        out, seen = [], set()
        for m in months:
            if m not in seen:
                out.append(m); seen.add(m)
        return out

    def _list_strikes_for_month(self, underlying_conid: int | str, month_code: str, exchange: str = "SMART") -> List[float]:
        params = {"conid": str(underlying_conid), "secType": "OPT", "exchange": exchange, "month": month_code}
        js = self._get("/iserver/secdef/strikes", params)
        strikes: List[float] = []
        raw: List = []
        if isinstance(js, dict) and "strikes" in js:
            raw = js["strikes"]
        elif isinstance(js, list):
            for item in js:
                if isinstance(item, dict) and "strikes" in item:
                    raw.extend(item["strikes"])
        seen = set()
        for v in raw:
            try:
                f = float(v)
                if f not in seen:
                    strikes.append(f); seen.add(f)
            except Exception:
                pass
        strikes.sort()
        return strikes

    def _resolve_option_conid(
        self,
        underlying_conid: int | str,
        month_code: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> Optional[int]:
        """
        Resolve option conid for (underlying, month, strike, right) using /iserver/secdef/info.
        Tries both MONYY and YYYYMM encodings.
        """
        side = right.upper()[0]  # 'C' or 'P'
        for m in _fmt_month_variants(month_code):
            params = {
                "conid": str(underlying_conid),
                "secType": "OPT",
                "month": m,
                "strike": f"{strike:g}",
                "right": side,
                "exchange": exchange,
            }
            try:
                js = self._get("/iserver/secdef/info", params)
            except Exception:
                js = None
            if not js:
                continue
            # dict or list, conid may be top-level or under 'contracts'
            if isinstance(js, dict):
                cid = js.get("conid") or js.get("conId") or js.get("contractId")
                if cid:
                    return int(cid)
                if isinstance(js.get("contracts"), list):
                    for c in js["contracts"]:
                        cid2 = c.get("conid") or c.get("conId") or c.get("contractId")
                        if cid2:
                            return int(cid2)
            elif isinstance(js, list):
                for item in js:
                    if isinstance(item, dict):
                        cid = item.get("conid") or item.get("conId") or item.get("contractId")
                        if cid:
                            return int(cid)
                        if isinstance(item.get("contracts"), list):
                            for c in item["contracts"]:
                                cid2 = c.get("conid") or c.get("conId") or c.get("contractId")
                                if cid2:
                                    return int(cid2)
        return None

    # --------- public: option snapshot & surface ---------

    def get_option_snapshot(self, underlying_conid: int | str, strike: float, option_type: str, T_years: float) -> Dict:
        """
        Pick expiry closest to T_years, snap to nearest listed strike, resolve option conid,
        and return snapshot with last/bid/ask/IV/Greeks.
        """
        months = self._list_option_months(underlying_conid)
        month_code = _pick_best_month(months, T_years) if months else None
        if not month_code:
            return {k: None for k in list(_FIELD_MAP.keys())}  # graceful fallback

        strikes = self._list_strikes_for_month(underlying_conid, month_code)
        K = float(strike)
        if strikes:
            K = min(strikes, key=lambda x: abs(x - K))

        conid_opt = self._resolve_option_conid(underlying_conid, month_code, K, option_type)
        if not conid_opt:
            return {k: None for k in list(_FIELD_MAP.keys())}

        fields = list(_FIELD_MAP.values())
        row = self._snapshot([conid_opt], fields)[0]
        out = {}
        for k, f in _FIELD_MAP.items():
            out[k] = _safe_float(row.get(f))
        out["conid"] = conid_opt
        out["strike_resolved"] = K
        out["month_code"] = month_code
        return out

    def build_surface(self, underlying_conid: int | str, spot: float) -> Dict:
        """
        Live IV surface around ATM for the expiry closest to ~0.25y.
        Falls back to mock surface if insufficient points or endpoints unavailable.
        """
        target_T = 0.25
        months = self._list_option_months(underlying_conid)
        month_code = _pick_best_month(months, target_T) if months else None
        if not month_code:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        strikes = self._list_strikes_for_month(underlying_conid, month_code)
        if not strikes:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        atm = float(spot or 100.0)
        strikes_sorted = sorted(strikes, key=lambda x: abs(x - atm))
        band = sorted(set(strikes_sorted[:15]))  # small band to avoid 429s

        points = []
        T_years = _years_between(_month_code_to_date(month_code))
        fields = list(_FIELD_MAP.values())

        # try calls first; if IV missing, try puts
        for K in band:
            iv = None
            conid_c = self._resolve_option_conid(underlying_conid, month_code, K, "C")
            if conid_c:
                row = self._snapshot([conid_c], fields)[0]
                iv = _safe_float(row.get(_FIELD_MAP["implied_volatility"]))
            if iv is None:
                conid_p = self._resolve_option_conid(underlying_conid, month_code, K, "P")
                if conid_p:
                    row2 = self._snapshot([conid_p], fields)[0]
                    iv = _safe_float(row2.get(_FIELD_MAP["implied_volatility"]))
            if iv is None:
                continue
            points.append({"K": float(K), "T": float(T_years), "iv": float(iv)})

        if len(points) < 5:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        return {"spot": float(spot or 100.0), "points": points}

    # --------- public: strike by delta ---------

    def find_strike_by_delta(
        self,
        underlying_conid: int | str,
        option_type: str,
        target_abs_delta: float = 0.25,
        expiry_years: float | None = None,
        exchange: str = "SMART",
    ) -> float:
        """
        Choose the strike with |Δ| closest to target_abs_delta for expiry ~expiry_years.
        Samples near-ATM to reduce API load.
        """
        months = self._list_option_months(underlying_conid, exchange=exchange)
        if not months:
            spot = self._spot_from_snapshot(underlying_conid) or 100.0
            return float(spot)

        T_target = expiry_years if (expiry_years and expiry_years > 0) else 0.25
        month_code = _pick_best_month(months, T_target) or months[0]

        strikes = self._list_strikes_for_month(underlying_conid, month_code, exchange=exchange)
        if not strikes:
            spot = self._spot_from_snapshot(underlying_conid) or 100.0
            return float(spot)

        spot = self._spot_from_snapshot(underlying_conid) or 100.0
        near = sorted(strikes, key=lambda k: abs(k - spot))[:21]  # ±10 around ATM

        best_K, best_err = near[0], 1e9
        fields = list(_FIELD_MAP.values())
        side = option_type.upper()[0]

        for K in near:
            conid = self._resolve_option_conid(underlying_conid, month_code, K, side, exchange=exchange)
            if not conid:
                continue
            row = self._snapshot([conid], fields)[0]
            delta_val = _safe_float(row.get(_FIELD_MAP["delta"]))
            if delta_val is None:
                continue
            err = abs(abs(delta_val) - target_abs_delta)
            if err < best_err:
                best_err, best_K = err, K

        return float(best_K)

# ---- Backward-compat shim for older callers ----
def find_strike_by_delta(
    ib: IBClient,
    underlying_conid,
    option_type: str,
    target_abs_delta: float = 0.25,
    expiry_years: float | None = None,
    exchange: str = "SMART",
) -> float:
    """
    Back-compat for code that imported `find_strike_by_delta` as a function.
    """
    return ib.find_strike_by_delta(
        underlying_conid=underlying_conid,
        option_type=option_type,
        target_abs_delta=target_abs_delta,
        expiry_years=expiry_years,
        exchange=exchange,
    )
