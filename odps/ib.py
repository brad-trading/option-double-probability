# odps/ib.py
from __future__ import annotations

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

# ----------------- date helpers -----------------

def _third_friday(year: int, month: int) -> datetime.date:
    from calendar import monthcalendar, FRIDAY
    cal = monthcalendar(year, month)
    first_friday = cal[0][FRIDAY] != 0
    idx = 2 if first_friday else 3
    day = cal[idx][FRIDAY]
    return datetime(year, month, day, tzinfo=timezone.utc).date()

def _month_code_to_date(month_code: str) -> datetime.date:
    """
    Accepts 'YYYYMM', 'YYYYMMDD', or 'MONYY' (e.g., NOV25).
    Defaults to 3rd Friday when day is absent.
    """
    s = str(month_code).strip().upper()
    today = datetime.now(timezone.utc).date()
    if s.isdigit():
        if len(s) == 6:
            y, m = int(s[:4]), int(s[4:6])
            return _third_friday(y, m)
        if len(s) == 8:
            y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
            return datetime(y, m, d, tzinfo=timezone.utc).date()
    if len(s) == 5 and s[:3] in _MONTHS:  # MONYY
        m = _MONTHS[s[:3]]
        yy = int(s[3:])
        y = 2000 + yy if yy < 80 else 1900 + yy
        return _third_friday(y, m)
    # fallback ~3 months ahead
    m = (today.month + 2) % 12 + 1
    y = today.year + (1 if today.month > 10 else 0)
    return _third_friday(y, m)

def _years_between(d: datetime.date) -> float:
    today = datetime.now(timezone.utc).date()
    days = (d - today).days
    return max(days / 365.25, 1e-6)

def _derive_month_code_from_T(target_T_years: float) -> str:
    """
    Derive a YYYYMM code by adding T to today (UTC), snapping forward to the 3rd Friday month.
    """
    today = datetime.now(timezone.utc).date()
    import math
    days = int(round(target_T_years * 365.25))
    future = today.fromordinal(today.toordinal() + max(days, 1))
    y, m = future.year, future.month
    d = _third_friday(y, m)
    if d < future:
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        d = _third_friday(y, m)
    return f"{y:04d}{m:02d}"

def _month_variants_from_T(target_T_years: float) -> List[str]:
    """
    Return both YYYYMM and MONYY codes derived from T.
    """
    yyyymm = _derive_month_code_from_T(target_T_years)
    y, m = int(yyyymm[:4]), int(yyyymm[4:6])
    inv = {v: k for k, v in _MONTHS.items()}
    monyy = f"{inv[m]}{str(y)[2:]}"
    return [yyyymm, monyy]

def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

# ----------------- IB client -----------------

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

    def _client(self) -> httpx.Client:
        if not self.base_url:
            raise RuntimeError("IB base_url not set. Define IB_GATEWAY_URL or pass to IBClient().")
        verify = self.ca_bundle if self.ca_bundle else self.verify_ssl  # bool or path
        # trust_env=False so proxies don't interfere with localhost
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

    def _post(self, path: str, json: Dict | None = None):
        try:
            with self._client() as c:
                r = c.post(path, json=json or {})
                r.raise_for_status()
                if r.text and r.headers.get("content-type", "").startswith("application/json"):
                    return r.json()
                return {}
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"IB POST {path} returned {e.response.status_code}: {e.response.text}") from e
    # ------------- diagnostics -------------

    def auth_status(self) -> Dict:
        return self._get("/iserver/auth/status", params={})

    # ------------- low-level market data -------------

    def _snapshot(self, conids: List[int | str], fields: List[str]) -> List[Dict]:
        """
        Get snapshot for conids. If we hit 403 (no entitlements), auto-enable delayed data and retry once.
        """
        params = {"conids": ",".join(str(c) for c in conids), "fields": ",".join(fields)}
        try:
            js = self._get("/iserver/marketdata/snapshot", params)
            return js if isinstance(js, list) else [js]
        except RuntimeError as e:
            if " 403" in str(e) or "Forbidden" in str(e):
                # attempt to enable delayed data then retry once
                self._post("/iserver/marketdata/delayed/enable", {})
                js = self._get("/iserver/marketdata/snapshot", params)
                return js if isinstance(js, list) else [js]
            raise

    def _spot_from_snapshot(self, conid: int | str) -> Optional[float]:
        """Request last + bid/ask; fall back to mid if last is missing."""
        fields = [_FIELD_MAP["last"], _FIELD_MAP["bid"], _FIELD_MAP["ask"]]
        rows = self._snapshot([conid], fields)
        if not rows:
            return None
        row = rows[0]
        last = _safe_float(row.get(_FIELD_MAP["last"]))
        bid  = _safe_float(row.get(_FIELD_MAP["bid"]))
        ask  = _safe_float(row.get(_FIELD_MAP["ask"]))
        if last is not None:
            return last
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        return None

    # ------------- underlying selection -------------

    def get_contract(self, symbol: str) -> Dict:
        """
        Resolve underlying conid robustly and return {'conid': int, 'price': float|None}.
        Strategy:
          1) Search candidates with /iserver/secdef/search.
          2) Drop obvious bads (conid=2147483647, missing conid).
          3) Prefer exact symbol, US listings (NYSE/NASDAQ), SMART routes.
          4) Probe top candidates with snapshot (31/84/86); pick first with usable spot (last or mid).
          5) If none yield quotes, return best-scored candidate with price=None.
        """
        data = self._get("/iserver/secdef/search", {"symbol": symbol})
        if not data or not isinstance(data, list):
            raise RuntimeError(f"No contracts found for symbol '{symbol}'.")

        symu = symbol.upper()

        def good(x):
            conid = (x.get("conid") or x.get("conId") or x.get("contractId"))
            if not conid:
                return False
            try:
                if int(conid) == 2147483647:  # IB sentinel; not a real contract
                    return False
            except Exception:
                pass
            return True

        filtered = [x for x in data if good(x)] or data

        def score(x):
            sx = (str(x.get("symbol") or "")).upper()
            listing = (str(x.get("listingExchange") or x.get("exchange") or "")).upper()
            desc = (str(x.get("description") or "")).upper()
            exch = (str(x.get("exchange") or "")).upper()
            is_exact = 0 if sx == symu else 1
            is_us   = 0 if (listing in ("NYSE", "NASDAQ") or "NYSE" in desc or "NASDAQ" in desc) else 1
            is_smart= 0 if exch == "SMART" else 1
            return (is_exact, is_us, is_smart)

        candidates = sorted(filtered, key=score)

        fields = [_FIELD_MAP["last"], _FIELD_MAP["bid"], _FIELD_MAP["ask"]]
        for x in candidates[:5]:
            conid = x.get("conid") or x.get("conId") or x.get("contractId")
            if not conid:
                continue
            rows = self._snapshot([conid], fields)
            if not rows:
                continue
            row = rows[0]
            last = _safe_float(row.get(_FIELD_MAP["last"]))
            bid  = _safe_float(row.get(_FIELD_MAP["bid"]))
            ask  = _safe_float(row.get(_FIELD_MAP["ask"]))
            spot = last if last is not None else (0.5 * (bid + ask) if (bid is not None and ask is not None) else None)
            if spot is not None:
                return {"conid": int(conid), "price": float(spot)}

        # Fallback: usable conid but no quotes (likely missing entitlements)
        best = candidates[0]
        conid = best.get("conid") or best.get("conId") or best.get("contractId")
        if not conid:
            raise RuntimeError(f"Could not resolve conid for symbol '{symbol}'. Response: {best}")
        return {"conid": int(conid), "price": None}

    # ------------- chain discovery (month derived from T) -------------

    def _list_strikes_for_month(self, underlying_conid: int | str, yyyymm: str, exchange: str = "SMART") -> List[float]:
        """
        Get valid strikes for a specific month (YYYYMM) via /iserver/secdef/strikes.
        """
        params = {"conid": str(underlying_conid), "secType": "OPT", "exchange": exchange, "month": yyyymm}
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
        for m in (month_code,) if len(month_code) == 5 else (month_code,):  # allow direct pass-through
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

    # ------------- public: option snapshot & surface -------------

    def get_option_snapshot(self, underlying_conid: int | str, strike: float, option_type: str, T_years: float) -> Dict:
        """
        Derive month from T, fetch strikes for that month, resolve the exact option conid,
        and return snapshot with last/bid/ask/IV/Greeks. Falls back gracefully.
        """
        # Month candidates from T
        month_candidates = _month_variants_from_T(T_years)  # [YYYYMM, MONYY]
        # get strikes using YYYYMM (first candidate) or convert MONYY -> YYYYMM
        strikes = []
        month_code_for_strikes = None
        for mc in month_candidates:
            try:
                if len(mc) == 6 and mc.isdigit():
                    yyyymm = mc
                else:
                    d = _month_code_to_date(mc)
                    yyyymm = f"{d.year:04d}{d.month:02d}"
                strikes = self._list_strikes_for_month(underlying_conid, yyyymm)
                month_code_for_strikes = yyyymm
                if strikes:
                    break
            except Exception:
                continue

        if not strikes:
            return {k: None for k in list(_FIELD_MAP.keys())}

        # nearest strike
        K = float(min(strikes, key=lambda x: abs(x - float(strike))))

        # resolve conid: try MONYY then YYYYMM for info endpoint
        conid_opt = None
        chosen_mon = None
        for mon in month_candidates:
            conid_opt = self._resolve_option_conid(underlying_conid, mon, K, option_type)
            if conid_opt:
                chosen_mon = mon
                break
        if not conid_opt and month_code_for_strikes:
            conid_opt = self._resolve_option_conid(underlying_conid, month_code_for_strikes, K, option_type)
            chosen_mon = month_code_for_strikes
        if not conid_opt:
            return {k: None for k in list(_FIELD_MAP.keys())}

        fields = list(_FIELD_MAP.values())
        row = self._snapshot([conid_opt], fields)[0]
        out = {k: _safe_float(row.get(f)) for k, f in _FIELD_MAP.items()}
        out["conid"] = conid_opt
        out["strike_resolved"] = K
        out["month_code"] = chosen_mon
        return out

    def build_surface(self, underlying_conid: int | str, spot: float) -> Dict:
        """
        Live IV surface around ATM for an expiry close to 0.25y (derived locally).
        Falls back to mock if insufficient points.
        """
        target_T = 0.25
        month_candidates = _month_variants_from_T(target_T)

        # strikes for derived month (prefer YYYYMM)
        strikes = []
        for mc in month_candidates:
            try:
                if len(mc) == 6 and mc.isdigit():
                    yyyymm = mc
                else:
                    d = _month_code_to_date(mc)
                    yyyymm = f"{d.year:04d}{d.month:02d}"
                strikes = self._list_strikes_for_month(underlying_conid, yyyymm)
                if strikes:
                    break
            except Exception:
                continue

        if not strikes:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        atm = float(spot or 100.0)
        strikes_sorted = sorted(strikes, key=lambda x: abs(x - atm))
        band = sorted(set(strikes_sorted[:15]))  # avoid 429s

        points = []
        T_years = _years_between(_month_code_to_date(month_candidates[0]))
        fields = list(_FIELD_MAP.values())

        for K in band:
            iv = None
            # try call then put
            for side in ("C", "P"):
                conid = self._resolve_option_conid(underlying_conid, month_candidates[0], K, side)
                if conid:
                    row = self._snapshot([conid], fields)[0]
                    iv = _safe_float(row.get(_FIELD_MAP["implied_volatility"]))
                    if iv is not None:
                        break
            if iv is None:
                continue
            points.append({"K": float(K), "T": float(T_years), "iv": float(iv)})

        if len(points) < 5:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        return {"spot": float(spot or 100.0), "points": points}

    # ------------- public: strike by delta -------------

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
        Uses derived month from expiry_years and samples near-ATM to reduce API load.
        """
        T_target = expiry_years if (expiry_years and expiry_years > 0) else 0.25
        month_candidates = _month_variants_from_T(T_target)

        # get strikes for derived month
        strikes = []
        for mc in month_candidates:
            try:
                if len(mc) == 6 and mc.isdigit():
                    yyyymm = mc
                else:
                    d = _month_code_to_date(mc)
                    yyyymm = f"{d.year:04d}{d.month:02d}"
                strikes = self._list_strikes_for_month(underlying_conid, yyyymm, exchange=exchange)
                if strikes:
                    month_for_info = mc  # keep the variant we'll use for info
                    break
            except Exception:
                continue

        if not strikes:
            spot = self._spot_from_snapshot(underlying_conid) or 100.0
            return float(spot)

        spot = self._spot_from_snapshot(underlying_conid) or 100.0
        near = sorted(strikes, key=lambda k: abs(k - spot))[:21]  # ±10 around ATM

        best_K, best_err = near[0], 1e9
        fields = list(_FIELD_MAP.values())
        side = option_type.upper()[0]

        for K in near:
            conid = self._resolve_option_conid(underlying_conid, month_for_info, K, side, exchange=exchange)
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
    Back-compat wrapper so callers can still import
    `from odps.ib import find_strike_by_delta`.
    """
    return ib.find_strike_by_delta(
        underlying_conid=underlying_conid,
        option_type=option_type,
        target_abs_delta=target_abs_delta,
        expiry_years=expiry_years,
        exchange=exchange,
    )
