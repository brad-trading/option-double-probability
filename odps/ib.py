# odps/ib.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx


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

# --- helpers to parse/format months ---

_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _month_code_to_date(month_code: str) -> datetime.date:
    """
    Accepts common IB month encodings and returns a date representing (year, month, 3rd Friday).

    Supported examples: '202511', '20251115', 'NOV25', 'Nov25'
    We default day to the 3rd Friday of the month (typical US equity options).
    """
    s = month_code.strip().upper()
    today = datetime.now(timezone.utc).date()

    # YYYYMM or YYYYMMDD
    if s.isdigit():
        if len(s) == 6:
            y = int(s[:4])
            m = int(s[4:6])
            # pick 3rd Friday
            return _third_friday(y, m)
        if len(s) == 8:
            y = int(s[:4])
            m = int(s[4:6])
            d = int(s[6:8])
            return datetime(y, m, d, tzinfo=timezone.utc).date()

    # MONYY (e.g., NOV25)
    if len(s) == 5 and s[:3] in _MONTHS:
        m = _MONTHS[s[:3]]
        yy = int(s[3:5])
        y = 2000 + yy if yy < 80 else 1900 + yy  # crude pivot; fine for near-dated equity options
        return _third_friday(y, m)

    # Fallback: 3 months from now
    mth = (today.month + 2) % 12 + 1
    yr = today.year + (1 if today.month > 10 else 0)
    return _third_friday(yr, mth)


def _third_friday(year: int, month: int) -> datetime.date:
    """Return the 3rd Friday of the given month."""
    from calendar import monthcalendar, FRIDAY

    cal = monthcalendar(year, month)
    # If the first Friday is in the first week, third Friday is cal[2][FRIDAY], else cal[3][FRIDAY]
    first_friday = cal[0][FRIDAY] != 0
    week_idx = 2 if first_friday else 3
    day = cal[week_idx][FRIDAY]
    return datetime(year, month, day, tzinfo=timezone.utc).date()


def _years_between(d: datetime.date) -> float:
    """Year fraction from today (UTC) to date d."""
    today = datetime.now(timezone.utc).date()
    days = (d - today).days
    return max(days / 365.25, 1e-6)


def _to_month_codes(obj) -> List[str]:
    """
    Normalize the 'months' payload IB may return into a flat list of strings.
    IB sometimes returns {'months': ['202511','202512',...]} or nested structures.
    """
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict) and "months" in obj:
        return [str(x) for x in obj["months"]]
    return []


def _pick_best_month(months: List[str], target_T_years: float) -> Optional[str]:
    """Choose the month whose date (3rd Friday) is closest to target T."""
    if not months:
        return None
    # Build (month_code, |diff_days|)
    target_days = target_T_years * 365.25
    scored: List[Tuple[str, float]] = []
    for m in months:
        try:
            d = _month_code_to_date(m)
            diff = abs((d - datetime.now(timezone.utc).date()).days - target_days)
            scored.append((m, diff))
        except Exception:
            continue
    if not scored:
        return None
    scored.sort(key=lambda x: x[1])
    return scored[0][0]


def _fmt_month_variants(month_code: str) -> List[str]:
    """
    For /iserver/secdef/info we try multiple month encodings:
      - MONYY (e.g., NOV25)
      - YYYYMM (e.g., 202511)

    We produce both so we can call the endpoint twice if needed.
    """
    d = _month_code_to_date(month_code)
    yyyymm = f"{d.year:04d}{d.month:02d}"
    # e.g., NOV25
    inv = {v: k for k, v in _MONTHS.items()}
    monyy = f"{inv[d.month]}{str(d.year)[2:]}"
    return [monyy, yyyymm]


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
        verify = self.ca_bundle if self.ca_bundle else self.verify_ssl  # httpx accepts bool or path
        # trust_env=False avoids env proxies interfering with localhost
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

    # ------------- underlying ----------

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

    # ------------- options: live chain + conid -------------

    def _list_option_months(self, underlying_conid: int | str, exchange: str = "SMART") -> List[str]:
        """
        Query available months for the underlying's option chain.
        We call /iserver/secdef/strikes with only conid/sectype/exchange; IB returns months list.
        """
        params = {"conid": str(underlying_conid), "secType": "OPT", "exchange": exchange}
        js = self._get("/iserver/secdef/strikes", params)
        # try to normalize
        months = []
        if isinstance(js, dict) and "months" in js:
            months = _to_month_codes(js["months"])
        elif isinstance(js, list):
            # some deployments return list with one dict
            for item in js:
                if isinstance(item, dict) and "months" in item:
                    months.extend(_to_month_codes(item["months"]))
        # unique preserve order
        uniq = []
        seen = set()
        for m in months:
            if m not in seen:
                uniq.append(m)
                seen.add(m)
        return uniq

    def _list_strikes_for_month(
        self, underlying_conid: int | str, month_code: str, exchange: str = "SMART"
    ) -> List[float]:
        """
        Get valid strikes for a specific month.
        """
        params = {
            "conid": str(underlying_conid),
            "secType": "OPT",
            "exchange": exchange,
            "month": month_code,
        }
        js = self._get("/iserver/secdef/strikes", params)
        strikes = []
        if isinstance(js, dict) and "strikes" in js:
            strikes = js["strikes"]
        elif isinstance(js, list):
            for item in js:
                if isinstance(item, dict) and "strikes" in item:
                    strikes.extend(item["strikes"])
        # coerce to floats, unique
        out: List[float] = []
        seen = set()
        for v in strikes:
            try:
                f = float(v)
                if f not in seen:
                    out.append(f)
                    seen.add(f)
            except Exception:
                continue
        out.sort()
        return out

    def _resolve_option_conid(
        self,
        underlying_conid: int | str,
        month_code: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> Optional[int]:
        """
        Resolve the actual option conid for (underlying, month, strike, right).
        We call /iserver/secdef/info with both MONYY and YYYYMM variants to be robust.
        """
        right = right.upper()[0]  # 'C' or 'P'
        for m in _fmt_month_variants(month_code):
            params = {
                "conid": str(underlying_conid),
                "secType": "OPT",
                "month": m,
                "strike": f"{strike:g}",
                "right": right,
                "exchange": exchange,
            }
            try:
                js = self._get("/iserver/secdef/info", params)
            except Exception:
                js = None
            if not js:
                continue
            # Response may be a dict or list; look for conid
            if isinstance(js, dict):
                cid = js.get("conid") or js.get("conId") or js.get("contractId")
                if cid:
                    return int(cid)
                # sometimes nested under 'contracts'
                if "contracts" in js and isinstance(js["contracts"], list):
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
                        if "contracts" in item and isinstance(item["contracts"], list):
                            for c in item["contracts"]:
                                cid2 = c.get("conid") or c.get("conId") or c.get("contractId")
                                if cid2:
                                    return int(cid2)
        return None

    # ------------- public API used by your CLI -------------

    def get_option_snapshot(self, underlying_conid: int | str, strike: float, option_type: str, T_years: float) -> Dict:
        """
        Resolve the option conid for the expiry closest to T_years and return its live snapshot
        including implied_volatility (field 7283). Falls back gracefully if anything fails.
        """
        # 1) pick month closest to T_years
        months = self._list_option_months(underlying_conid)
        month_code = _pick_best_month(months, T_years) if months else None

        if not month_code:
            # fallback: return empty-ish structure
            return {"last": None, "bid": None, "ask": None, "implied_volatility": None, "delta": None, "gamma": None,
                    "theta": None, "vega": None}

        # 2) ensure strike exists; if not, choose nearest strike from chain
        chain_strikes = self._list_strikes_for_month(underlying_conid, month_code)
        K = float(strike)
        if chain_strikes:
            # take nearest strike
            K = min(chain_strikes, key=lambda x: abs(x - K))

        # 3) resolve option conid
        conid_opt = self._resolve_option_conid(underlying_conid, month_code, K, option_type)
        if not conid_opt:
            # fallback
            return {"last": None, "bid": None, "ask": None, "implied_volatility": None, "delta": None, "gamma": None,
                    "theta": None, "vega": None}

        # 4) snapshot actual option with all fields
        fields = list(_FIELD_MAP.values())
        row = self._snapshot([conid_opt], fields)[0]
        out = {}
        for k, f in _FIELD_MAP.items():
            val = row.get(f)
            try:
                out[k] = float(val) if val is not None else None
            except Exception:
                out[k] = None
        out["conid"] = conid_opt
        out["strike_resolved"] = K
        out["month_code"] = month_code
        return out

    def build_surface(self, underlying_conid: int | str, spot: float) -> Dict:
        """
        Live volatility surface around ATM for the expiry closest to ~3 months (0.25y),
        sampling strikes near ATM and pulling IVs (field 7283) for each strike.

        To avoid rate limits, we sample a small band (default ~15 strikes).
        """
        # Choose a target T = 0.25y for generic surface; your CLI will still query the correct T
        target_T = 0.25
        months = self._list_option_months(underlying_conid)
        month_code = _pick_best_month(months, target_T) if months else None
        if not month_code:
            # fallback to mock if we can't find months
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        # strikes around ATM
        strikes = self._list_strikes_for_month(underlying_conid, month_code)
        if not strikes:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        # pick up to ±7 around ATM
        atm = spot or 100.0
        strikes_sorted = sorted(strikes, key=lambda x: abs(x - atm))
        band = sorted(set(strikes_sorted[:15]))  # keep it small to avoid 429s

        # resolve and snapshot each option (calls) to get IV
        points = []
        T_years = _years_between(_month_code_to_date(month_code))
        fields = list(_FIELD_MAP.values())

        # We alternate call/put if call IV is None; this gives another shot at IV
        for K in band:
            conid_c = self._resolve_option_conid(underlying_conid, month_code, K, "C")
            row = None
            if conid_c:
                row = self._snapshot([conid_c], fields)[0]
                iv = _safe_float(row.get(_FIELD_MAP["implied_volatility"]))
            else:
                iv = None

            if iv is None:
                conid_p = self._resolve_option_conid(underlying_conid, month_code, K, "P")
                if conid_p:
                    row2 = self._snapshot([conid_p], fields)[0]
                    iv = _safe_float(row2.get(_FIELD_MAP["implied_volatility"]))

            if iv is None:
                # skip points without IV to avoid poisoning the surface
                continue

            points.append({"K": float(K), "T": float(T_years), "iv": float(iv)})

        # If too few points, fall back to mock
        if len(points) < 5:
            from .vol_surface import build_mock_surface
            return build_mock_surface(spot or 100.0)

        return {"spot": float(spot or 100.0), "points": points}

    # ------------- convenience for delta-selection -------------

    def find_strike_by_delta(
        self,
        underlying_conid: int | str,
        option_type: str,
        target_abs_delta: float = 0.25,
        expiry_years: float | None = None,
        exchange: str = "SMART",
    ) -> float:
        """
        Find strike with |Delta| closest to target_abs_delta for the expiry closest to expiry_years.
        """
        months = self._list_option_months(underlying_conid, exchange=exchange)
        if not months:
            # fallback: ATM
            spot = self._spot_from_snapshot(underlying_conid) or 100.0
            return float(spot)

        # pick expiry
        T_target = expiry_years if expiry_years and expiry_years > 0 else 0.25
        month_code = _pick_best_month(months, T_target) or months[0]

        strikes = self._list_strikes_for_month(underlying_conid, month_code, exchange=exchange)
        if not strikes:
            spot = self._spot_from_snapshot(underlying_conid) or 100.0
            return float(spot)

        # compute |Delta| for a small band around ATM to reduce API load
        spot = self._spot_from_snapshot(underlying_conid) or 100.0
        near = sorted(strikes, key=lambda k: abs(k - spot))[:21]  # ~±10 strikes

        best_K = near[0]
        best_err = 1e9
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
                best_err = err
                best_K = K

        return float(best_K)


# ---------- small utils ----------

def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None
