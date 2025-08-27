from datetime import datetime

def parse_expiry_to_years(expiry: str) -> float:
    """
    Accepts:
      - 'YYYY-MM-DD' (absolute date)
      - 'N' or 'N.5' (days if >=3, years if <3)
      - '0.25' (years)
    """
    s = str(expiry).strip()
    try:
        if "-" in s:
            d = datetime.strptime(s, "%Y-%m-%d")
            days = (d.date() - datetime.utcnow().date()).days
            return max(days/365.25, 1e-6)
        val = float(s)
        return val if val < 3 else val/365.25
    except Exception:
        # Fallback: 0.25y
        return 0.25
