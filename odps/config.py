from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()  # ensure .env / exported vars are loaded

class Settings(BaseModel):
    ib_url: str | None = os.getenv("IB_GATEWAY_URL") or None
    ib_ca_bundle: str | None = os.getenv("IB_CA_BUNDLE") or None
    ib_verify_ssl: bool = (os.getenv("IB_VERIFY_SSL", "true").lower() not in ("0","false","no"))
    risk_free: float = float(os.getenv("DEFAULT_RISK_FREE") or 0.05)
