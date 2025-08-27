from pydantic import BaseModel
import os

class Settings(BaseModel):
    ib_url: str | None = os.getenv("IB_GATEWAY_URL") or None
    ib_ca_bundle: str | None = os.getenv("IB_CA_BUNDLE") or None
    risk_free: float = float(os.getenv("DEFAULT_RISK_FREE") or 0.05)
