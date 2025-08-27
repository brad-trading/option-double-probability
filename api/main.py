from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import time, uuid
from dotenv import load_dotenv

from odps.config import Settings
from odps.vol_surface import VolatilitySurface, build_mock_surface
from odps.mc import MonteCarlo
from odps.pricing import wilson_ci, bs_price
from odps.ib import IBClient, find_strike_by_delta
from odps.utils import parse_expiry_to_years

app = FastAPI(title="ODPS API", version="1.0.0")
load_dotenv()
cfg = Settings()

TASKS = {}

class CalcRequest(BaseModel):
    symbol: str
    expiry: str = Field(..., description="YYYY-MM-DD or days or fractional years")
    option_type: str = Field("call", pattern="^(call|put)$")
    strike: float | None = None
    target_delta: float = 0.25
    sims: int = 25000
    steps_per_day: int = 48
    seed: int = 42
    live: bool = False

class CalcResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime

class ResultResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    result: dict | None = None
    error: str | None = None

@app.get("/health")
def health():
    return {"status": "ok", "ib_enabled": bool(cfg.ib_url)}

@app.post("/calculate", response_model=CalcResponse)
def calculate(req: CalcRequest, bg: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status":"pending", "created_at": datetime.utcnow()}
    bg.add_task(_run, task_id, req)
    return CalcResponse(task_id=task_id, status="pending", created_at=TASKS[task_id]["created_at"])

@app.get("/status/{task_id}", response_model=ResultResponse)
def status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "task not found")
    rec = TASKS[task_id]
    return ResultResponse(task_id=task_id, status=rec["status"], created_at=rec["created_at"],
                          completed_at=rec.get("completed_at"), result=rec.get("result"), error=rec.get("error"))

def _run(task_id: str, req: CalcRequest):
    start = time.time()
    try:
        T = parse_expiry_to_years(req.expiry)
        ib = IBClient(cfg.ib_url, cfg.ib_ca_bundle) if (req.live and cfg.ib_url) else None

        if ib:
            contract = ib.get_contract(req.symbol)
            spot = contract["price"]
            K = req.strike or find_strike_by_delta(ib, contract["conid"], req.option_type, req.target_delta, T)
            opt = ib.get_option_snapshot(contract["conid"], K, req.option_type, T)
            px = opt["last"] or (opt["bid"] + opt["ask"]) / 2.0
            surface = ib.build_surface(contract["conid"], spot)
            vol_surface = VolatilitySurface(surface)
        else:
            spot = 100.0
            K = req.strike or (spot * (1 + (0.5 - req.target_delta)) if req.option_type=="call" else
                               spot * (1 - (0.5 + req.target_delta)))
            vol_surface = VolatilitySurface(build_mock_surface(spot))
            px = bs_price(spot, K, T, cfg.risk_free, vol_surface.get_vol(K, T), req.option_type)

        engine = MonteCarlo(num_paths=req.sims, steps_per_day=req.steps_per_day, seed=req.seed)
        res = engine.doubling_probability(S0=spot, K=K, T=T, r=cfg.risk_free,
                                          option_type=req.option_type, current_option_price=px,
                                          vol_fn=vol_surface.get_vol)
        out = {
            "symbol": req.symbol,
            "spot": spot, "strike": K, "expiry_years": T, "option_type": req.option_type,
            "current_option_price": px,
            "num_paths": res["total_paths"], "paths_doubled": res["paths_doubled"],
            "doubling_probability": res["probability_double"],
            "wilson_ci_95": wilson_ci(res["probability_double"], res["total_paths"]),
            "expected_max_value": res["expected_max_value"],
            "p5_max_value": res["p5_max_value"], "p95_max_value": res["p95_max_value"],
            "avg_time_to_double_years": res["avg_time_to_double_years"],
            "seconds": res["seconds"],
        }
        TASKS[task_id].update({"status":"completed", "completed_at": datetime.utcnow(), "result": out})
    except Exception as e:
        TASKS[task_id].update({"status":"failed", "completed_at": datetime.utcnow(), "error": str(e)})
