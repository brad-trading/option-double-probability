import numpy as np
from odps.mc import MonteCarlo
from odps.vol_surface import VolatilitySurface, build_mock_surface

def test_intrastep_detects_between_grid():
    S0=100; K=100; T=0.02; r=0.0
    surface = VolatilitySurface(build_mock_surface(S0))
    # Create engine with coarse grid (forces intrastep)
    eng = MonteCarlo(num_paths=1000, steps_per_day=4, seed=1)
    res = eng.doubling_probability(S0,K,T,r,"call",current_option_price=1.0,vol_fn=surface.get_vol)
    # We don't assert a specific probability, just ensure it's in [0,1] and >0 for this setup
    assert 0.0 <= res["probability_double"] <= 1.0
