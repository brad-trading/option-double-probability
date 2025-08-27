from odps.greeks import bs_price, delta, gamma

def test_price_put_call_parity():
    S,K,T,r,sigma = 100,100,0.5,0.05,0.25
    c = bs_price(S,K,T,r,sigma,"call")
    p = bs_price(S,K,T,r,sigma,"put")
    # Parity: C - P = S - K e^{-rT}
    assert abs((c - p) - (S - K*(2.718281828459045**(-r*T))) ) < 1e-1

def test_greeks_signs():
    S,K,T,r,sigma = 100,100,0.25,0.03,0.2
    assert delta(S,K,T,r,sigma,"call") > 0
    assert delta(S,K,T,r,sigma,"put") < 0
    assert gamma(S,K,T,r,sigma) > 0
