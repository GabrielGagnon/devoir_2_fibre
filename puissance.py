import numpy as np
from scipy.special import kv

def psi_l(w,l, eps=1e-14):
    if w<= 0:
        return np.nan
    Kl = kv(l,w)
    Klp1 = kv(l+1,w)

    if l == 0:
        Klm1 = kv(1,w)
    else:
        Klm1 = kv(l-1,w)

    denom = Klp1*Klm1
    if abs(denom) < eps:
        return np.nan
    return (Kl*Kl)/(denom)

def gamma_core(u,w,V,l):
    ps = psi_l(w,l)
    if not np.isfinite(ps):
        return np.nan
    return 1.0-(u*u)/(V*V)*(1.0-ps)
    