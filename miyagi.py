import numpy as np
from dataclasses import dataclass
from scipy.special import jn_zeros

@dataclass
class ModeApprox:
    l:int
    m:int
    u:float
    w:float
    beta:float
    neff:float

def compute_V(a, lam, n1, n2):
    k0 = 2*np.pi / lam
    NA = np.sqrt(n1**2-n2**2)
    V = k0*a*NA
    return V,k0

def miyagi_u(V, l, m, n_terms = 2):
    if V<= 0:
        return None
    
    u_inf = jn_zeros(l,m)[-1]

    denom = V+1.0
    series = 1.0

    if n_terms >= 1:
        series -= (u_inf**2)/(6.0*denom**3)
    if n_terms >= 2:
        series -=(u_inf**4)/(20.0*denom**5)

    u = u_inf*(V/denom)*series

    if not(0.0 < u < V):
        return None
    
    return float(u)

def lp_cutoff(l, m):
    if l == 0:
        if m == 1:
            return 0.0
        return jn_zeros(1, m-1)[-1]
    else:
        return jn_zeros(l-1, m)[-1]
    
def solve_modes_miyagi(a, lam, n1, n2, l_max=10, m_max =10, n_terms = 2):
    V,k0 = compute_V(a, lam, n1, n2)
    modes = []

    for l in range(l_max+1):
        for m in range(1,m_max+1):
            Vc = lp_cutoff(l,m)
            if V <= Vc:
                continue
            u = miyagi_u(V , l, m , n_terms=n_terms)
            if u is None:
                continue

            w = np.sqrt(max(V*V-u*u,0.0))

            beta = np.sqrt(max(0.0,((n1*k0)**2-(u/a)**2)))
            neff = beta / k0

            if not(n2 < neff < n1):
                continue

            modes.append(ModeApprox(l=l, m=m, u=u, w=float(w), beta=float(beta), neff=float(neff)))
    modes.sort(key=lambda x: (-x.neff,x.l,x.m))
    return V, modes