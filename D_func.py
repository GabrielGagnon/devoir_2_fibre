import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq
from scipy.special import jv, kv
from miyagi import solve_modes_miyagi
from puissance import gamma_core
import matplotlib.pyplot as plt



@dataclass
class ModeRoot:
    l:int
    m:int
    u:float
    w:float
    beta:float
    neff:float

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

def f_char(u,l, V):
    if u <= 0 or u >= V:
        return np.nan
    w = np.sqrt(max(V*V-u*u,0.0))
    if w==0:
        w= 1e-15
    if w <= 0:
        return np.nan
    
    eps = 1e-12
    if l==0:
        if abs(jv(1,u)) < eps or abs(kv(1,w)) < eps:
            return np.nan
        left = -jv(0,u)/(u*jv(1,u))
        right = kv(0, w)/(w*kv(1,w))
        return right+left
    else:
        if abs(jv(l-1,u)) < eps or abs(kv(l-1, w)) < eps:
            return np.nan
        
        
        
        left = jv(l,u)/(u*jv(l-1,u))
        right = kv(l,w)/(w*kv(l-1,w))

        return left + right
    '''
    if l==0:
        left = u *jv(1,u)/jv(0,u)
        right = w*kv(1,w)/kv(0,w)
        return left - right
    else:
        left = u *jv(l-1,u)/jv(l,u)
        right = -w*kv(l-1,w)/kv(l,w)
        return left - right
        '''


def find_roots_for_l(V,l,n_samples=2000):
    u_grid = np.linspace(1e-5, V-1e-5, n_samples)
    f_grid = np.array([f_char(u,l,V) for u in u_grid])

    roots = []

    for i in range( len(u_grid)-1):
        f1,f2 = f_grid[i], f_grid[i+1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        if f1==0:
            roots.append(u_grid[i])
        if f1*f2 < 0:
            a,b = u_grid[i], u_grid[i+1]
            try:
                r = brentq(lambda u: f_char(u,l,V), a, b)
                roots.append(r)
            except ValueError:
                pass
    
    roots = np.array(sorted(roots))
    if len(roots)==0:
        return []
    cleaned = [roots[0]]
    for r in roots[1:]:
        if np.abs(r-cleaned[-1]) > 1e-4:
            cleaned.append(r)
    return cleaned

def solve_modes(a, lam, n1, n2, l_max =8):
    V,k0 = compute_V(a, lam, n1, n2)
    modes = []

    for l in range(0, l_max+1):
        roots_u = find_roots_for_l(V, l)
        for m, u in enumerate(roots_u, start=1):
            w = np.sqrt(max(V*V-u*u,0.0))
            beta = np.sqrt(max(k0**2*n1**2-u**2/a**2,0.0))
            neff = beta/k0
            if not (n2 < neff < n1):
                continue
            modes.append(ModeRoot(l=l,m=m,u=u,w=w,beta=beta,neff=neff))
    
    modes.sort(key = lambda x:(-x.neff,x.l,x.m))
    return V, modes

c = 299792458.0

A_Si = np.array([0.696166, 0.407942, 0.897479])
lam_Si = np.array([0.068404, 0.116241, 9.896161])

A_Ge = np.array([0.806866, 0.718158, 0.854168])
lam_Ge = np.array([0.068972, 0.153966, 11.84193])

def n_sellmeier(lam_um, A, lam_res_um):
    lam2 = lam_um**2
    return np.sqrt(1 + np.sum(A*lam2/(lam2-lam_res_um**2)))

def n_silica(lam_um):
    return n_sellmeier(lam_um, A_Si, lam_Si)

def n_germanosilicate(lam_um, x_molar):
    A_mix = A_Si+x_molar*(A_Ge-A_Si)
    lam_mix = lam_Si+x_molar*(lam_Ge-lam_Si)
    return n_sellmeier(lam_um, A_mix, lam_mix)

def neff_LP01_from_modes(modes):
    for md in modes:
        if md.l==0 and md.m==1:
            return float(md.neff)
    raise RuntimeError("LP01 introuvable")

def D_from_neff(lam_um, neff):
    lam_m = lam_um*1e-6
    d1 = np.gradient(neff, lam_m)
    d2 = np.gradient(d1, lam_m)
    D_SI = -(lam_m/c)*d2

    D_ps_nm_km = D_SI*1e12 * 1e-9*1e3
    return D_ps_nm_km
def find_zero_cross(lam_um, D):
    s = np.sign(D)
    for i in range(len(lam_um)-1):
        if s[i]*s[i+1] < 0:
            l1,l2 = lam_um[i], lam_um[i+1]
            d1,d2 = D[i], D[i+1]
            return l1 +(0-d1)*(l2-l1)/(d2-d1)
    return None
def neff_LP01_at(a_m, lam_um, x_molar):
    n2 = float(n_silica(lam_um))
    n1 = float(n_germanosilicate(lam_um, x_molar))
    V, modes = solve_modes(a_m, lam_um*1e-6, n1, n2, l_max=0)
    return neff_LP01_from_modes(modes),n1,n2

def a_from_V_const(a0_m,x0,x_new, lam_ref_um = 1.5):
    _,n1_0,n2_0 = neff_LP01_at(a0_m, lam_ref_um, x0)
    NA0 = np.sqrt(max(n1_0**2-n2_0**2,0.0))
    aNA_const = a0_m * NA0

    _,n1_new,n2_new = neff_LP01_at(a0_m, lam_ref_um, x_new)
    NA_new = np.sqrt(max(n1_new**2-n2_new**2,0.0))
    return aNA_const / NA_new

def D15_fast_for_x(a0_m, x0, x_new, h_um = 0.01):
    a_new = a_from_V_const(a0_m, x0, x_new, lam_ref_um = 1.5)
    
    lam0 = 1.5
    n_m,_,_ = neff_LP01_at(a_new,lam0-h_um, x_new)
    n_0,_,_ = neff_LP01_at(a_new,lam0, x_new)
    n_p,_,_ = neff_LP01_at(a_new,lam0+h_um, x_new)

    h_m = h_um*1e-6
    lam0_m = lam0*1e-6
    d2 = (n_p - 2*n_0 + n_m) / (h_m**2)
    D_SI = -(lam0_m/c)*d2
    return float(D_SI*1e12 * 1e-9*1e3)