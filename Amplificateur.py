#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amplificateur
authors: Pascal Paradis & Frédéric Maes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as op
from D_func import n_silica, n_germanosilicate, solve_modes
from puissance import gamma_core
# Sections efficaces
cs_s_abs_data = np.genfromtxt("abs_ErALP_1550nm.csv")
cs_s_ems_data = np.genfromtxt("ems_ErALP_1550nm.csv")

# Interpolation des sections efficaces
cs_s_abs = interp1d(cs_s_abs_data[:, 0], cs_s_abs_data[:, 1], fill_value="extrapolate")
cs_s_ems = interp1d(cs_s_ems_data[:, 0], cs_s_ems_data[:, 1], fill_value="extrapolate")

class Amplificateur(object):
    """Ceci est la classe amplificateur. Elle contient l'ensemble des
       paramètres de l'amplificateur ainsi que de l'ion Er3+
    """
    # définition des constantes utiles
    h = 6.626e-34   # Planck
    c = 3e8         # Vitesse de la lumière

    def __init__(self, L, lambda_s, pump="clad"):
        """Constructeur de la classe Amplificateur
        """

        #1) On commence avec les paramètres qui sont constant dans
        #   l'amplificateur simulé
        self.Ntot = 1e26 #ToDo        # Densité ioniques de dopant [m^3]
        self.ac= 4e-6#ToDo            # Rayon du coeur
        self.ag= 62.5e-6#ToDo         # Rayon de la gaine
        self.x_germ= 0.045#ToDo        # Concentration de germanium coeur
        self.lambda_p= 978e-9#ToDo     # Longeur d'onde de pompage
        self.w22= 1.7e-23#ToDo         # Transfert par pair d'ion
        self.tau2= 10.8e-3#ToDo        # Temps de vie du niveau 2
        self.tau3= 0.005e-3#ToDo           # Temps de vie du niveau 3
        self.tau4=  0.005e-3#ToDo           # Temps de vie du niveau 4
        self.sigma_p= 31.2e-26#ToDo    # Section efficace d'absorption à la
                                #  longueur d'onde de la pompe
        self.sigma_esa= 6.2e-26#ToDo   # Section efficace du ESA
        self.alpha_p= 0.01#ToDo        # Pertes de la fibre à la
                                #  longueur d'onde de pompage
        self.alpha_s= 0.01#ToDo        # Pertes de la fibre à la
                                #  longueur d'onde du signal

        # 2) Pour les paramètres ci-dessous , nous leur attribuons la valeur
        # donnée au constructeur
        self.L = L #ToDo               # longueur de l'amplificateur
        self.lambda_s = lambda_s #ToDo # Longueur d'onde du signal.

        # 3) Les paramètres qui dépendent des paramètres
        # définis précédemment à l'étape 1) et 2).
        self.Ac = np.pi * self.ac**2#ToDo                # Aire coeur
        self.Ag = np.pi * self.ag**2#ToDo                # Aire de la gaine
        # Calcul du confinement de la pompe selon qu'elle est injectée dans
        # la gaine ou dans le coeur de la fibre
        if pump == "clad":
            self.gamma_p = self.Ac / self.Ag#ToDo            # Confinement de la pompe dans la gaine
        elif pump == "core":
            self.gamma_p = self.confinement(self.lambda_p, self.x_germ)
            #ToDo                           # Confinement de la pompe dans le coeur
        # Confinement du signal
        self.gamma_s = self.confinement(self.lambda_s, self.x_germ) #ToDo Confinement du signal dans le coeur.
        # Section efficace d'absorption du signal
        self.sigma_abs = cs_s_abs(self.lambda_s)
        # Section efficace d'émission du signal
        self.sigma_ems = cs_s_ems(self.lambda_s)
        # Autres propriétés nécessaire pour le bon fonctionnement du code.
        self.N = np.ones(shape=(1,4)) * 0.25 * self.Ntot
        self.Pp = 0
        self.Ps = 0
        self.rp = 0
        self.rs = 0
        self.resa = 0
        self.dz = 0
        self.num_elements = 501
    
    

    def confinement(self,  lambda_i, x_germ):
        # ToDo: Créer votre propre fonction confinement qui prend les arguments lambda_i et x_germ, et qui utilise la(es)
        # ToDo  propriété(s) de l'objet Amplificateur(ex: self.ac)
        lam_um = lambda_i * 1e6

        n2 = float(n_silica(lam_um))
        n1 = float(n_germanosilicate(lam_um, x_germ))

        V, modes = solve_modes(self.ac, lambda_i, n1, n2, l_max=0)

        for md in modes:
            if md.l == 0 and md.m == 1:   # mode LP01
                gamma = gamma_core(md.u, md.w, V, md.l)
                return float(gamma)

        raise RuntimeError("Mode LP01 introuvable")
        

    def sol(self, Pp_launch, Ps_launch, num_elements):
        """Cette fonction intègre les équations de puissance élément par élément dans
        l'amplificateur pour une puissance injectée de pompe et de signal donnée.

        Inputs:
               Pp_launch(float):   Puissance pompe injectée (W)
               Ps_launch(float):   Puissance du signal injectée (W)
               num_elements(int):  Discrétization de la fibre dopée

        Outputs:
               z_sol(vect ix1):    Vecteur discrétisant la longueur de l'ampli
               Pp_sol(vect ix1):   Solution de la puissance pompe à
                                   chaque position de z_sol
               Ps_sol(vect ix1):   Solution de la puissance du signal à
                                   chaque position de z_sol
               N_sol(vect ix4):    Populations des niveaux d'énergie à chaque
                                   position de z_sol.
        """

        # Initialisation
        self.num_elements = num_elements
        z_sol = np.linspace(0, self.L, num_elements)
        self.dz = z_sol[1] - z_sol[0]
        Pp_sol = np.zeros(len(z_sol))
        Ps_sol = np.zeros(len(z_sol))
        N_sol = np.zeros((len(z_sol), 4))

        # assignation des valeurs à z=0
        Pp_sol[0] = Pp_launch
        Ps_sol[0] = Ps_launch
        N_sol[0,:] = self.sol_eq_niv(Pp_launch, Ps_launch)
        # Itérations de 0 à L pour trouver les solutions des équations de niveau
        # et de puissance
        for i in range(1, len(z_sol)):
            self.Pp = self._Pp() # Ici on intègre la puissance précédente pour obtenir la puissance suivante
            self.Ps = self._Ps()
            Pp_sol[i] = self.Pp
            Ps_sol[i] = self.Ps
            N_sol[i,:] = self.sol_eq_niv(self.Pp, self.Ps)
        return z_sol, Pp_sol, Ps_sol, N_sol

    def eq_niv(self, N):
        """Cette fonction contient les équations des niveaux (rate equations)
        de l'erbium en régime permanent.

        Inputs:
            N(vect 1x4):     Vecteur solution qui entraine un résidu nul (F=0).
            Pp (float):      Puissance de la pompe [W]
            Ps (float):      Puissance du signal laser [W]

        Outputs:
            F (vect 1x4):   Résidu des équations de niveaux en régime permanent
         """
        self.N = N
        self._Rs()
        self._Rp()
        self._Resa()
        return (self._dN4(),
                self._dN3(),
                self._dN2(),
                self._zero())

    def sol_eq_niv(self, Pp, Ps):
        """Cette fonction résoud les équations de niveaux avec une puissance de
        pompe et de signal donnée.

        Inputs
            Pp (float):  Puissances de la pompe[W]
            Ps (float):  Puissances du signal laser [W]

        Outputs
            N (vect 1x4):   Densité atomiques des niveaux d'énergie [m^-3]

        Les équations des niveaux sont un ensemble d'équations non-linéaires à
        4 variables.
        """
        self.Pp = Pp
        self.Ps = Ps
        self.N = op.root(self.eq_niv, self.N).x
        return self.N

    # Les fonctions suivantes (méthodes de l'objet Amplificateur) régissent la physique de
    # l'amplificateur

    # Taux de pompage et laser
    def _Rp(self):
        self.rp = self.sigma_p * self.gamma_p * self.lambda_p /   \
                (self.h * self.c * self.Ac) * (self.N[0] - self.N[2]) * self.Pp
        return

    def _Resa(self):
        # ToDo
        self.resa = self.sigma_esa * self.gamma_p * self.lambda_p / \
                (self.h * self.c * self.Ac) * self.N[2] * self.Pp
        return
    def _Rs(self):
        self.rs = self.gamma_s * self.lambda_s / \
                (self.h * self.c * self.Ac) * \
                (self.sigma_ems * self.N[1] - self.sigma_abs * self.N[0]) * self.Ps
        return
    # Équations des niveaux

    def _dN4(self):
        return -self.N[3]/self.tau4 + self.resa

    def _dN3(self):
        return -self.N[2]/self.tau3 + self.N[3]/self.tau4 + self.rp - self.resa + self.w22*self.N[1]**2

    def _dN2(self):
        return -self.N[1]/self.tau2 + self.N[2]/self.tau3 - self.rs - 2*self.w22*self.N[1]**2

    def _zero(self):
        return self.Ntot - (self.N[0] + self.N[1] + self.N[2] + self.N[3])

    # Intégration de la puissance de pompe et du signal
    def _Pp(self):
        return self.Pp * np.exp(-(self.gamma_p*(self.sigma_p*self.N[0] + self.sigma_esa*self.N[2]) +
                                  self.alpha_p)*self.dz)

    def _Ps(self):
        return self.Ps * np.exp((self.gamma_s*(self.sigma_ems*self.N[1] - self.sigma_abs*self.N[0]) -
                             self.alpha_s)*self.dz)

if __name__ == '__main__':
    # C'est ici que vous pouvez répondre aux questions. L'appel ci-bas est un exemple qui devrait fonctionner une fois
    # que le code est complété.:
#   EXEMPLE
    ampli = Amplificateur(20, 1580e-9, pump='clad')
    z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(30,1e-6,301)
    plt.plot(z_sol, Ps_sol) # Ici on met en graphique l'évolution du signal dans l'amplificateur.

    plt.show()

    
    lambda_s = np.linspace(1450e-9, 1600e-9, 500)   
    lambda_s_nm = lambda_s * 1e9                    

    
    pump_powers = [0, 2, 4, 8, 16, 32, 64, 128]     

    plt.figure(figsize=(9, 6))

    sigma_abs_vals = cs_s_abs(lambda_s)
    sigma_ems_vals = cs_s_ems(lambda_s)

    for Pp in pump_powers:
        
        ampli = Amplificateur(L=20, lambda_s=1550e-9, pump='clad')

        
        N = ampli.sol_eq_niv(Pp, 0.0)

        N1 = N[0]
        N2 = N[1]


        
        gamma = sigma_ems_vals * N2 - sigma_abs_vals * N1  

        plt.plot(lambda_s_nm, gamma, label=f'Pp = {Pp} W')

    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel("Longueur d'onde du signal λs (nm)")
    plt.ylabel("Coefficient de gain γ (m$^{-1}$)")
    plt.title("Coefficient de gain à l'entrée de la fibre (Ps = 0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    L = 5                   
    lambda_s = 1580e-9       
    pump_powers = [1, 2, 4, 6, 8, 10]   
    signal_powers = np.logspace(-9, 1, 80)  
    num_elements = 301

    plt.figure(figsize=(8, 5))

    for Pp_in in pump_powers:
        ampli = Amplificateur(L, lambda_s, pump='clad')

        gains_db = []
        Ps_out_list = []

        
        for Ps_in in signal_powers:
            z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(Pp_in, Ps_in, num_elements)

            Ps_out = Ps_sol[-1]
            Ps_out_list.append(Ps_out)

            G_db = 10 * np.log10(Ps_out / Ps_in)
            gains_db.append(G_db)

        plt.semilogx(signal_powers, gains_db, label=f'Pompe = {Pp_in} W')

        
        print(f"\nPompe = {Pp_in} W")
        print(f"  Gain petit signal  ≈ {gains_db[0]:.2f} dB")
        print(f"  Gain à 10 W entrée ≈ {gains_db[-1]:.2f} dB")

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Puissance du signal à l'entrée $P_s(0)$ [W]")
    plt.ylabel("Gain $G = 10\\log_{10}(P_s(L)/P_s(0))$ [dB]")
    plt.title("Gain d'un amplificateur de 5 m à 1580 nm")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

L = 15                  
lambda_s = 1540e-9      
Ps_in = 1e-6            
num_elements = 601

cas = [
    ("Pompage dans la gaine : 10 W", "clad", 10.0),
    ("Pompage dans le coeur : 0.75 W", "core", 0.75),
]

# Figures séparées
plt.figure(figsize=(8, 5))   # Figure 1 : gain
for etiquette, pump_type, Pp_in in cas:
    ampli = Amplificateur(L, lambda_s, pump=pump_type)
    z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(Pp_in, Ps_in, num_elements)

    G_db = 10 * np.log10(Ps_sol / Ps_in)
    plt.plot(z_sol, G_db, label=etiquette)

plt.xlabel("Longueur z [m]")
plt.ylabel("G(z) [dB]")
plt.title("Gain du signal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(8, 5))   # Figure 2 : coefficient de gain
for etiquette, pump_type, Pp_in in cas:
    ampli = Amplificateur(L, lambda_s, pump=pump_type)
    z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(Pp_in, Ps_in, num_elements)

    # gamma = sigma_ems*N2 - sigma_abs*N1
    gamma_z = ampli.sigma_ems * N_sol[:, 1] - ampli.sigma_abs * N_sol[:, 0]
    plt.plot(z_sol, gamma_z, label=etiquette)

plt.xlabel("Longueur z [m]")
plt.ylabel(r"$\gamma(z)$ [m$^{-1}$]")
plt.title("Coefficient de gain")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(8, 5))   # Figure 3 : absorption de la pompe
for etiquette, pump_type, Pp_in in cas:
    ampli = Amplificateur(L, lambda_s, pump=pump_type)
    z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(Pp_in, Ps_in, num_elements)

    pump_db = 10 * np.log10(Pp_sol / Pp_in)
    plt.plot(z_sol, pump_db, label=etiquette)

plt.xlabel("Longueur z [m]")
plt.ylabel(r"$10\log_{10}[P_p(z)/P_p(0)]$ [dB]")
plt.title("Absorption de la pompe")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()