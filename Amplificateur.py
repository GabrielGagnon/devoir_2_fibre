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
        self.Ntot = #ToDo        # Densité ioniques de dopant [m^3]
        self.ac=#ToDo            # Rayon du coeur
        self.ag=#ToDo         # Rayon de la gaine
        self.x_germ=#ToDo        # Concentration de germanium coeur
        self.lambda_p=#ToDo     # Longeur d'onde de pompage
        self.w22=#ToDo         # Transfert par pair d'ion
        self.tau2=#ToDo        # Temps de vie du niveau 2
        self.tau3=#ToDo           # Temps de vie du niveau 3
        self.tau4=#ToDo           # Temps de vie du niveau 4
        self.sigma_p=#ToDo    # Section efficace d'absorption à la
                                #  longueur d'onde de la pompe
        self.sigma_esa=#ToDo   # Section efficace du ESA
        self.alpha_p=#ToDo        # Pertes de la fibre à la
                                #  longueur d'onde de pompage
        self.alpha_s=#ToDo        # Pertes de la fibre à la
                                #  longueur d'onde du signal

        # 2) Pour les paramètres ci-dessous , nous leur attribuons la valeur
        # donnée au constructeur
        self.L = #ToDo               # longueur de l'amplificateur
        self.lambda_s = #ToDo # Longueur d'onde du signal.

        # 3) Les paramètres qui dépendent des paramètres
        # définis précédemment à l'étape 1) et 2).
        self.Ac = #ToDo                # Aire coeur
        self.Ag = #ToDo                # Aire de la gaine
        # Calcul du confinement de la pompe selon qu'elle est injectée dans
        # la gaine ou dans le coeur de la fibre
        if pump == "clad":
            self.gamma_p = #ToDo            # Confinement de la pompe dans la gaine
        elif pump == "core":
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
        return

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
        return
    def _Rs(self):
        # ToDo
        return

    # Équations des niveaux

    def _dN4(self):
        return -self.N[3]/self.tau4 + self.resa

    def _dN3(self):
        return #ToDo

    def _dN2(self):
        return #ToDo

    def _zero(self):
        return self.Ntot - #ToDo

    # Intégration de la puissance de pompe et du signal
    def _Pp(self):
        return self.Pp * np.exp(-(self.gamma_p*(self.sigma_p*self.N[0] + self.sigma_esa*self.N[2]) +
                                  self.alpha_p)*self.dz)

    def _Ps(self):
        return #ToDo

if __name__ == '__main__':
    # C'est ici que vous pouvez répondre aux questions. L'appel ci-bas est un exemple qui devrait fonctionner une fois
    # que le code est complété.:
#   EXEMPLE
    ampli = Amplificateur(20, 1580e-9, pump='clad')
    z_sol, Pp_sol, Ps_sol, N_sol = ampli.sol(30,1e-6,301)
    plt.plot(z_sol, Ps_sol) # Ici on met en graphique l'évolution du signal dans l'amplificateur.

