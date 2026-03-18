# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:21:32 2026

@author: marin
"""

import numpy as np
from models import SimState, SimParameters
from forces import forces_potential_interactions, forces_potential_wall


def energy(s : SimState, p : SimParameters):
    """
    Input:
        s = state of the system
        p = parameters of the system
    Output :
        K = kinetic energy 
        U_inter = potential energy due to interparticle forces
        U_wall = potential due to wall interactions
        E_tot = total energy 
    This function, given a state of the system 
    returns the various energies of the system.
    """
    #K = 1/2 v^2
    K = 0.5 * np.sum(s.v**2)
    _, U_inter = forces_potential_interactions(s, p)
    _, U_wall = forces_potential_wall(s, p)
    
    E_tot = K + U_inter + U_wall
    return K, U_inter, U_wall, E_tot

def dmin(s : SimState, p: SimParameters):
    """
    Input:
        s = state of the system
        p = parameters of the system
    Output:
        r_min = the minimum distance between particles 
    """
    iu, ju = np.triu_indices(p.N, k=1)
    
    d = s.x[iu] - s.x[ju]
    
    r2 = np.sum(d*d, axis=1)
    r_min = np.sqrt(np.min(r2))
    return r_min