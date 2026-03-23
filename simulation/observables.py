# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:21:32 2026

@author: marin
"""

import numpy as np
from simulation.models import SimParameters
from simulation.forces import forces_potential_interactions, forces_potential_wall


def energy(x, v, cells, p : SimParameters):
    """
    Input:
        x = (N, 2) np.array of positions. 
        v = (N, 2) np.array of velocities
        cells = cell list
        p = parameters of the system
    Output :
        K = kinetic energy 
        U_inter = potential energy due to interparticle forces
        U_wall = potential due to wall interactions
        E_tot = total energy 
    This function, given an array of positions x and the corresponding cell list
    returns the various energies of the system.
    """
    #K = 1/2 v^2
    K = 0.5 * np.sum(v**2)
    _, U_inter = forces_potential_interactions(x, cells, p)
    _, U_wall = forces_potential_wall(x, p)
    
    E_tot = K + U_inter + U_wall
    return K, U_inter, U_wall, E_tot

def dmin(x, p: SimParameters):
    """
    Input:
        s = state of the system
        p = parameters of the system
    Output:
        r_min = the minimum distance between particles 
    """
    iu, ju = np.triu_indices(p.N, k=1)
    
    d = x[iu] - x[ju]
    
    r2 = np.sum(d*d, axis=1)
    r_min = np.sqrt(np.min(r2))
    return r_min