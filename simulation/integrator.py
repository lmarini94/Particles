# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:53:29 2026

@author: marin
"""

from simulation.forces import forces_potential_interactions, forces_potential_wall

def step (x, v, cells, p):
    """
    Input:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
        cells = array of np.arrays where cells[c] is the array which 
        contains the indices of particles in cell
        p = simulation parameters
    Output:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
    At every iteration, this function updates the positions and velocities
    using a velocity-Vertlet symplectic integrator. 
    """
    
    f, _ = forces_potential_interactions(x, cells, p)
    f_wall, _ = forces_potential_wall(x, p)
    f += f_wall
    
    v += 0.5*p.h*f
    
    x += p.h*v
    
    f, _ = forces_potential_interactions(x, cells, p)
    f_wall, _ = forces_potential_wall(x, p)
    f += f_wall
    
    v += 0.5*p.h*f
    
    return x, v
    