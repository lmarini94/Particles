# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:20:10 2026

@author: marin
"""

import numpy as np
from simulation.models import SimParameters

def build_cell(x, p : SimParameters):
    """
    Input:
        x = (N, 2) dimensional np.array of positions
        p = system parameters, instance of SimParameters
    Output:
        cells = cell list, cell[c] is a list of indices of particles in cell c
    This function takes a configuration of positions x, divides the [0, L]^2 
    box into n_cells^2 equal square cells and populates a cell list.
    """
    
    cells = [[] for _ in range(p.n_cells**2)]
    
    for i, pos in enumerate(x):
        
        #Vectorial coordinates of cell
        #min and max to prevent overshooting outside [O, L]
        cx = max(0, min(int(pos[0]/p.cell_size), p.n_cells-1))
        cy = max(0, min(int(pos[1]/p.cell_size), p.n_cells-1))
        
        #Scalar coordinates of cell
        c = cx + cy*p.n_cells
        cells[c].append(i)
    
    #Convert into np.arrays
    cells = [np.array(c, dtype = int) for c in cells]
    
    return cells
