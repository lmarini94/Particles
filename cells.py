# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:20:10 2026

@author: marin
"""

import numpy as np

def build_cell(s, p):
    """
    Input:
        s = state of the system
        p = system parameters
    This function takes a configuration of the system and updates s.cells 
    accordingly. 
    """
    
    cells = [[] for _ in range(p.n_cells**2)]
    
    for i, pos in enumerate(s.x):
        #Vectorial coordinates of cell
        cx = int(pos[0]/p.cell_size)
        cy = int(pos[1]/p.cell_size)
        #Scalar coordinates of cell
        c = cx + cy*p.n_cells
        cells[c].append(i)
    
    #Convert into np.arrays
    cells = [np.array(c, dtype = int) for c in cells]
    
    s.cells = cells
