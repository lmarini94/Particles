# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:05:53 2026

@author: marin
"""

import numpy as np

def forces_potential_interactions(x, cells, p):
    """
    Input: 
        x = (N, 2) np.array of positions. 
        cells = cell list
        p = simulation parameters. 
    Output: 
        f = (N, 2) array of forces
        U = potential due to interactions
    Given a state of the sys
    This function computes the array of forces f and the potential energy U
    of particle interactions using a cell list approach. Forces are computed 
    using the potential U(r) = 1/r^6 -1/r^4 which is set to zero for r>r_c and
    shifted to ensure continuity. U = U(r) -U(r_c). A smoothing correction 
    of eps is added to the computation of the distance between particles.
    """
    N = p.N
    n_cells = p.n_cells
    rc2 = p.r_c*p.r_c
    eps2 = p.eps*p.eps
    
    f = np.zeros((N, 2))
    U = 0.0
    
    #Compute potential shift
    U_shift = 1/(p.r_c**2 + eps2)**3 - 1/(p.r_c**2 + eps2)**2
    
    #Neighbouring cells that will be checked 
    neighbors = [(0,0),(1,0),(-1,1),(0,1),(1,1)]
    
    #Cycle across all cells
    for cy in range(n_cells):
        for cx in range(n_cells):
            #Cycle across all cells
            c = cx + cy*n_cells
            A = cells[c]
            
            #If cell is empty skip
            if len(A)==0:
                continue
            
            #Check all 5 neighbouring cells
            for nx, ny in neighbors:
                
                cxn = cx + nx
                cyn = cy + ny 
                
                #If neighbour goes out of box skip
                if cxn < 0 or cxn >= n_cells or cyn >= n_cells or cyn < 0:
                    continue
                n = cxn + cyn*n_cells
                B = cells[n]
                
                #If neighbouring cell is empty skip
                if len(B) == 0:
                    continue
                
                #COMPUTE INTERACTIONS BETHWEEN CELL A AND B
                for i in A:
                    for j in B:
                        if c == n and i >= j:
                            continue
                        #dx = [dx, dy]
                        dx = x[i] - x[j]
                        r2 = np.sum(dx*dx)
                        if r2 >= rc2:
                            #Skip if distance more than cutoff
                            continue
                        inv_r2 = 1.0 / (r2 + eps2)
                        inv_r4 = inv_r2 * inv_r2
                        inv_r6 = inv_r4 * inv_r2
                        inv_r8 = inv_r4 * inv_r4
                        coef = 6*inv_r8 - 4*inv_r6
                        
                        f[i] += coef*dx
                        f[j] -= coef*dx
                        
                        U += inv_r6 - inv_r4 - U_shift
                        
    return f, U

def forces_potential_wall(x, p):
    """
    Input: 
        x = (N, 2) np.array of positions. 
        p = instance of SimParameters
    Output: 
        f = (N, 2) array of forces
        U = potential energy due to wall
    This function, computes the forces due to the wall interactions and the 
    corresponding potential energy.This is modelled as an elastic force which 
    acts on a buffer zone of lenght delta from the walls.
    """
    f = np.zeros_like(x)
    U = 0.0
    
    #Upper wall x = L, y= L
    active = x > (p.L-p.delta)
    d = x[active] - (p.L - p.delta)
    f [active] += -p.k_wall*d
    U += 0.5 * p.k_wall * np.sum(d*d)
    
    #Lower wall x = 0, y= 0
    active = x < p.delta
    d = p.delta - x[active]
    f [active] += p.k_wall*d
    U += 0.5 * p.k_wall * np.sum(d*d)
    
    return f, U