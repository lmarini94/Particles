# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:05:53 2026

@author: marin
"""

import numpy as np

def forces_potential_interactions(s, p):
    """
    Input: 
        s = instance of SimSate.
        p = instance of SimParameters
    Output: 
        f = (N, 2) array of forces
        U = potential due to interactions
    This function computes the array of forces f and the potential energy U
    of particle interactions using a cell list approach. Forces are computed 
    using the potential U(r) = 1/r^6 -1/r^4 which is set to zero for r>r_c and
    shifted to ensure continuity. U = U(r) -U(r_c). A smoothing correction 
    of eps is added to the computation of the distance between particles.
    """
    N = p.N
    n_cells = int(np.sqrt(len(s.cells)))
    rc2 = p.r_c*p.r_c
    eps2 = p.eps*p.eps
    
    f = np.zeros((N, 2))
    U = 0.0
    
    #Compute potential shift
    U_shift = 1/(p.r_c)**6 - 1/(p.r_c)**4
    
    #Neighbouring cells that will be checked 
    neighbors = [(0,0),(1,0),(-1,1),(0,1),(1,1)]
    
    #Cycle across all cells
    for cy in range(n_cells):
        for cx in range(n_cells):
            #Cycle across all cells
            c = cx + cy*n_cells
            A = s.cells[c]
            
            #If cell is empty skip
            if len(A)==0:
                continue
            
            #Check all 5 neighbouring cells
            for nx, ny in neighbors:
                #If neighbour goes out of box skip
                if (cx + nx) < 0 or (cx + nx) >= n_cells or (cy + ny) >= n_cells:
                    continue
                n = cx + nx + (cy + ny)*n_cells
                B = s.cells[n]
                
                #If neighbouring cell is empty skip
                if len(B) == 0:
                    continue
                
                #COMPUTE INTERACTIONS BETHWEEN CELL A AND B
                for i in A:
                    for j in B:
                        if c == n and i >= j:
                            continue
                        #dx = [dx, dy]
                        dx = s.x[i] - s.x[j]
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

def forces_potential_wall(s, p):
    """
    Input: 
        s = instance of SimSate.
        p = instance of SimParameters
    Output: 
        f = (N, 2) array of forces
        U = potential energy due to wall
    This function, computes the forces due to the wall interactions and the 
    corresponding potential energy.This is modelled as an elastic force which 
    acts on a buffer zone of lenght delta from the walls.
    """
    f = np.zeros_like(s.x)
    U = 0.0
    
    #Upper wall x = L, y= L
    active = s.x > (p.L-p.delta)
    d = s.x[active] - (p.L - p.delta)
    f [active] += -p.k_wall*d
    U += 0.5 * p.k_wall * np.sum(d*d)
    
    #Lower wall x = 0, y= 0
    active = s.x < p.delta
    d = p.delta - s.x[active]
    f [active] += p.k_wall*d
    U += 0.5 * p.k_wall * np.sum(d*d)
    
    return f, U