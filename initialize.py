# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:00:28 2026

@author: marin
"""

import numpy as np

def initialize_pos(s, p, rng, max_tries = 2000):  
    """
    Input:
        p = Instance of SimParameters
        rng = random number generator
        max_tries = maximum number of attempts
    Output:
        pos = (N, 2) dimensional numpy array of positions
    This function generates N points randomly in [delta, L-delta] so that their 
    mutual distance is greater than d_min. Points are generated using inputed rng.    
    At every iteration a new point is generated ensuring that its dstance from
    all previously generated points is d_min. If not, point is rejected and new 
    attempt is made at most max_tries times. 
    """
    # Initialize empty array
    
    pos = np.empty((p.N, 2), dtype=float)
    
    placed = 0
    tries = 0
    
    d2_min = p.d_min * p.d_min

    while placed < p.N:
        if tries > max_tries:
            raise RuntimeError("Failed to place all particles. Decrease d_min or increase L.")
        
        #Candidate [x, y] where x and y are generated uniformly on [delta, L-delta] 
        candidate = rng.random(2) * (p.L-2*p.delta) + p.delta
        
        #First particle is placed
        if placed == 0:
            pos[0] = candidate
            placed = 1
            continue
        tries += 1
        
        #Distance of candidate from previously placed particles, size (placed,2)
        d = pos[:placed] - candidate
        d2 = np.sum(d*d, axis=1)
        #d2 has size (placed), d2[i] = squared distance of particle i and candidate
        
        if np.all(d2 >= d2_min):
            pos[placed] = candidate
            placed += 1
            tries = 0
    
    return pos

def initialize_vel(p, rng):
    """
    Input: 
        p = Instance of SimParameters
        rng = input random number generator
    Output:
        v = (N,2) array of velocities
    This function generates the velocity vectors of N particles. Velocities are 
    generated according to a normal distribution (given rng), net momentum is 
    removed and they are rescaled so that the system has a target kinetic energy.
    """
    
    #V generated accoding to (0, 1) normal. v is shape (N, 2)
    v = rng.normal(0.0, 1.0, (p.N, 2))

    # remove net momentum
    v -= v.mean(axis = 0)

    #Compute total kinetic energy
    K = 0.5*np.sum(v**2)

    scale = np.sqrt(p.K_0 / K)
    v *= scale

    return v