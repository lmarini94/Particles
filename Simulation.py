# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:19:28 2026

@author: marin
"""

import numpy as np
import time, json
from pathlib import Path


###############################################################################
################################# PARAMETERS ##################################
###############################################################################

#PHYSICAL PARAMETERS
N = 128          #Number of particles
L = 100         #Size of the box
d_min = 1.5     #Minimum initial distance between particles
K_target= 1.5   #The target initial kinetik energy
delta = 1.0     #Size of buffer zone around the walls
k_wall = 5.0    #Elastic coefficient of wall

#SIMULATION PARAMETER
h = 0.0005      #Step of the integrator
eps = 1e-2      #Smoothing parameter for close range interactions
speed = 1000    #Speed at which we store 
T_MAX = 700     #Total time of the simulation


###############################################################################
############################### MAIN FUNCTIONS ################################
###############################################################################

def initialize_pos(d_min, rng = None, max_tries = 2000):  
    """
    This function generates N points uniformly in [0, L]x[0, L] so that their 
    mutual distance is greater than d_min. At every iteration it generates a 
    new point and ensures that it is distnat more than d_min from previously 
    placed particles, if not the point is rejected and new attempt is made. 
    If rejects exceed max_tries function raises Error. 
    """
    rng = np.random.default_rng() if rng is None else rng
    
    pos = np.empty((N, 2), dtype=float)
    
    placed = 0
    tries = 0
    
    d2_min = d_min * d_min

    while placed < N:
        if tries > max_tries:
            raise RuntimeError("Failed to place all particles. Decrease r_min or increase L.")
        tries += 1
        
        #Generate uniformly on [delta, L-delta] x,y coordinates of candidate particle
        candidate = rng.random(2) * (L-2*delta) + delta
        
        #First particle is placed
        if placed == 0:
            pos[0] = candidate
            placed = 1
            continue
        
        #Compute the array of distances between placed particles and candidate
        d = pos[:placed] - candidate
        d2 = np.sum(d*d, axis=1)
        #d2 contains the distance squared of candidate from all placed particles
        
        if np.all(d2 >= d2_min):
            #if distance of candidate from all previously placed particles is enough
            #candedate is placed 
            pos[placed] = candidate
            placed += 1
    
    return pos[:, 0], pos[:, 1]

def initialize_vel(K_target, rng = None):
    """
    This function initializes veloities of the particles so that the system 
    has a given target kinetik energy K_target
    """
    rng = np.random.default_rng() if rng is None else rng
    
    #Velocities are ranomly generated according to normal distribution
    vx = rng.normal(0.0, 1.0, N)
    vy = rng.normal(0.0, 1.0, N)

    # remove net momentum
    vx -= vx.mean()
    vy -= vy.mean()

    #Compute total kinetik energy
    K = 0.5*np.sum(vx*vx + vy*vy)

    scale = np.sqrt(K_target / K)
    vx *= scale
    vy *= scale

    return vx, vy
    
def forces_and_potential_interactions(x, y): 
    """
    Given a configuration of the system x, y, this function computes the 
    two arrays of forces fx and fy and the tital potential energy U. The 
    forces are computed using U = 1/r^6 -1/r^4. A smoothing correction 
    of eps is added to the computation of the distance between particles.
    """
    
    #Compute the matrices of differences dx_ij = x[i] - x[j]
    dx = x[:, None] - x[None,:]
    dy = y[:, None] - y[None,:]
    
    r2 = dx*dx + dy*dy + eps
    
    #Since r2 is zero on the diagonal then replace the zeros with infty
    np.fill_diagonal(r2, np.inf)
    
    r4 = r2*r2
    r6 = r2*r2*r2
    r8 = r4*r4
    
    coef = 6.0/r8 -4.0/r6
    
    #fx = (6/r^8 -4/r^6)*dx
    #fy = 6/r^8 -4/r^6)*dy
    fx = np.sum(coef * dx, axis=1)
    fy = np.sum(coef * dy, axis=1)
    U = 0.5*np.sum(1/r6 - 1/r4)
    
    return fx, fy, U

def forces_and_potential_wall(x, y):
    """
    This function, given a configuration of the system x, y computes the forces
    due to the interaction of particles with the wall (elastic forces), as well
    as the corresponding potetial energy.
    """
    fx = np.zeros(N)
    fy = np.zeros(N)
    U = 0.0
    
    #Wall x = L effect
    active = x > (L-delta)
    dx = x[active]- L + delta
    fx[active] += -k_wall*dx
    U += 0.5*k_wall*np.sum(dx*dx)
    
    #Wall x = 0 effect
    active = x < delta
    dx = delta - x[active]
    fx[active] += k_wall*dx
    U += 0.5*k_wall*np.sum(dx*dx)
    
    #Wall y = L effect
    active = y > (L-delta)
    dy = y[active] - L + delta
    fy[active] += -k_wall*dy
    U += 0.5*k_wall*np.sum(dy*dy)
    
    #Wall y = 0 effect
    active = y < delta
    dy = delta - y[active]
    fy[active] += k_wall*dy
    U += 0.5*k_wall*np.sum(dy*dy)
    
    return fx, fy, U

def total_energy (x, y, vx, vy):
    """
    This function, given a state of the system x, y, vx, vy (positions and 
    velocities in x and y directions) returns the total energy of the system. 
    The kinetik energy K = 1/2 v^2 and potential energy U = 1/r^6 - 1/r^4.
    """
    #K = 1/2 v^2
    K = 0.5 * np.sum(vx**2 + vy**2)
    _, _, U = forces_and_potential_interactions(x, y)
    _, _, U_wall = forces_and_potential_wall(x, y)
            
    return [K, U, U_wall, K+U+U_wall]

def d2min(x, y):
    """
    This function given a configuration of the system x, y returns the 
    minimum squared distance between two particles.
    """
    
    dx = x[:, None] - x[None,:]
    dy = y[:, None] - y[None,:]
    
    r2 = dx*dx + dy*dy 
    np.fill_diagonal(r2, np.inf)
    return np.min(r2)


# def wall_reflection_vel(pos, vel):
#     """
#     This function taxes as imput the position and velocity vectors. It flips
#     the direction of the velocity for every particle whose position is outside
#     the box of side L.
#     """
#     out = pos > L
#     vel[out] *= -1
    
#     out = pos < 0 
#     vel[out] *= -1
    

def step (x, y, vx, vy):
    """
    At every iteration, this function updates the positions and velocities
    using a velocity-vertlet symplectic integrator. 
    """
    
    fx, fy, _ = forces_and_potential_interactions(x, y)
    fx_wall, fy_wall, _ = forces_and_potential_wall(x, y)
    fx += fx_wall
    fy += fy_wall
    
    vx += 0.5*h*fx
    vy += 0.5*h*fy
    
    x += h*vx
    y += h*vy
    
    fx, fy, _ = forces_and_potential_interactions(x, y)
    fx_wall, fy_wall, _ = forces_and_potential_wall(x, y)
    fx += fx_wall
    fy += fy_wall
    
    vx += 0.5*h*fx
    vy += 0.5*h*fy
    
def save_metadata(path, **meta):
    meta = dict(meta)  # copy
    meta["created_at_unix"] = time.time()
    meta["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(path).write_text(json.dumps(meta, indent=2))
    
###############################################################################
############################### MAIN EXECUTION ################################
###############################################################################

# INITIALIZATION OF VARIABLES

start_time = time.time()
states = []
energies = []
rng = np.random.default_rng(seed=25) #set to None 

x, y = initialize_pos(d_min, rng)
vx, vy = initialize_vel(K_target, rng)


t_sim = 0 
t = 0

while (t_sim <= T_MAX):
    if t%speed == 0:
        states.append([x.copy(), y.copy()])
        energies.append([t_sim] + total_energy(x, y, vx, vy))
        print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
    step(x, y, vx, vy)
    t_sim += h
    t += 1
total_time = time.time()-start_time
states = np.array(states)
energies = np.array(energies) 


###############################################################################
########################## DATA AND METADATA SAVE #############################
###############################################################################
E = energies[:, 4] 
dE = E -E[0]                   
dE_rel = dE/abs(E[0])

meta = {
        "physical": {
            "N": N,
            "L": L,
            "d_min": d_min,
            "K_target": K_target,
            "delta": delta,
            "k_wall": k_wall,
            },
        "simulation": {
            "h": h,
            "eps": eps,
            "speed": speed,
            "T_MAX": T_MAX,
            },
        "timing": {
            "Sim_duration" : total_time
            },
        "energy_analysis": {
            "E_tot_0": energies[:, 4][0],
            "U_int_0": energies[:, 2][0],
            "U_wall_0": energies[:, 3][0],
            "K_0": energies[:, 1] [0],
            "rel_max_dev": np.max(np.abs(dE)) / abs(E[0]),
            "rel_range":(E.max() - E.min()) / abs(E[0]),
            },
        "model": {
            "pair_potential": "1/r^6 - 1/r^4",
            "soft_walls": True,
            },
        "outputs": {
            "states_file": "particles.npy",
            "energies_file": "energies.npy",
            },
        "rng": {
            "seed": 25,   # set to None if not used
            },
}

save_metadata("run_metadata.json", **meta)

np.save("particles.npy", states) 
np.save("energies.npy", energies)
