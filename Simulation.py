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

# LOAD CONFIGURATION FILE
CONFIG_PATH = Path("CONFIG.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)
    
#EXTRACT PARAMETERS

#PHYSICAL PARAMETERS
#N -> Number of particles
#L -> Size of the box
#d_min -> Min initial dist between particles
#K_target -> Initial kinetik energy
#delta -> Size of buffer zone around walls
#k_wall -> Elastic constant of wall

N = config["physical"]["N"]                     
L = config["physical"]["L"]                     
d_min = config["physical"]["d_min"]             
K_target = config["physical"]["K_target"]       
delta = config["physical"]["delta"]             
k_wall = config["physical"]["k_wall"]

#SIMULATION PARAMETERS
#h -> Pace of the simulation
#eps -> Smoothing parameter for close range interactions
#speed -> Speed at which we store positions/energies
#T_MAX -> Total time of the simulation

h = config["simulation"]["h"]
eps = config["simulation"]["eps"]
speed = config["simulation"]["speed"]
T_MAX = config["simulation"]["T_MAX"]
seed = config["simulation"]["seed"]


rng = np.random.default_rng(seed)
#If "seed":null then seed = None


###############################################################################
############################### MAIN FUNCTIONS ################################
###############################################################################

def initialize_pos(d_min, delta, rng, max_tries = 2000):  
    """
    This function generates N points randomly in [delta, L-delta] so that their 
    mutual distance is greater than d_min. Points are generated using rng.    
    At every iteration a new point is generated ensuring that its dstance from
    all previously generated points is d_min. If not, point is rejected and new 
    attempt is made at most max_tries times. 
    Function return two N dimensional arrays x, y
    """

    pos = np.empty((N, 2), dtype=float)
    
    placed = 0
    tries = 0
    
    d2_min = d_min * d_min

    while placed < N:
        if tries > max_tries:
            raise RuntimeError("Failed to place all particles. Decrease d_min or increase L.")
        
        #Generate uniformly on [delta, L-delta] x,y coordinates of candidate particle
        candidate = rng.random(2) * (L-2*delta) + delta
        
        #First particle is placed
        if placed == 0:
            pos[0] = candidate
            placed = 1
            continue
        tries += 1
        
        #Compute the array of distances between placed particles and candidate
        d = pos[:placed] - candidate
        d2 = np.sum(d*d, axis=1)
        #d2 contains the distance squared of candidate from all placed particles
        
        if np.all(d2 >= d2_min):
            #if distance of candidate from all previously placed particles is enough
            #candedate is placed 
            pos[placed] = candidate
            placed += 1
            tries = 0
    
    return pos[:, 0], pos[:, 1]

def initialize_vel(K_target, rng):
    """
    This function initializes the x and y velocity vectors of N particles.
    Velocities are generated according to a normal distribution, net momentum 
    is removed and they are rescaled so that the system has a target kinetik 
    energy.
    The function returns vx, vy two N dimensional arrays.
    """
    
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
    two arrays of forces fx and fy and the potential energy U due to 
    interactions. Forces are computed using U = 1/r^6 -1/r^4. A smoothing 
    correction of eps is added to the computation of the distance between 
    particles.
    """
    
    #Compute the matrices of differences dx_ij = x[i] - x[j]
    dx = x[:, None] - x[None,:]
    dy = y[:, None] - y[None,:]
    
    r2 = dx*dx + dy*dy + eps
    
    #Sets the diagonal of r2 to infinity.
    np.fill_diagonal(r2, np.inf)
    
    r4 = r2*r2
    r6 = r2*r2*r2
    r8 = r4*r4
    
    coef = 6.0/r8 -4.0/r6
    #This is 0 on the diagonal.
    
    #fx = (6/r^8 -4/r^6)*dx
    #fy = 6/r^8 -4/r^6)*dy
    fx = np.sum(coef * dx, axis=1)
    fy = np.sum(coef * dy, axis=1)
    U = 0.5*np.sum(1/r6 - 1/r4)
    
    return fx, fy, U

def forces_and_potential_wall(x, y, delta, k_wall):
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

def energy(x, y, vx, vy, delta, k_wall):
    """
    This function, given a state of the system x, y, vx, vy (positions and 
    velocities in x and y directions) returns an array of the form 
    [K, U_inter, U_wall, E_tot] where K is the kinetik energy of the system
    U_inter is the potential due to the particle interactions and U_wall
    the potential due to the interaction with the wall. Finally it returns the
    total energy of the system.
    """
    #K = 1/2 v^2
    K = 0.5 * np.sum(vx**2 + vy**2)
    _, _, U_inter = forces_and_potential_interactions(x, y)
    _, _, U_wall = forces_and_potential_wall(x, y, delta, k_wall)
            
    return [K, U_inter, U_wall, K+U_inter+U_wall]

def d2min(x, y, N):
    """
    This function given a configuration of the system x, y returns the 
    minimum squared distance between two particles.
    """
    
    iu, ju = np.triu_indices(N, k=1)
    
    dx = x[iu] - x[ju]
    dy = y[iu] - y[ju]
    
    r2 = dx*dx + dy*dy 
    return np.min(r2)
    

def step (x, y, vx, vy):
    """
    At every iteration, this function updates the positions and velocities
    using a velocity-Vertlet symplectic integrator. 
    It returns a tuple of positions x, y and velocities vx, vy
    """
    
    fx, fy, _ = forces_and_potential_interactions(x, y)
    fx_wall, fy_wall, _ = forces_and_potential_wall(x, y, delta, k_wall)
    fx += fx_wall
    fy += fy_wall
    
    vx += 0.5*h*fx
    vy += 0.5*h*fy
    
    x += h*vx
    y += h*vy
    
    fx, fy, _ = forces_and_potential_interactions(x, y)
    fx_wall, fy_wall, _ = forces_and_potential_wall(x, y, delta, k_wall)
    fx += fx_wall
    fy += fy_wall
    
    vx += 0.5*h*fx
    vy += 0.5*h*fy    
    
    return x, y, vx, vy

###############################################################################
############################### MAIN EXECUTION ################################
###############################################################################

# INITIALIZATION OF VARIABLES

start_time = time.time()
states = []
energies = []

x, y = initialize_pos(d_min, delta, rng)
vx, vy = initialize_vel(K_target, rng)


t_sim = 0 
t = 0

while (t_sim <= T_MAX):
    if t%speed == 0:
        states.append(np.column_stack((x, y)))
        energies.append([t_sim] + energy(x, y, vx, vy, delta, k_wall))
        print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
    x, y, vx, vy = step(x, y, vx, vy)
    t_sim = t*h
    t += 1
print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
total_time = time.time()-start_time
states = np.array(states)
energies = np.array(energies) 

formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
print(f"\nSimulation completed in {formatted_time}")


###############################################################################
########################## DATA AND METADATA SAVE #############################
###############################################################################

state_path = config["outputs"]["states_file"]
energies_path = config["outputs"]["energies_file"]

# CREATE TIMESTAMPED OUTPUT DIRECTORY
timestamp = time.strftime("%Y%m%d_%H%M")
run_dir = Path("run") / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

# ADD TIMING PARAMETERS TO CONFIG AND SAVE TO FOLDER
config["Created_at_unix"] = time.time()
config["Created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
config["Simulation_duration"] = total_time
Path(run_dir/"run_parameters.json").write_text(json.dumps(config, indent=2))


# SAVE DATA FILES
np.save(run_dir / state_path, states) 
np.save(run_dir / energies_path, energies)

print(f"Results saved to: {run_dir}")

