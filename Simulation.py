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

def initialize_pos(N, L, d_min, delta, rng, max_tries = 2000):  
    """
    Input:
        N = number of particles
        d_min = minimum distance between particles
        delta = size of wall buffer zone
        rng = random number generator
        max_tries = maximum number of attempts
    Output:
        pos = (N, 2) dimensional array of positions
    This function generates N points randomly in [delta, L-delta] so that their 
    mutual distance is greater than d_min. Points are generated using inputed rng.    
    At every iteration a new point is generated ensuring that its dstance from
    all previously generated points is d_min. If not, point is rejected and new 
    attempt is made at most max_tries times. 
    """
    
    # Initialize empty array
    pos = np.empty((N, 2), dtype=float)
    
    placed = 0
    tries = 0
    
    d2_min = d_min * d_min

    while placed < N:
        if tries > max_tries:
            raise RuntimeError("Failed to place all particles. Decrease d_min or increase L.")
        
        #Candidate [x, y] where x and y are generated uniformly on [delta, L-delta] 
        candidate = rng.random(2) * (L-2*delta) + delta
        
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

def initialize_vel(N, K_target, rng):
    """
    Input: 
        N = number of particles
        K_target = target kinetik energy of the system of particles
        rng = input random number generator
    Output:
        v = (N,2) array of velocities
    This function generates the velocity vectors of N particles. Velocities are 
    generated according to a normal distribution (given rng), net momentum is 
    removed and they are rescaled so that the system has a target kinetik energy.
    """
    
    #V generated accoding to (0, 1) normal. v is shape (N, 2)
    v = rng.normal(0.0, 1.0, (N, 2))

    # remove net momentum
    v -= v.mean(axis = 0)

    #Compute total kinetik energy
    K = 0.5*np.sum(v**2)

    scale = np.sqrt(K_target / K)
    v *= scale

    return v
    
def forces_and_potential_interactions(x): 
    """
    Input:
        x = (N,2) array of positions.
    Outputs: 
        f = (N, 2) array of forces
        U = potential energy 
    This function computes the array of forces f and the potential U due to 
    interparticles interactions of the system. Forces are computed using 
    U = 1/r^6 -1/r^4. A smoothing correction of eps is added to the computation 
    of the distance between particles.
    """
    
    #Compute the matrices of differences if x is shape (N, 2) then dx is (N, N, 2)
    dx = x[:, None, :] - x[None, :, :]
    
    #r2[i, j] = squared distance between particle i and j shape (N, N)
    r2 = np.sum(dx*dx, axis = 2) + eps
    
    
    #Sets the diagonal of r2 to infinity.
    np.fill_diagonal(r2, np.inf)
    
    r4 = r2*r2
    r6 = r2*r2*r2
    r8 = r4*r4
    
    coef = 6.0/r8 -4.0/r6
    #This is 0 on the diagonal.
    
    #fx = (6/r^8 -4/r^6)*dx
    #fy = (6/r^8 -4/r^6)*dy
    f = np.sum(coef[:, :, None] * dx, axis=1)
    U = 0.5*np.sum(1/r6 - 1/r4)
    
    return f, U

def forces_and_potential_wall(x, L, delta, k_wall):
    """
    Input:
        x = (N, 2) dimensional array of positions
        delta = size of wall buffer
        k_wall = wall elastic constant
    Output:
        f = (N, 2) array of forces
        U = elastic potential energy
    This function, computes the forces due to the wall interactions and the 
    corresponding potetial energy.This is modelled as an elastic force which 
    acts on a buffer zone of lenght delta from the walls.
    """
    f = np.zeros_like(x)
    U = 0.0
    
    #Upper wall x = L, y= L
    active = x > (L-delta)
    d = x[active] - (L - delta)
    f [active] += -k_wall*d
    U += 0.5 * k_wall * np.sum(d*d)
    
    #Lower wall x = 0, y= 0
    active = x < delta
    d = delta - x[active]
    f [active] += k_wall*d
    U += 0.5 * k_wall * np.sum(d*d)
    
    return f, U

def energy(x, v, L, delta, k_wall):
    """
    Input:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
        L = size of box
        delta = size of wall buffer
        k_wall = wall elastic constant
    Output :
        K = kinetik energy 
        U_inter = potential energy due to interparticle forces
        U_wall = potential due to wall interactions
        E_tot = total energy 
    This function, given a state of the system x, v (positions and 
    velocities) returns the various energies of the system.
    """
    #K = 1/2 v^2
    K = 0.5 * np.sum(v**2)
    _, U_inter = forces_and_potential_interactions(x)
    _, U_wall = forces_and_potential_wall(x, L, delta, k_wall)
    
    E_tot = K + U_inter + U_wall
    return K, U_inter, U_wall, E_tot

def dmin(x):
    """
    Input:
        x = (N, 2) dimensional array of positions
    Output:
        r_min = the minimum distance between particles 
    """
    N = len(x)
    iu, ju = np.triu_indices(N, k=1)
    
    d = x[iu] - x[ju]
    
    r2 = np.sum(d*d, axis=1)
    r_min = np.sqrt(np.min(r2))
    return r_min
    

def step (x, v, L, delta, k_wall, h):
    """
    Input:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
        L = size of box
        delta = size of wall buffer
        k_wall = wall elastic constant
    Output:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
    At every iteration, this function updates the positions and velocities
    using a velocity-Vertlet symplectic integrator. 
    """
    
    f, _ = forces_and_potential_interactions(x)
    f_wall, _ = forces_and_potential_wall(x, L, delta, k_wall)
    f += f_wall
    
    v += 0.5*h*f
    
    x += h*v
    
    f, _ = forces_and_potential_interactions(x)
    f_wall, _ = forces_and_potential_wall(x, L, delta, k_wall)
    f += f_wall
    
    v += 0.5*h*f
    
    return x, v

###############################################################################
############################### MAIN EXECUTION ################################
###############################################################################

# INITIALIZATION OF VARIABLES

start_time = time.time()
states = []
energies = []

x = initialize_pos(N, L, d_min, delta, rng)
v = initialize_vel(N, K_target, rng)

t_sim = 0 
t = 0

while (t_sim <= T_MAX):
    if t%speed == 0:
        states.append(x.copy())
        K, U_inter, U_wall, E_tot = energy(x, v, L, delta, k_wall)
        energies.append([t_sim, K, U_inter, U_wall, E_tot])
        print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
    x, v = step(x, v, L, delta, k_wall, h)
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

