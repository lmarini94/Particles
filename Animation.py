# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:09:22 2026

@author: marin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import json, time

###############################################################################
############################# SELECTION OF RUN ################################
###############################################################################


def select_run(latest=True, runs_dir="run"):
    runs_dir = Path(runs_dir)
    
    #Check if run directory exists
    if not runs_dir.exists():
        raise FileNotFoundError(f"Run directory '{runs_dir}' does not exist.")

    runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])

    if not runs:
        raise RuntimeError("No runs found in directory.")

    if latest:
        run_name = runs[-1]
        print(f"Using latest run: {run_name}")
        return runs_dir / run_name

    # manual selection
    run_name = input("Enter run timestamp (YYYYMMDD_HHMM): ")

    if run_name not in runs:
        raise ValueError(f"Run '{run_name}' not found in {runs_dir}")

    return runs_dir / run_name

myrun = select_run()

###############################################################################
############################### IMPORT DATA ###################################
###############################################################################

# LOAD CONFIGURATION FILE
CONFIG_PATH = Path(myrun/"run_parameters.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)
    
#LOAD STATES AND ENERGIES FILE
states = np.load(myrun/"states.npy")
energies = np.load(myrun/"energies.npy")

#PHYSICAL PARAMETER
L = config["physical"]["L"]
N = config["physical"]["N"]
K_target = config["physical"]["K_target"] 
delta = config["physical"]["delta"]             
k_wall = config["physical"]["k_wall"]

#SIMULATION PARAMETER
speed = config["simulation"]["speed"]
h = config["simulation"]["h"]
T_MAX = config["simulation"]["T_MAX"]

#MODEL PARAMETER
soft_walls = config["model"]["soft_walls"]

#MODEL TIMING
created_at = config["Created_at"]
t_tot = config["Simulation_duration"]
formatted_time = time.strftime("%H:%M:%S", time.gmtime(t_tot))

print("\nSimulated on " + created_at)
print("\n-------Physical parameters-------")
print("Number of particles N =", N)
print("L =", L)
print("Initial energy K0 =", K_target)
if soft_walls:
    print("Wall model = elastic")
    print("Size of wall buffer =", delta)
    print("k_wall =", k_wall)
else:
    print("Wall model = hard reflection")
print("------Simulation parameters------")
print("Speed =", speed)
print("Total time =", T_MAX)
print("Step of the simulation h =", h)
print("-------------Timing--------------")
print(f"Simulation completed in {formatted_time}")



###############################################################################
########################## ANALYSIS AND DIAGOSTICS ############################
###############################################################################

#Recall energies is of the type [t_sim, K, U, U_wall, E_tot]
physical_time = energies[:, 0]  #Physiscal time
K = energies[:, 1]              #Kinetik energy
U_int = energies[:, 2]          #Interaction potential
U_wall = energies[:, 3]         #Wall potential
E = energies[:, 4]              #Total energy

dE = E -E[0]                    #Energy drift
dE_rel = dE/abs(E[0])           #Relative energy drift

print("-------Stability of energy-------")
print("$E0$ =", E[0])
print("U_int =", U_int[0])
print("U_wall0 =", U_wall[0])
print("K0 =", K[0])
print("rel max dev =", np.max(np.abs(dE)) / abs(E[0]))
print("rel range =", (E.max() - E.min()) / abs(E[0]))

###############################################################################
################################### GRAPHS ####################################
###############################################################################


#RELATIVE ENERGY DRIFT
plt.plot(physical_time, dE_rel, linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=0.5)
plt.xlabel("Physical time")
plt.ylabel("Relative energy drift $(E-E_0)/|E_0|$")
plt.title("Energy drift")
plt.savefig(myrun/"energy_drift.jpg", bbox_inches="tight", dpi=200)
plt.show()

#POTENTIAL, WALL AND KINETIK ENERGY
plt.plot(physical_time, U_wall/abs(E[0]), label = "Wall potential", linewidth = 0.5, color = "red")
plt.plot(physical_time, U_int/abs(E[0]), label = "Interaction potential",linewidth = 0.5, color = "blue")
plt.plot(physical_time, K/abs(E[0]), label = "Kinetik", linewidth = 0.5, color = "green")
plt.xlabel("Physical time")
plt.ylabel("Normalized energies")
plt.title("Energy components vs time")
plt.legend()
plt.tight_layout()
plt.savefig(myrun/"energy_components.jpg")
plt.show()


###############################################################################
################################# ANIMATION ###################################
###############################################################################

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Particle animation", fontsize=12, fontweight='bold')
scat = ax.scatter([], [], s = 8)

# ---- Parameter box ----
params = (
    f"$N$ = {N}\n"
    f"$L$ = {L}\n"
    f"$K_0$ = {K_target}\n"
    f"$T_{{max}}$ = {T_MAX}"
)

ax.text(
    1.02, 0.98, params,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top'
)

time_text = ax.text(
    1.02, 0.02, "",
    transform=ax.transAxes,
    fontsize=10,
    va='bottom'
)

def init():
    scat.set_offsets(np.empty((0, 2)))
    time_text.set_text("")
    return scat,

def update(i):
    scat.set_offsets(states[i].T)
    time_text.set_text(f"t = {physical_time[i]:.2f}")
    return scat,

ani = FuncAnimation(
    fig, update,
    frames=len(states),
    init_func= init,
    interval=2,
    blit=True
)

ani.save(myrun/"particles_film.gif", writer="pillow")

