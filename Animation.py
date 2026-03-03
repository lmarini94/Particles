# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:09:22 2026

@author: marin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

#IMPORT METADATA
with open("run_metadata.json", "r") as f:
    meta = json.load(f)
    
L = meta["physical"]["L"]
N = meta["physical"]["N"]
K_target = meta["physical"]["K_target"] 
delta = meta["physical"]["delta"]
k_wall = meta["physical"]["k_wall"]

speed = meta["simulation"]["speed"]
h = meta["simulation"]["h"]
T_MAX = meta["simulation"]["T_MAX"]

states = np.load(meta["outputs"]["states_file"])
energies = np.load(meta["outputs"]["energies_file"])

###############################################################################
########################## ANALYSIS AND DIAGOSTICS ############################
###############################################################################
#Timing of algorithm
total_time = meta["timing"]["Sim_duration"]

#Recall energies is of the type [t_sim, K, U, U_wall, E_tot]
physical_time = energies[:, 0]  #Physiscal time
K = energies[:, 1]              #Kinetik energy
U = energies[:, 2]              #Interaction potential
U_wall = energies[:, 3]         #Wall potential
E = energies[:, 4]              #Total energy

dE = E -E[0]                    #Energy drift
dE_rel = dE/abs(E[0])           #Relative energy drift

###############################################################################
################################### GRAPHS ####################################
###############################################################################


#RELATIVE ENERGY DRIFT
plt.plot(physical_time, dE_rel, linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=0.5)
plt.xlabel("Physical time")
plt.ylabel("Relative energy drift $(E-E_0)/|E_0|$")
plt.title("Energy drift")
plt.show()

#POTENTIAL, WALL AND KINETIK ENERGY
plt.plot(physical_time, U_wall/abs(E[0]), label = "Normalized wall potential", linewidth = 0.5, color = "red")
plt.plot(physical_time, U/abs(E[0]), label = "Normalized interaction potential",linewidth = 0.5, color = "blue")
plt.plot(physical_time, K/abs(E[0]), label = "Normalized kinetik", linewidth = 0.5, color = "green")

plt.xlabel("Physical time")
plt.ylabel("Normalized energies")
plt.title("Energy components vs time")

plt.legend()
plt.tight_layout()
plt.show()



print("-------Physical parameters-------")
print("Number of particles N =", N)
print("L =", L)
print("Initial energy K0 =", K_target)
print("Size of wall buffer =", delta)
print("k_wall =", k_wall)
print("-------Simulation parameters-------")
print("Speed =", speed)
print("Total time =", T_MAX)
print("Step of the simulation h =", h)
print("-------Stability of energy-------")
print("E0 =", E[0])
print("U0 =", U[0])
print("U_wall0 =", U_wall[0])
print("K0 =", K[0])
print("rel max dev =", np.max(np.abs(dE)) / abs(E[0]))
print("rel range =", (E.max() - E.min()) / abs(E[0]))
print("-------Timing-------")
print("Time elapsed %s" % total_time)

###############################################################################
################################# ANIMATION ###################################
###############################################################################

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal', adjustable='box')
scat = ax.scatter([], [], s = 8)

# ---- Parameter box ----
params = (
    f"$N$ = {N}\n"
    f"$L$ = {L}\n"
    f"$K_0$ = {K_target}"
)

ax.text(
    1.02, 0.98, params,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top'
)

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update(i):
    scat.set_offsets(states[i].T)
    return scat,

ani = FuncAnimation(
    fig, update,
    frames=len(states),
    init_func= init,
    interval=2,
    blit=True
)

ani.save("particles.gif", writer="pillow")

