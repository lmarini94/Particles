# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:55:55 2026

@author: marin
"""

import time
import numpy as np
from pathlib import Path
from simulation.models import SimState, SimParameters
from simulation.cells import build_cell
from simulation.initialize import initialize_pos, initialize_vel
from simulation.integrator import step
from simulation.observables import energy
from simulation.saving import save_run



###############################################################################
############################# SIMULATION CLASS ################################
###############################################################################

class Simulation:
    """
    The Simulation class wraps the fill lifecycle of a simulation run. To delcare
    instance of Simulation class we need:
        params = instance of SimParameters containing the simulation parameters
        rng = np.rnaodm.Generator
    Other attributes are:
        states = list of np.arrays recording the  positions (T,N,2) size
        energies = list of lists of size (T, 5)
        total_time = timing of the simulation
   Main functions:
        1. initialize()  — set up positions, velocities, cell lists (state)
        2. run()         — evolve the system up to T_MAX
        3. save()        — write states, energies, and metadata
    """

    def __init__(self, params: SimParameters, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.state: SimState | None = None

        # Recorded data
        self.states: list[np.ndarray] = []
        self.energies: list[list[float]] = []

        # Timing
        self.total_time: float = 0.0


    ###########################################################################
    ############################# INITIALIZATION ##############################
    ###########################################################################

    def initialize(self):
        """
        Initialize positions, velocities and cell lists. Must be called
        before run().
        """
        p = self.params
        
        #Initialize position and velocities
        x = initialize_pos(p, self.rng)
        v = initialize_vel(p, self.rng)
        
        #Create initial cell list
        cells = build_cell(x, p)
        
        #Populate state attribute
        self.state = SimState(
            x         = x,
            v         = v,
            cells     = cells,
            t         = 0,
            t_sim     = 0.0,
        )


    ###########################################################################
    ############################### MAIN LOOP #################################
    ###########################################################################

    def run(self):
        """
        Runs the simulation from t=0 to T_MAX using a velocity-Verlet
        integrator. States and energies are recorded every `speed` steps.
        """
        #Check that state has been initialized
        if self.state is None:
            raise RuntimeError("Call initialize() before run().")

        p = self.params
        start_time = time.time()
        x_ref = self.state.x
        r_skin = p.r_skin

        while True:
            
            #Save every speed step
            if self.state.t % p.speed == 0:
                self._record()
                self._print_progress()
        
            #Check if we need to compute next step
            if (self.state.t +1)*p.h > p.T_MAX:
                break
            
            #Increment
            self._step()
            
            #If particles have moved more than r_skin/2 rebuild cells
            if np.max(np.sum((self.state.x - x_ref)**2, axis=1)) > (r_skin/2)**2:
                self.state.cells = build_cell(self.state.x, p.n_cells, p.cell_size)
                x_ref = self.state.x.copy()         
            

        self.total_time = time.time() - start_time
        self._print_completion()


    ###########################################################################
    ############################### PRIVATE ###################################
    ###########################################################################

    def _step(self):
        """Advances the system by one time step"""
        p = self.params
        s = self.state
        
        s.x, s.v = step(s.x, s.v, s.cells, p)
        s.t += 1
        s.t_sim = s.t * p.h 
    
    def _record(self):
        """Records the current state and energies."""
        p = self.params
        s = self.state

        self.states.append(s.x.copy())

        K, U_inter, U_wall, E_tot = energy(s.x, s.v, s.cells, p)
        self.energies.append([s.t_sim, K, U_inter, U_wall, E_tot])

    def _print_progress(self):
        """Prints the current progress to stdout."""
        pct = (self.state.t_sim / self.params.T_MAX) * 100
        print(f"Progress = {pct:.2f}%", end="\r", flush=True)

    def _print_completion(self):
        """Prints the total simulation time to stdout."""
        formatted = time.strftime("%H:%M:%S", time.gmtime(self.total_time))
        print(f"\nSimulation completed in {formatted}")


    ###########################################################################
    ################################# SAVE ####################################
    ###########################################################################

    def save(self, save_path, base_dir: str | Path = "runs"):
        """
        Saves states, energies and metadata to a timestamped subdirectory.

        Input:
            base_dir = root directory where run folders are created
        """
        if not self.states:
            raise RuntimeError("No data to save. Did you call run()?")

        states_arr   = np.array(self.states)
        energies_arr = np.array(self.energies)
        
        save_run(
            self.params, 
            save_path, 
            states_arr, 
            energies_arr, 
            self.total_time, 
            base_dir)
      