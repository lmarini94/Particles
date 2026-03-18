# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:55:55 2026

@author: marin
"""

import time
import numpy as np
from pathlib import Path
from models import SimState, SimParameters
from cells import build_cell
from initialize import initialize_pos, initialize_vel
from integrator import step
from observables import energy
from io import save_run



###############################################################################
############################# SIMULATION CLASS ################################
###############################################################################

class Simulation:
    """
    Wraps the full lifecycle of a simulation run:
        1. initialize()  — set up positions, velocities, cell lists
        2. run()         — evolve the system up to T_MAX
        3. save()        — write states, energies, and metadata to disk
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
        Initializes positions, velocities and cell lists. Must be called
        before run().
        """
        p = self.params

        x = initialize_pos(p, self.rng)
        v = initialize_vel(p, self.rng)

        cells = build_cell(x, p)

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
        if self.state is None:
            raise RuntimeError("Call initialize() before run().")

        p = self.params
        start_time = time.time()

        while True:
            # Update simulation time at the TOP of the loop
            self.state.t_sim = self.state.t * p.h

            if self.state.t_sim > p.T_MAX:
                break

            if self.state.t % p.speed == 0:
                self._record()
                self._print_progress()

            self._step()

        self.total_time = time.time() - start_time
        self._print_completion()


    ###########################################################################
    ############################### PRIVATE ###################################
    ###########################################################################

    def _step(self):
        """Advances the system by one time step and rebuilds the cell list."""
        p = self.params
        s = self.state
        
        s.x, s.v = step(s, p)
        
        s.cells = build_cell(s.x, p)
        s.t += 1

    def _record(self):
        """Records the current state and energies."""
        p = self.params
        s = self.state

        self.states.append(s.x.copy())

        K, U_inter, U_wall, E_tot = energy(s, p)
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
      