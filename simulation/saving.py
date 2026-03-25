# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:29:58 2026

@author: marin
"""

import json
import time
import numpy as np
from pathlib import Path

def save_run(p, outputs, states, energies, total_time, base_dir = "runs"):
    """
    Input:
        p = instance of SimParameters
        outputs = instance of SimOutputs
        states = (T, N, 2) array of recorded positions
        energies = (T, 5) array of recorded energies
        total_time = total simulation time in seconds
        base_dir = root directory where run folders are created
    This function saves states, energies and metadata to a timestamped 
    subdirectory of base_dir.
    """
    
    # CREATE TIMESTAMPED OUTPUT DIRECTORY
    timestamp = time.strftime("%Y%m%d_%H%M")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # SAVE DATA FILES
    
    np.save(run_dir / outputs.states_file, states)
    np.save(run_dir / outputs.energies_file, energies)
    
    # BUILD METADATA AND SAVE
    metadata = {
        "parameters": {
            "physical": {
                "N"      : p.N,
                "L"      : p.L,
                "d_min"  : p.d_min,
                "r_c"    : p.r_c,
                "K_0"    : p.K_0,
                "delta"  : p.delta,
                "k_wall" : p.k_wall,
            },
            "simulation": {
                "h"    : p.h,
                "eps"  : p.eps,
                "speed": p.speed,
                "T_MAX": p.T_MAX,
                "seed" : p.seed,
            },
            "outputs": {
                "states_file"  : outputs.states_file,
                "energies_file": outputs.energies_file,
            }
        },
        "Created_at"          : time.strftime("%Y-%m-%d %H:%M:%S"),
        "Created_at_unix"     : time.time(),
        "Simulation_duration" : total_time,
    }
    
    Path(run_dir / "run_parameters.json").write_text(json.dumps(metadata, indent=2))
    print(f"Results saved to: {run_dir}")
