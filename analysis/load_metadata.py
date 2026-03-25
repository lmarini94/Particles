# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:20:15 2026

@author: marin
"""

import json
from pathlib import Path
import numpy as np

###############################################################################
############################# SELECTION OF RUN ################################
###############################################################################


def select_run(latest=True, runs_dir = "runs"):
    """
    Input:
        runs_dir = Name of the directory where runs are stored
        latest = option, if True use latest run, if not user selection
    Check the existence of a directory with name runs_dir in the parent folder
    and returns the path to either latest run or user selected run.
    """
    
    runs_dir = Path(__file__).resolve().parent.parent / runs_dir
    
    #Check if runs directory exists
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory '{runs_dir}' does not exist.")

    runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])

    if not runs:
        #If there are no runs in directory
        raise RuntimeError("No runs found in directory.")

    if latest:
        #Use latest run
        run_name = runs[-1]
        print(f"Using latest run: {run_name}")
    else:
        #Input manually run 
        run_name = input("Enter run timestamp (YYYYMMDD_HHMM): ")
    
        if run_name not in runs:
            raise ValueError(f"Run '{run_name}' not found in {runs_dir}")
        print(f"Using run: {run_name}")
    
    return runs_dir / run_name

def load_data(myrun, metadata = "run_parameters.json"):
    
    metadata_path = Path(myrun/metadata)
    
    #Check that CONFIG file exists. 
    if not metadata_path.exists():
        raise FileNotFoundError(f"Config file not found: {metadata}")
    
    with open(metadata_path) as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    #OUTPUTS
    states_path = metadata["parameters"]["outputs"]["states_file"]
    energies_path = metadata["parameters"]["outputs"]["energies_file"]
    states = np.load(myrun/states_path)
    energies = np.load(myrun/energies_path)
    
    
    return metadata, states, energies

