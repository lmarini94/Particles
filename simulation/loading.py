# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:27:33 2026

@author: marin
"""

import json
import numpy as np
from pathlib import Path
from simulation.models import SimParameters, SimIO

def _validate(p: SimParameters):
    """
    Input:
        p = instance of SimParameters
    Validates the simulation parameters and raises ValueError if any parameter
    is invalid.
    """
    
    if p.N <= 0:
        raise ValueError(f"N must be positive, got {p.N}")
    if p.L <= 0:
        raise ValueError(f"L must be positive, got {p.L}")
    if p.d_min <= 0:
        raise ValueError(f"d_min must be positive, got {p.d_min}")
    if p.r_c <= 0:
        raise ValueError(f"r_c must be positive, got {p.r_c}")
    if p.r_c >= p.L:
        raise ValueError(f"r_c must be < L, got r_c={p.r_c}, L={p.L}")
    if p.K_0 <= 0:
        raise ValueError(f"K_target must be positive, got {p.K_0}")
    if p.delta <= 0:
        raise ValueError(f"delta must be positive, got {p.delta}")
    if p.delta >= p.L / 2:
        raise ValueError(f"delta must be < L/2, got delta={p.delta}, L/2={p.L/2}")
    if p.k_wall <= 0:
        raise ValueError(f"k_wall must be positive, got {p.k_wall}")

    # Simulation parameters
    if p.h <= 0:
        raise ValueError(f"h must be positive, got {p.h}")
    if p.eps < 0:
        raise ValueError(f"eps must be non-negative, got {p.eps}")
    if p.speed <= 0 or not isinstance(p.speed, int) :
        raise ValueError(f"speed must be positive integer, got {p.speed}")
    if p.T_MAX <= 0:
        raise ValueError(f"T_MAX must be positive, got {p.T_MAX}")

    # Cross-parameter checks
    if p.d_min >= p.L - 2 * p.delta:
        raise ValueError(
            f"d_min={p.d_min} is too large for box of effective size {p.L - 2*p.delta}"
        )

def load_config(path = "CONFIG.json"):
    """
    Input:
        path = Path to the JSON configuration file. Default is "CONFIG.json"
    Output:
        params = SimParameters dataclass with all simulation parameters
        output_path = SimIo dataclass with the path where to save files
        rng = numpy random number generator (seeded or unseeded)
    Loads and validates the configuration file populating an instance of 
    SimParameters, of SimIo and returning a numpy radom number generator. 
    """
    
    config_path = Path(path)
    
    #Check that CONFIG file exists. 
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
                
    #EXTRACT PARAMETERS
    try:
        _L = config["physical"]["L"]
        _r_c = config["physical"]["r_c"]
        _r_skin = config["physical"]["r_skin"]
        r_list = _r_c + _r_skin
        _n_cells = int(_L/r_list)
        _cell_size = _L/_n_cells
        params = SimParameters(
            #PHYSICAL
            N = config["physical"]["N"],                    
            L = _L,                  
            d_min = config["physical"]["d_min"], 
            r_c = _r_c,   
            r_skin = _r_skin,
            K_0 = config["physical"]["K_0"],       
            delta = config["physical"]["delta"],             
            k_wall = config["physical"]["k_wall"],
            n_cells = _n_cells,
            cell_size = _cell_size,
            #SIMULATION PARAMETERS
            h = config["simulation"]["h"],
            eps = config["simulation"]["eps"],
            speed = config["simulation"]["speed"],
            T_MAX = config["simulation"]["T_MAX"],
            seed = config["simulation"]["seed"]
            )
    except KeyError as e:
        raise KeyError((f"Missing required config field: {e}"))
    
    _validate(params)
    
    rng = np.random.default_rng(params.seed)
    #If "seed":null then seed = None
    
    #EXTRACT IO PAHTS
    try: 
        output_path = SimIO(
            states_file = config["outputs"]["states_file"],
            energies_file = config["outputs"]["energies_file"]
            )
    except KeyError as e:
        raise KeyError((f"Missing required config field: {e}"))
        
    return params, output_path, rng
