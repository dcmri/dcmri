from copy import deepcopy

import numpy as np

from dcmri.core.dictionaries import QUANTITIES # OBSOLETE
import dcmri.core.quantity_tools as quantity_tools
import dcmri.core.pseq_tools as pseq_tools


# OBSOLETE FUNCTIONS - Phase out
def get_sequence(*args, **kwargs):
    return pseq_tools.get_sequence(*args, **kwargs)

def unit(*args, **kwargs):
    return quantity_tools.quantity_unit(*args, **kwargs)

def group(*args, **kwargs):
    return quantity_tools.quantity_group(*args, **kwargs)

def get_quantity(*args, **kwargs):
    return quantity_tools.quantity_desc(*args, **kwargs)

def build_varname(*args, **kwargs):
    return quantity_tools.quantity_key(*args, **kwargs)

def parse_varname(*args, **kwargs):
    return quantity_tools.parse_quantity(*args, **kwargs)

def extend_varname(*args, **kwargs):
    return quantity_tools.modify_quantity(*args, **kwargs)

def increment_varindex(*args, **kwargs):
    return quantity_tools.increment_quantity(*args, **kwargs)

def remove_varindex(*args, **kwargs):
    return quantity_tools.popindex_quantity(*args, **kwargs)

def print_quantities(*args, **kwargs):
    return quantity_tools.print_quantities(*args, **kwargs)

def to_json(*args, **kwargs):
    return quantity_tools.quantity_to_json(*args, **kwargs)

def from_json(*args, **kwargs):
    return quantity_tools.quantity_from_json(*args, **kwargs)

def to_csv(*args, **kwargs):
    return quantity_tools.quantity_to_csv(*args, **kwargs)



def get_bounds(free:dict=None, bounds: dict=None, quantities:dict=None, free_pars:set=None):
    if free is None:
        free = {p: get_quantity(p, quantities)['bounds'] for p in free_pars}

    # --- 1. Update Bounds ---
    if bounds is not None:
        for p, b in bounds.items():
            if b is None:
                free.pop(p, None)
            else:
                free[p] = b

    return free

def update_bounds(free, quantities:dict=None, value:dict=None, all=True):
    
    # --- 2. Boundary Validation ---
    lexicon = QUANTITIES
    if quantities is not None:
        lexicon |= quantities

    bounds_add = select_params(lexicon, bounds_type='add')
    bounds_mult = select_params(lexicon, bounds_type='mult')

    new_free = deepcopy(free)

    for p, bnds in free.items():
        var = parse_varname(p)['name']
        if p not in value:
            if all:
                raise ValueError(
                    f"'{p}' is not a valid parameter for this configuration.\n"
                    f"Use print_params() to print a list of all valid parameters."
                )
            new_free.pop(p)

        elif var in bounds_add:
            if (bnds[0] > 0) or (bnds[1] < 0):
                raise ValueError(f"Bounds on {p} must be (negative, positive).")
            new_free[p] = [  
                np.min(value[p]) + bnds[0],
                np.max(value[p]) + bnds[1],
            ]
            
        elif var in bounds_mult: 
            if not (0 <= bnds[0] < bnds[1]):
                raise ValueError(f"Invalid bounds on {p}: Bounds are relative and must be positive.")
            new_free[p] = [
                np.min(value[p]) * bnds[0],
                np.max(value[p]) * bnds[1],
            ]

        elif not (bnds[0] <= np.min(value[p]) <= np.max(value[p]) <= bnds[1]):
            raise ValueError(f"Initial {p} is out of bounds {bnds}.")


    return new_free


def select_params(lexicon=None, **kwargs):
    """Return lexicon parameters with specific properties"""
    if lexicon is None:
        lexicon = QUANTITIES
    result = {}
    for p in lexicon:
        select_p = True
        for key, val in kwargs.items():
            if key not in lexicon[p]:
                select_p = False
            elif lexicon[p][key] != val:
                select_p = False
        if select_p:
            result[p] = deepcopy(lexicon[p])
    
    return result




