from copy import deepcopy

import numpy as np

from dcmri.lexicon.dicts import QUANTITIES

def init(pars:list=None, lexicon:dict=None, **kwargs) -> dict:
    """Return a dictionary with parameter initial values"""
    if lexicon is None:
        lexicon = QUANTITIES
    if pars is None:
        pars = lexicon.keys()
    
    p = {}
    for k in pars:
        p[k] = deepcopy(lexicon[k]['init']) 
    # Override with user-defined values
    for key, value in kwargs.items():
        if key in p:
            p[key] = value
    return p

def bounds(pars:list=None, lexicon:dict=None):
    """Return a dictionary with parameter bounds"""
    if lexicon is None:
        lexicon = QUANTITIES
    if pars is None:
        pars = lexicon.keys()
    p = {}
    for k in pars:
        p[k] = deepcopy(lexicon[k]['bounds']) 
    return p

# TODO: select is a better name
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

# Consider renaming. export_vals, string_vals
def export_params(val: dict, sdev=None, lexicon=None, num_only=False, scalar_only=False, group=None) -> dict:
    """Parameters with header information added."""
    if lexicon is None:
        lexicon = QUANTITIES
    if sdev is None:
        sdev = {}
    if group is not None:
        val = {k: v for k, v in val.items() if lexicon[k]['group']==group}
    if scalar_only:
        val = {k: v for k, v in val.items() if np.isscalar(v)}
    if num_only:
        val = {k: v for k, v in val.items() if not isinstance(v, str)}
    result = {}
    for p in val:
        result[p] = {
            'name': lexicon[p]['name'],
            'unit': lexicon[p]['unit'],
            'value': val[p],
            'sdev': sdev[p] if p in sdev else None,
        }
    return result

def string_params(val: dict, sdev=None, round_to=None, lexicon=None):
    """Print parameters and uncertainties to console."""
    if lexicon is None:
        lexicon = QUANTITIES
    if sdev is None:
        sdev = {}
    left_sides = {}
    pars = export_params(val, sdev, lexicon)
    for p, v in pars.items():

        # Format value
        val = v['value']

        if isinstance(val, list):
            val = f"list ({len(val)})"

        elif isinstance(val, np.ndarray):
            val = f"array {val.shape}"

        elif round_to is not None:
            if isinstance(val, (float, np.float64, np.float32)):
                val = round(val, round_to)

        # Format unit       
        unit = v['unit'] if v['unit'] is not None else ''

        # Format left side string
        if p in sdev:
            sd = v['sdev']
            if round_to is not None:
                sd = round(sd, round_to)
            left_sides[p] = f"{p} = {val} +/- {sd} {unit}"
        else:
            left_sides[p] = f"{p} = {val} {unit}"

    # Finding max string length for padding
    max_len = 4 + max(len(s) for s in left_sides.values()) 

    # Combine them, padding the left side to 'max_len' so brackets align
    strings = {}
    for p, v in pars.items():
        # '<' aligns left, and 'max_len' dynamically sets the width
        strings[p] = f"{left_sides[p]:<{max_len}} ({v['name']})"

    # Sort
    sorted_strings = {key: strings[key] for key in sorted(strings, key=str.lower)}

    return sorted_strings

def print_params(val: dict, sdev=None, round_to=None, group=None, lexicon=None):
    """Print parameters and uncertainties to console."""
    if lexicon is None:
        lexicon = QUANTITIES
    if group is not None:
        val = {k: v for k, v in val.items() if lexicon[k]['group']==group}

    groups = {
        'indicator': 'Indicator quantities',
        'signal': 'Signal quantities',
        'EM': 'Electromagnetic quantities',
        'phys': 'Physiological quantities',
        'hyper': 'Hyperparameter quantities',
    }
    for gr, label in groups.items():
        val_group = {k: v for k, v in val.items() if lexicon[k]['group']==gr}
        if len(val_group) > 0:
            msg = string_params(val_group, sdev, round_to, lexicon)
            print(f"\n{label}\n")
            for v in msg.values():
                print(v)


