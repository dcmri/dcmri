# TODO integrate with lexicon.py into a single namespace

from copy import deepcopy

from dcmri.lexicon import LEXICON

def init(pars:list=None, lexicon:dict=LEXICON, **kwargs) -> dict:
    """Return a dictionary with parameter initial values"""
    if pars is None:
        pars = lexicon.keys()
    p = {
        p: deepcopy(lexicon[p]['init']) 
        for p in pars
    }
    # Override with user-defined values
    for key, value in kwargs.items():
        if key in p:
            p[key] = value
    return p

def bounds(pars:list=None, lexicon:dict=LEXICON):
    """Return a dictionary with parameter bounds"""
    if pars is None:
        pars = lexicon.keys()
    return {
        p: deepcopy(lexicon[p]['bounds']) 
        for p in pars
    }

def export_params(val: dict, sdev=None, lexicon=LEXICON) -> dict:
    """Parameters with header information added."""
    if sdev is None:
        sdev = {}
    result = {}
    for p in val:
        result[p] = {
            'name': lexicon[p]['name'],
            'unit': lexicon[p]['unit'],
            'value': val[p],
            'sdev': sdev[p] if p in sdev else None,
        }
    return result

def string_params(val: dict, sdev=None, round_to=None, lexicon=LEXICON):
    """Print parameters and uncertainties to console."""
    if sdev is None:
        sdev = {}
    strings = {}
    pars = export_params(val, sdev, lexicon)
    for p, v in pars.items():
        val = v['value']
        if round_to is not None:
            val = round(val, round_to)
        if p in sdev:
            sd = v['sdev']
            if round_to is not None:
                sd = round(sd, round_to)
            strings[p] = f"{v['name']} ({p}) = {val} +/- {sd} {v['unit']}"
        else:
            strings[p] = f"{v['name']} ({p}) = {val} {v['unit']}"
    return strings

def print_params(val: dict, sdev=None, round_to=None, lexicon=LEXICON):
    """Print parameters and uncertainties to console."""
    msg = string_params(val, sdev, round_to, lexicon)
    for v in msg.values():
        print(v)


def select_params(lexicon=LEXICON, **kwargs):
    """Return lexicon parameters with specific properties"""
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