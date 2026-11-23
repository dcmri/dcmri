from copy import deepcopy
from numbers import Real
import json
import csv
import numbers


import numpy as np

from dcmri.core.quantities import QUANTITIES, ROIS, COMPS, GROUPS
from dcmri.core.values import QDATA_ROIS
from dcmri.core.sequences import SEQUENCES

import dcmri.core.qtools as qtools


def get_sequence(prop, sequence=None):
    if prop=='name':
        return set(SEQUENCES.keys()) - {'3D-SPGR-SSI'}

    if prop=='steady-state':
        if sequence is None:
            return {s for s in SEQUENCES if SEQUENCES[s]['steady-state'] and s != '3D-SPGR-SSI'}
        return SEQUENCES[sequence]['steady-state']
                                        
    if prop=='mz_prep_inflow':
        return SEQUENCES[sequence]['mz_prep_inflow']

    if prop=='mz_prep_tissue':
        return SEQUENCES[sequence]['mz_prep_tissue']

    if prop=='prep_params':
        return SEQUENCES[sequence]['parameters']['prep']

    if prop=='read_params':
        return SEQUENCES[sequence]['parameters']['read']

    if prop=='tissue_params':
        return SEQUENCES[sequence]['parameters']['tissue']

    raise ValueError(f'{prop} is an unknown sequence property')



# OBSOLETE FUNCTIONS - Phase out

def unit(k: str, **kwargs):
    q = get_quantity(k, **kwargs)
    return q['unit']

def group(k: str, **kwargs):
    q = get_quantity(k, **kwargs)
    return q['group']

def get_quantity(*args, **kwargs):
    return qtools.qget(*args, **kwargs)

def build_varname(*args, **kwargs):
    return qtools.qkey(*args, **kwargs)

def parse_varname(*args, **kwargs):
    return qtools.qparse(*args, **kwargs)

def extend_varname(*args, **kwargs):
    return qtools.qmod(*args, **kwargs)

def increment_varindex(*args, **kwargs):
    return qtools.qincrement(*args, **kwargs)

def remove_varindex(*args, **kwargs):
    return qtools.qpopindex(*args, **kwargs)



def print_quantities(q: set | dict, title=None, decimals:int=None, digits:int=None, as_dict=False):

    if as_dict:
        _print_quantities_as_dict(q)
        return

    # Possible input formats:
    # q = set of keys
    # q = dict with key: value
    # q = dict with key: dict of quantity
    
    if isinstance(q, set):
        q = {k: get_quantity(k) for k in q}
    else:
        template = {}
        for k, v in q.items():
            if isinstance(v, dict):
                if 'name' in v: # v is full quantity description
                    template[k] = v
                    continue
            # v is value alone - read rest of description from dictionary
            template[k] = get_quantity(k)
            template[k]['init'] = v
        q = template

    if title is None:
        title = 'Quantities'

    def format_bounds(b):
        if b is None:
            return ""
        lower, upper = b
        return f"({lower}, {upper})"

    def format_optional(v):
        return "" if v is None else str(v)

    def round_sig(value, significant_digits):
        """Round a float to the given number of significant digits."""
        if significant_digits < 1:
            raise ValueError("significant_digits must be >= 1")
        return float(f"{value:.{significant_digits - 1}e}")

    def format_value(v, decimals=None, digits=None):
        init = v["init"]
        if isinstance(init, np.ndarray):
            return f"array {init.shape}"
        if isinstance(init, dict):
            return f"dict ({len(init)})"
        if isinstance(init, bool) or not isinstance(init, Real):
            return str(init)
        if decimals is not None:
            return str(round(init, decimals))
        if digits is not None:
            return str(round_sig(init, digits))
        return str(init)

    # Check values
    for k, v in q.items():
        if v is None:
            raise ValueError(f"Unknown quantity {k}.")

    # compute column widths from content
    values = {k: format_value(v, decimals, digits) for k, v in q.items()}

    key_w = max((len(k) for k in q), default=3)
    unit_w = max((len(format_optional(v["unit"])) for v in q.values()), default=4)
    name_w = max((len(v["name"]) for v in q.values()), default=4)
    group_w = max((len(label) for label in GROUPS), default=5)
    init_w = max((len(s) for s in values.values()), default=5)
    bounds_w = max((len(format_bounds(v["bounds"])) for v in q.values()), default=6)
    dicom_w = max((len(format_optional(v.get("dicom_key"))) for v in q.values()), default=5)
    osipi_w = max((len(format_optional(v.get("osipi_key"))) for v in q.values()), default=5)

    key_w = max(key_w, len("Key"))
    unit_w = max(unit_w, len("Unit"))
    name_w = max(name_w, len("Name"))
    group_w = max(group_w, len("Group"))
    init_w = max(init_w, len("Value"))
    bounds_w = max(bounds_w, len("Bounds"))
    dicom_w = max(dicom_w, len("DICOM"))
    osipi_w = max(osipi_w, len("OSIPI"))

    # 8 columns, each " x " padded (width+2), plus 9 "+" separators (before each col + trailing)
    table_width = key_w + unit_w + name_w + group_w + init_w + bounds_w + dicom_w + osipi_w + 25

    def row(key, init, unit, name, group, bounds, dicom, osipi):
        return (
            f"| {key:<{key_w}} | {init:<{init_w}} | {unit:<{unit_w}} "
            f"| {name:<{name_w}} | {group:<{group_w}} | {bounds:<{bounds_w}} "
            f"| {dicom:<{dicom_w}} | {osipi:<{osipi_w}} |"
        )

    def spanning_row(text):
        inner_width = table_width - 4  # account for "| " and " |"
        return f"| {text.center(inner_width)} |"

    divider = (
        "+" + "-" * (key_w + 2)
        + "+" + "-" * (init_w + 2)
        + "+" + "-" * (unit_w + 2)
        + "+" + "-" * (name_w + 2)
        + "+" + "-" * (group_w + 2)
        + "+" + "-" * (bounds_w + 2)
        + "+" + "-" * (dicom_w + 2)
        + "+" + "-" * (osipi_w + 2)
        + "+"
    )
    outer_border = "+" + "-" * (table_width - 2) + "+"

    lines = [outer_border]
    lines.append(spanning_row(title))
    lines.append(divider)
    lines.append(row("Key", "Value", "Unit", "Name", "Group", "Bounds", "DICOM", "OSIPI"))
    lines.append(divider)

    first_group = True
    # Groups in alphabetical order of their long name, ignoring case
    for gr in sorted(GROUPS):
        group = {k: v for k, v in q.items() if v["group"] == gr}
        if not group:
            continue
        # Keys within the group in alphabetical order, ignoring case
        group = dict(sorted(group.items(), key=lambda item: item[0].lower()))

        if not first_group:
            lines.append(divider)
        first_group = False

        for k, v in group.items():
            lines.append(row(
                k, values[k], format_optional(v["unit"]), v["name"], gr,
                format_bounds(v["bounds"]), format_optional(v.get("dicom_key")), format_optional(v.get("osipi_key")),
            ))

    lines.append(outer_border)
    lines.append("")

    for line in lines:
        print(line)   



def _plain(v):
    """Convert NumPy scalars and arrays to plain Python so repr() prints cleanly."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

def _print_quantities_as_dict(q, name="p"):
    """Print q as copy-paste code, grouped by the long group name.

    Groups are sorted alphabetically by long name, and parameters within each
    group are sorted alphabetically by key.

    q : dict mapping parameter name -> {'unit': ..., 'init': ..., 'group': <short key>}
    name : variable name used on the printed assignment line

    Uses the module-level GROUPS dict to translate short group keys into long names.
    """
    # Collect entries per long group name (a missing group goes under 'other')
    groups = {}
    for key, v in q.items():
        label = v.get('group') or 'other'    # fall back to the short key if not in GROUPS
        groups.setdefault(label, []).append((key, v))

    # Alphabetical order of the long names, and of the keys within each group (ignoring case)
    groups = {
        g: sorted(groups[g], key=lambda item: item[0].lower())
        for g in sorted(groups, key=str.lower)
    }

    # Build the text of each entry, so one width can be used for every comment
    entries = {
        g: [f"    {key!r}: {_plain(v['init'])!r}," for key, v in items]
        for g, items in groups.items()
    }
    width = max((len(line) for lines in entries.values() for line in lines), default=0)

    print(f"{name} = {{")
    for i, (g, items) in enumerate(groups.items()):
        if i > 0:
            print()
        print(f"    # {g}")
        for line, (_, v) in zip(entries[g], items):
            unit = v.get('unit')
            comment = f"  # {unit}" if unit else ""
            print(f"{line:<{width}}{comment}")
    print("}")



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






def _encode(obj):
    """Fallback for objects json can't serialise natively."""
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            data = {'real': obj.real.tolist(), 'imag': obj.imag.tolist()}
        else:
            data = obj.tolist()
        return {'__ndarray__': data, 'dtype': str(obj.dtype), 'shape': list(obj.shape)}
    if isinstance(obj, np.generic):  # numpy scalars such as np.float64
        return obj.item()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serialisable')

def _decode(d):
    """Rebuild arrays from the dicts produced by _encode."""
    if '__ndarray__' in d:
        data = d['__ndarray__']
        if isinstance(data, dict):  # complex array
            data = np.array(data['real']) + 1j * np.array(data['imag'])
        return np.array(data, dtype=d['dtype']).reshape(d['shape'])
    return d

def to_json(path, dct, indent=2):
    """Save a (nested) dict, which may contain numpy arrays, as a JSON file."""
    with open(path, 'w') as f:
        json.dump(dct, f, default=_encode, indent=indent)

def from_json(path):
    """Read a JSON file written by save_dict, restoring numpy arrays."""
    with open(path) as f:
        return json.load(f, object_hook=_decode)



def to_csv(path, dct):
    """Save a flat dict of scalar values as a two-column CSV (key, value)."""
    for key, val in dct.items():
        is_scalar = isinstance(val, (numbers.Number, str, bool, np.generic)) or val is None
        if not is_scalar:
            raise TypeError(
                f"Value for key '{key}' is of type {type(val).__name__}; "
                "only scalar values can be saved to CSV."
            )

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'value'])
        for key, val in dct.items():
            writer.writerow([key, val])

