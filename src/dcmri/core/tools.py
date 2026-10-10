from copy import deepcopy
from numbers import Real

import numpy as np

from dcmri.core.quantities import QUANTITIES, ROIS, COMPS, GROUPS
from dcmri.core.values import QDATA_ROIS
from dcmri.core.sequences import SEQUENCES


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




def _ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def _describe_compartment(compartment):
    if '2' in compartment:
        comp1, comp2 = compartment.split('2')
        return f"from {COMPS[comp1]} to {COMPS[comp2]}"
    return f"in {COMPS[compartment]}"


def get_quantity(k: str, quantities=None):
    lexicon = QUANTITIES
    if quantities is not None:
        lexicon |= quantities

    if k in lexicon:
        return deepcopy(lexicon[k])

    parsed = parse_varname(k)
    name, index, compartment, roi = (
        parsed['name'], parsed['index'], parsed['compartment'], parsed['roi']
    )

    if name not in lexicon:
        raise ValueError(f"Unknown quantity {k}")

    quant = deepcopy(lexicon[name])

    # Build the human-readable description
    desc = quant['name']
    if compartment:
        desc = f"{desc} {_describe_compartment(compartment)}"
    if roi:
        desc = f"{desc} {'of' if compartment else 'in'} the {ROIS[roi]}"
    if index is not None:
        desc = f"{_ordinal(index)} {desc}"
    quant['name'] = desc

    # Override init/bounds with any roi/compartment-specific data (index doesn't affect data)
    data_key = build_varname(name, compartment=compartment, roi=roi)
    if data_key in QDATA_ROIS:
        quant['init'] = QDATA_ROIS[data_key]['init']
        quant['bounds'] = QDATA_ROIS[data_key]['bounds']
    elif compartment is not None and roi is not None:
        fallback_key = build_varname(name, compartment=compartment)
        if fallback_key in QDATA_ROIS:
            quant['init'] = QDATA_ROIS[fallback_key]['init']
            quant['bounds'] = QDATA_ROIS[fallback_key]['bounds']

    return quant


def _is_valid_compartment(token):
    if token in COMPS:
        return True
    if "2" in token:
        comp1, _, comp2 = token.partition("2")
        return comp1 in COMPS and comp2 in COMPS
    return False


def build_varname(name, index=None, compartment=None, roi=None):
    if compartment is not None and not _is_valid_compartment(compartment):
        raise ValueError(
            f"'{compartment}' is not a valid compartment. "
            f"Must be one of {list(COMPS)}, or 'comp1_2_comp2' with both in {list(COMPS)}."
        )
    if roi is not None and roi not in ROIS:
        raise ValueError(f"'{roi}' is not a valid roi. Options are {list(ROIS)}.")

    parts = [name]
    if index is not None:
        parts.append(str(index))
    if compartment is not None:
        parts.append(compartment)
    if roi is not None:
        parts.append(roi)

    return "_".join(parts)


def parse_varname(varname):
    tokens = varname.split("_")

    roi = None
    if len(tokens) > 1 and tokens[-1] in ROIS:
        roi = tokens.pop()

    compartment = None
    if len(tokens) > 1 and _is_valid_compartment(tokens[-1]):
        compartment = tokens.pop()

    index = None
    if len(tokens) > 1 and tokens[-1].isdigit():
        index = int(tokens.pop())

    name = "_".join(tokens)

    return {
        "name": name,  # TODO: rename to key. name is used to refer to full description
        "index": index,
        "compartment": compartment,
        "roi": roi,
    }


def extend_varname(var, index=None, compartment=None, roi=None):
    parsed = parse_varname(var)

    if index is not None:
        parsed['index'] = index
    if compartment is not None:
        parsed['compartment'] = compartment
    if roi is not None:
        parsed['roi'] = roi

    return build_varname(
        parsed['name'],
        index=parsed['index'],
        compartment=parsed['compartment'],
        roi=parsed['roi'],
    )

def increment_varindex(var, increment=1):
    parsed = parse_varname(var)
    if parsed['index'] is None:
        new_index = 1
    else:
        new_index = parsed['index'] + increment
    return extend_varname(var, index=new_index)


def remove_varindex(var):
    parsed = parse_varname(var)
    return build_varname(
        parsed['name'],
        compartment=parsed['compartment'],
        roi=parsed['roi'],
    )



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
    group_w = max((len(label) for label in GROUPS.values()), default=5)
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
    for gr, label in sorted(GROUPS.items(), key=lambda item: item[1].lower()):
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
                k, values[k], format_optional(v["unit"]), v["name"], label,
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
        short = v.get('group') or 'other'
        label = GROUPS.get(short, short)    # fall back to the short key if not in GROUPS
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

def update_bounds(free, quantities:dict=None, value:dict=None):
    
    # --- 2. Boundary Validation ---
    lexicon = QUANTITIES
    if quantities is not None:
        lexicon |= quantities

    bounds_add = select_params(lexicon, bounds_type='add')
    bounds_mult = select_params(lexicon, bounds_type='mult')

    for p, bnds in free.items():
        var = parse_varname(p)['name']
        if p not in value:
            raise ValueError(
                f"'{p}' is not a valid parameter for this configuration.\n"
                f"Use print_params() to print a list of all valid parameters."
            )

        elif var in bounds_add:
            if (bnds[0] > 0) or (bnds[1] < 0):
                raise ValueError(f"Bounds on {p} must be (negative, positive).")
            free[p] = [  
                np.min(value[p]) + bnds[0],
                np.max(value[p]) + bnds[1],
            ]
            
        elif var in bounds_mult: 
            if not (0 <= bnds[0] < bnds[1]):
                raise ValueError(f"Invalid bounds on {p}: Bounds are relative and must be positive.")
            free[p] = [
                np.min(value[p]) * bnds[0],
                np.max(value[p]) * bnds[1],
            ]

        elif not (bnds[0] <= np.min(value[p]) <= np.max(value[p]) <= bnds[1]):
            raise ValueError(f"Initial {p} is out of bounds {bnds}.")


    return free


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


# # Obsolete below here



# def init(pars:list=None, lexicon:dict=None, **kwargs) -> dict:
#     """Return a dictionary with parameter initial values"""
#     if lexicon is None:
#         lexicon = QUANTITIES
#     if pars is None:
#         pars = lexicon.keys()
    
#     p = {}
#     for k in pars:
#         p[k] = deepcopy(lexicon[k]['init']) 
#         # p[k] = lexicon[k]['init'] 
#     # Override with user-defined values
#     for key, value in kwargs.items():
#         if key in p:
#             p[key] = value
#     return p

# def bounds(pars:list=None, lexicon:dict=None):
#     """Return a dictionary with parameter bounds"""
#     if lexicon is None:
#         lexicon = QUANTITIES
#     if pars is None:
#         pars = lexicon.keys()
#     p = {}
#     for k in pars:
#         # p[k] = lexicon[k]['bounds']
#         p[k] = deepcopy(lexicon[k]['bounds']) 
#     return p



# # Consider renaming. export_vals, string_vals
# def export_params(val: dict, sdev=None, lexicon=None, num_only=False, scalar_only=False, group=None) -> dict:
#     """Parameters with header information added."""
#     if lexicon is None:
#         lexicon = QUANTITIES
#     if sdev is None:
#         sdev = {}
#     if group is not None:
#         val = {k: v for k, v in val.items() if lexicon[k]['group']==group}
#     if scalar_only:
#         val = {k: v for k, v in val.items() if np.isscalar(v)}
#     if num_only:
#         val = {k: v for k, v in val.items() if not isinstance(v, str)}
#     result = {}
#     for p in val:
#         result[p] = {
#             'name': lexicon[p]['name'],
#             'unit': lexicon[p]['unit'],
#             'value': val[p],
#             'sdev': sdev[p] if p in sdev else None,
#         }
#     return result

# def string_params(val: dict, sdev=None, round_to=None, lexicon=None):
#     """Print parameters and uncertainties to console."""
#     if lexicon is None:
#         lexicon = QUANTITIES
#     if sdev is None:
#         sdev = {}
#     left_sides = {}
#     pars = export_params(val, sdev, lexicon)
#     for p, v in pars.items():

#         # Format value
#         val = v['value']

#         if isinstance(val, list):
#             val = f"list ({len(val)})"

#         elif isinstance(val, np.ndarray):
#             val = f"array {val.shape}"

#         elif round_to is not None:
#             if isinstance(val, (float, np.float64, np.float32)):
#                 val = round(val, round_to)

#         # Format unit       
#         unit = v['unit'] if v['unit'] is not None else ''

#         # Format left side string
#         if p in sdev:
#             sd = v['sdev']
#             if round_to is not None:
#                 sd = round(sd, round_to)
#             left_sides[p] = f"{p} = {val} +/- {sd} {unit}"
#         else:
#             left_sides[p] = f"{p} = {val} {unit}"

#     # Finding max string length for padding
#     max_len = 4 + max(len(s) for s in left_sides.values()) 

#     # Combine them, padding the left side to 'max_len' so brackets align
#     strings = {}
#     for p, v in pars.items():
#         # '<' aligns left, and 'max_len' dynamically sets the width
#         strings[p] = f"{left_sides[p]:<{max_len}} ({v['name']})"

#     # Sort
#     sorted_strings = {key: strings[key] for key in sorted(strings, key=str.lower)}

#     return sorted_strings

# def print_params(val: dict, sdev=None, round_to=None, group=None, lexicon=None):
#     """Print parameters and uncertainties to console."""
#     if lexicon is None:
#         lexicon = QUANTITIES
#     if group is not None:
#         val = {k: v for k, v in val.items() if lexicon[k]['group']==group}

#     groups = {
#         'indicator': 'Indicator quantities',
#         'signal': 'Signal quantities',
#         'EM': 'Electromagnetic quantities',
#         'phys': 'Physiological quantities',
#         'hyper': 'Hyperparameter quantities',
#     }
#     for gr, label in groups.items():
#         val_group = {k: v for k, v in val.items() if lexicon[k]['group']==gr}
#         if len(val_group) > 0:
#             msg = string_params(val_group, sdev, round_to, lexicon)
#             print(f"\n{label}\n")
#             for v in msg.values():
#                 print(v)


