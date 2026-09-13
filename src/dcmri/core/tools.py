from copy import deepcopy

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



# def _get_quantity(k: str, quantities=None):
#     lexicon = QUANTITIES
#     if quantities is not None:
#         lexicon |= quantities

#     if k in lexicon:
#         return lexicon[k]

#     # Possible formats
#     # T_roi
#     # T_comp
#     # T_comp_roi
    
#     k_split = k.split('_')

#     if len(k_split) == 2:
#         k_generic = k_split[0]

#         # T_roi
#         if k_split[1] in ROIS:
#             roi = ROIS[k_split[1]]
#             if k_generic in lexicon:
#                 quant = deepcopy(lexicon[k_generic])
#                 quant['name'] = f"{roi} {quant['name']}"
#                 if k in QDATA_ROIS:
#                     quant['init'] = QDATA_ROIS[k]['init']
#                     quant['bounds'] = QDATA_ROIS[k]['bounds']
#                 return quant

#         # T_comp
#         elif k_split[1] in COMPS:
#             comp = COMPS[k_split[1]]
#             if k_generic in lexicon:
#                 quant = deepcopy(lexicon[k_generic])
#                 quant['name'] = f"{comp} {quant['name']}"
#                 if k in QDATA_ROIS:
#                     quant['init'] = QDATA_ROIS[k]['init']
#                     quant['bounds'] = QDATA_ROIS[k]['bounds']
#                 return quant

#         # T_comp2comp
#         elif '2' in k_split[1]:
#             k_exch = k_split[1].split('2')
#             if len(k_exch) == 2:
#                 comp1 = COMPS[k_exch[0]]
#                 comp2 = COMPS[k_exch[1]]
#                 if k_generic in lexicon:
#                     quant = deepcopy(lexicon[k_generic])
#                     quant['name'] = f"{comp1}-to-{comp2} {quant['name']}"
#                     if k in QDATA_ROIS:
#                         # T_comp2comp
#                         quant['init'] = QDATA_ROIS[k]['init']
#                         quant['bounds'] = QDATA_ROIS[k]['bounds']
#                     return quant

#     elif len(k_split) == 3:
#         k_generic = k_split[0]

#         if k_split[2] in ROIS:
#             roi = ROIS[k_split[2]]

#             # T_comp_roi
#             if k_split[1] in COMPS:
#                 comp = COMPS[k_split[1]]
#                 if k_generic in lexicon:
#                     quant = deepcopy(lexicon[k_generic])
#                     quant['name'] = f"{roi} {comp} {quant['name']}"
#                     if k in QDATA_ROIS:
#                         # T_comp_roi values
#                         quant['init'] = QDATA_ROIS[k]['init']
#                         quant['bounds'] = QDATA_ROIS[k]['bounds']
#                     else:
#                         # T_comp values
#                         k = f"{k_split[0]}_{k_split[1]}"
#                         if k in QDATA_ROIS:
#                             quant['init'] = QDATA_ROIS[k]['init']
#                             quant['bounds'] = QDATA_ROIS[k]['bounds']
#                     return quant

#             # T_comp2comp_roi
#             elif '2' in k_split[1]:
#                 k_exch = k_split[1].split('2')
#                 if len(k_exch) == 2:
#                     comp1 = COMPS[k_exch[0]]
#                     comp2 = COMPS[k_exch[1]]
#                     if k_generic in lexicon:
#                         quant = deepcopy(lexicon[k_generic])
#                         quant['name'] = f"{roi} {comp1}-to-{comp2} {quant['name']}"
#                         if k in QDATA_ROIS:
#                             # T_comp2comp_roi
#                             quant['init'] = QDATA_ROIS[k]['init']
#                             quant['bounds'] = QDATA_ROIS[k]['bounds']
#                         else:
#                             # T_comp2comp values
#                             k = f"{k_split[0]}_{k_exch[0]}2{k_exch[1]}_{k_split[2]}"
#                             if k in QDATA_ROIS:
#                                 quant['init'] = QDATA_ROIS[k]['init']
#                                 quant['bounds'] = QDATA_ROIS[k]['bounds']
#                         return quant

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
        return lexicon[k]

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


# def parse_varname(varname):
#     tokens = varname.split("_")

#     roi = None
#     if tokens and tokens[-1] in ROIS:
#         roi = tokens.pop()

#     compartment = None
#     if tokens and _is_valid_compartment(tokens[-1]):
#         compartment = tokens.pop()

#     index = None
#     if tokens and tokens[-1].isdigit():
#         index = int(tokens.pop())

#     name = "_".join(tokens)

#     return {
#         "name": name, # TODO: rename to key. name is used to refer to full description
#         "index": index,
#         "compartment": compartment,
#         "roi": roi,
#     }

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


def print_quantities(title, q):

    def format_bounds(b):
        if b is None:
            return ""
        lower, upper = b
        return f"({lower}, {upper})"

    def format_optional(v):
        return "" if v is None else str(v)

    # Check values
    for k, v in q.items():
        if v is None:
            raise ValueError(f"Unknown quantity {k}.")

    # compute column widths from content
    key_w = max((len(k) for k in q), default=3)
    unit_w = max((len(format_optional(v["unit"])) for v in q.values()), default=4)
    name_w = max((len(v["name"]) for v in q.values()), default=4)
    group_w = max((len(label) for label in GROUPS.values()), default=5)
    init_w = max((len(str(v["init"])) for v in q.values()), default=4)
    bounds_w = max((len(format_bounds(v["bounds"])) for v in q.values()), default=6)
    dicom_w = max((len(format_optional(v.get("dicom_key"))) for v in q.values()), default=5)
    osipi_w = max((len(format_optional(v.get("osipi_key"))) for v in q.values()), default=5)

    key_w = max(key_w, len("Key"))
    unit_w = max(unit_w, len("Unit"))
    name_w = max(name_w, len("Name"))
    group_w = max(group_w, len("Group"))
    init_w = max(init_w, len("Init"))
    bounds_w = max(bounds_w, len("Bounds"))
    dicom_w = max(dicom_w, len("DICOM"))
    osipi_w = max(osipi_w, len("OSIPI"))

    # 8 columns, each " x " padded (width+2), plus 9 "+" separators (before each col + trailing)
    table_width = key_w + unit_w + name_w + group_w + init_w + bounds_w + dicom_w + osipi_w + 25

    def row(key, unit, name, group, init, bounds, dicom, osipi):
        return (
            f"| {key:<{key_w}} | {unit:<{unit_w}} | {name:<{name_w}} "
            f"| {group:<{group_w}} | {init:<{init_w}} | {bounds:<{bounds_w}} "
            f"| {dicom:<{dicom_w}} | {osipi:<{osipi_w}} |"
        )

    def spanning_row(text):
        inner_width = table_width - 4  # account for "| " and " |"
        return f"| {text.center(inner_width)} |"

    divider = (
        "+" + "-" * (key_w + 2)
        + "+" + "-" * (unit_w + 2)
        + "+" + "-" * (name_w + 2)
        + "+" + "-" * (group_w + 2)
        + "+" + "-" * (init_w + 2)
        + "+" + "-" * (bounds_w + 2)
        + "+" + "-" * (dicom_w + 2)
        + "+" + "-" * (osipi_w + 2)
        + "+"
    )
    outer_border = "+" + "-" * (table_width - 2) + "+"

    lines = [outer_border]
    lines.append(spanning_row(title))
    lines.append(divider)
    lines.append(row("Key", "Unit", "Name", "Group", "Init", "Bounds", "DICOM", "OSIPI"))
    lines.append(divider)

    first_group = True
    for gr, label in GROUPS.items():
        group = {k: v for k, v in q.items() if v["group"] == gr}
        if not group:
            continue
        group = dict(sorted(group.items(), key=lambda item: item[0]))
        # group = dict(sorted(group.items(), key=lambda item: item[0].lower()))

        if not first_group:
            lines.append(divider)
        first_group = False

        for k, v in group.items():
            lines.append(row(
                k, format_optional(v["unit"]), v["name"], label, str(v["init"]), format_bounds(v["bounds"]),
                format_optional(v.get("dicom_key")), format_optional(v.get("osipi_key")),
            ))

    lines.append(outer_border)
    lines.append("")

    for line in lines:
        print(line)    



# Obsolete below here



def init(pars:list=None, lexicon:dict=None, **kwargs) -> dict:
    """Return a dictionary with parameter initial values"""
    if lexicon is None:
        lexicon = QUANTITIES
    if pars is None:
        pars = lexicon.keys()
    
    p = {}
    for k in pars:
        p[k] = deepcopy(lexicon[k]['init']) 
        # p[k] = lexicon[k]['init'] 
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
        # p[k] = lexicon[k]['bounds']
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


