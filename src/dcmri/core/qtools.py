from numbers import Real
from copy import deepcopy

import numpy as np

from dcmri.core.quantities import QUANTITIES, ROIS, COMPS, GROUPS
from dcmri.core.values import QDATA_ROIS


def qkey(name, index=None, compartment=None, roi=None):
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


def qparse(key):
    tokens = key.split("_")

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


def qmod(key, index=None, compartment=None, roi=None):
    parsed = qparse(key)

    if index is not None:
        parsed['index'] = index
    if compartment is not None:
        parsed['compartment'] = compartment
    if roi is not None:
        parsed['roi'] = roi

    return qkey(
        parsed['name'],
        index=parsed['index'],
        compartment=parsed['compartment'],
        roi=parsed['roi'],
    )

def qincrement(key, increment=1):
    parsed = qparse(key)
    if parsed['index'] is None:
        new_index = 1
    else:
        new_index = parsed['index'] + increment
    return qmod(key, index=new_index)

def qpopindex(key):
    parsed = qparse(key)
    return qkey(
        parsed['name'],
        compartment=parsed['compartment'],
        roi=parsed['roi'],
    )


def qget(k: str, quantities=None):
    lexicon = QUANTITIES
    if quantities is not None:
        lexicon |= quantities

    if k in lexicon:
        return deepcopy(lexicon[k])

    parsed = qparse(k)
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
    data_key = qkey(name, compartment=compartment, roi=roi)
    if data_key in QDATA_ROIS:
        quant['init'] = QDATA_ROIS[data_key]['init']
        quant['bounds'] = QDATA_ROIS[data_key]['bounds']
    elif compartment is not None and roi is not None:
        fallback_key = qkey(name, compartment=compartment)
        if fallback_key in QDATA_ROIS:
            quant['init'] = QDATA_ROIS[fallback_key]['init']
            quant['bounds'] = QDATA_ROIS[fallback_key]['bounds']

    return quant


def qprint(q: set | dict, title=None, decimals:int=None, digits:int=None, as_dict=False):

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

def _is_valid_compartment(token):
    if token in COMPS:
        return True
    if "2" in token:
        comp1, _, comp2 = token.partition("2")
        return comp1 in COMPS and comp2 in COMPS
    return False


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