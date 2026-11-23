from numbers import Real
from copy import deepcopy
import json
import csv
import numbers

import numpy as np

from dcmri.core.dictionaries import QUANTITIES, ROIS, COMPS, GROUPS
from dcmri.core.values import QDATA_ROIS

def all_quantities():
    """Return the registry of all quantities known to the package.

    Each quantity is described by a dictionary of metadata with the
    following keys:

    - ``init``: default initial value.
    - ``bounds``: (lower, upper) bounds for fitting, or None if the
      quantity is not fitted.
    - ``bounds_type``: only present for some quantities, e.g. 'add'
      for bounds that are applied relative to the initial value.
    - ``name``: descriptive name.
    - ``unit``: unit of measurement, or None if dimensionless or not
      numeric.
    - ``group``: category the quantity belongs to, e.g.
      'hyperparameters' or 'indicator'.
    - ``dicom_key``: corresponding DICOM keyword, or None.
    - ``osipi_key``: corresponding OSIPI code, or None.

    Returns:
        dict: A dictionary mapping quantity symbols (e.g. 'BAT',
        'dose') to their metadata dictionaries. The outer
        dictionary is a copy, so adding or removing quantities
        does not change the registry. The metadata dictionaries
        are shared with the registry, so modify them with care.

    Example:
        >>> quantities = all_quantities()
        >>> quantities['dt']['unit']
        'sec'
    """
    return dict(QUANTITIES)

def all_groups():
    """Return the names of all quantity groups.

    Each quantity in the registry belongs to one group, stored under
    its ``group`` key (see :func:`all_quantities`).

    Returns:
        set: The group names, e.g. 'indicator', 'signal', 'sequence',
        'electromagnetic', 'physiological', 'hyperparameters',
        'whole-body' and 'inverse problem'. The set is a copy, so
        modifying it does not change the registry.

    Example:
        >>> 'indicator' in all_groups()
        True
    """
    return set(GROUPS)

def all_tissues():
    """Return the notations for all tissue regions of interest (ROIs).

    Returns:
        dict: A dictionary mapping the short notation of each tissue
        ROI (e.g. 'ab', 'ao', 'gm') to its full name (e.g.
        'arterial blood', 'aorta', 'grey matter'). The dictionary
        is a copy, so modifying it does not change the registry.

    Example:
        >>> all_tissues()['gm']
        'grey matter'
    """
    return dict(ROIS)

def all_compartments():
    """Return the notations for all subvoxel compartments.

    Returns:
        dict: A dictionary mapping the short notation of each
        compartment (e.g. 'b', 'e', 'dt') to its full name (e.g.
        'blood', 'extracellular space', 'distal tubuli'). The
        dictionary is a copy, so modifying it does not change the
        registry.

    Example:
        >>> all_compartments()['b']
        'blood'
    """
    return dict(COMPS)

def quantity_unit(k, **kwargs):
    """Return the unit of a quantity.

    Args:
        k (str): Symbol of the quantity, e.g. 'dt' or 'BAT'.
        **kwargs: Additional keyword arguments, passed on to
            :func:`quantity_desc`.

    Returns:
        str or None: The unit of the quantity, e.g. 'sec' or 'mL/kg',
        or None if the quantity has no unit.

    Example:
        >>> quantity_unit('dt')
        'sec'
    """
    q = quantity_desc(k, **kwargs)
    return q['unit']

def quantity_group(k, **kwargs):
    """Return the group a quantity belongs to.

    Args:
        k (str): Symbol of the quantity, e.g. 'dt' or 'BAT'.
        **kwargs: Additional keyword arguments, passed on to
            :func:`quantity_desc`.

    Returns:
        str: The name of the group, e.g. 'hyperparameters' or
        'indicator'. See :func:`all_groups` for the possible
        values.

    Example:
        >>> quantity_group('dt')
        'hyperparameters'
    """
    q = quantity_desc(k, **kwargs)
    return q['group']

def quantity_osipi(k, **kwargs):
    """Return the OSIPI code of a quantity.

    The codes follow the OSIPI CAPLEX lexicon:
    https://osipi.github.io/OSIPI_CAPLEX/

    Args:
        k (str): Symbol of the quantity, e.g. 'BAT'.
        **kwargs: Additional keyword arguments, passed on to
            :func:`quantity_desc`.

    Returns:
        str or None: The OSIPI code of the quantity, e.g. 'Q.BA1.001',
        or None if the quantity has no OSIPI code.

    Example:
        >>> quantity_osipi('BAT')
        'Q.BA1.001'
    """
    q = quantity_desc(k, **kwargs)
    return q['osipi_key']


def quantity_dicom(k, **kwargs):
    """Return the DICOM keyword of a quantity.

    Args:
        k (str): Symbol of the quantity, e.g. 'dt'.
        **kwargs: Additional keyword arguments, passed on to
            :func:`quantity_desc`.

    Returns:
        str or None: The DICOM keyword of the quantity, or None if the
        quantity has no DICOM equivalent.

    Example:
        >>> quantity_dicom('dt') is None
        True
    """
    q = quantity_desc(k, **kwargs)
    return q['dicom_key']


def quantity_key(name, index=None, compartment=None, roi=None):
    """Build the key of a quantity from its name and optional qualifiers.

    The key is the name followed by the index, compartment and ROI, in
    that order, joined by underscores. Qualifiers that are None are
    left out.

    Args:
        name (str): Base name of the quantity, e.g. 'T1' or 'F'. This
            is not checked against the quantity registry.
        index (int or str, optional): Index to distinguish repeated
            quantities, e.g. 1 for the first of several. It is
            converted to a string. Defaults to None.
        compartment (str, optional): Notation of the subvoxel
            compartment the quantity refers to. Must be a key of the
            compartment registry (see :func:`all_compartments`), or a
            pair of such keys joined by '2' (for example 'b2e', for
            exchange from blood to the extracellular space). Defaults
            to None.
        roi (str, optional): Notation of the tissue region of interest
            the quantity refers to. Must be a key of the ROI registry
            (see :func:`all_tissues`). Defaults to None.

    Returns:
        str: The key, e.g. 'T1_e_gm'.

    Raises:
        ValueError: If `compartment` or `roi` is not valid.

    Example:
        >>> quantity_key('T1', compartment='e', roi='gm')
        'T1_e_gm'
        >>> quantity_key('F', index=1, compartment='b')
        'F_1_b'
    """
    if compartment is not None and not _is_valid_compartment(compartment):
        raise ValueError(
            f"'{compartment}' is not a valid compartment. "
            f"Must be one of {list(COMPS)}, or a pair 'comp1' + '2' + 'comp2' (e.g. 'b2e') with both in {list(COMPS)}."
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


def parse_quantity(k):
    """Split a quantity key into its name and qualifiers.

    This is the inverse of :func:`quantity_key`. The key is split at
    underscores and read from the end: a trailing ROI notation is
    taken first, then a compartment notation, then an integer index.
    Whatever remains is the name. Qualifiers that are not present are
    returned as None.

    Args:
        k (str): Quantity key, e.g. 'T1_e_gm'.

    Returns:
        dict: A dictionary with the following keys:

        - ``name`` (str): Base name of the quantity.
        - ``index`` (int or None): Index of the quantity.
        - ``compartment`` (str or None): Compartment notation.
        - ``roi`` (str or None): Tissue ROI notation.

    Example:
        >>> parse_quantity('T1_e_gm')
        {'name': 'T1', 'index': None, 'compartment': 'e', 'roi': 'gm'}
    """
    tokens = k.split("_")

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


def modify_quantity(k, index=None, compartment=None, roi=None):
    """Return a quantity key with some of its qualifiers changed.

    The key is parsed with :func:`parse_quantity`, the qualifiers that
    are provided replace the existing ones (or are added if the key
    does not have them yet), and the key is rebuilt with
    :func:`quantity_key`. Qualifiers that are None are left as they
    are in the original key.

    Args:
        k (str): Quantity key to modify, e.g. 'T1_e_gm'.
        index (int or str, optional): New index. Defaults to None,
            which keeps the current index.
        compartment (str, optional): New compartment notation, with
            the same restrictions as in :func:`quantity_key`. Defaults
            to None, which keeps the current compartment.
        roi (str, optional): New tissue ROI notation, with the same
            restrictions as in :func:`quantity_key`. Defaults to None,
            which keeps the current ROI.

    Returns:
        str: The modified key.

    Raises:
        ValueError: If `compartment` or `roi` is not valid.

    Example:
        >>> modify_quantity('T1_e_gm', compartment='b')
        'T1_b_gm'
        >>> modify_quantity('T1_e', roi='gm')
        'T1_e_gm'
    """
    parsed = parse_quantity(k)

    if index is not None:
        parsed['index'] = index
    if compartment is not None:
        parsed['compartment'] = compartment
    if roi is not None:
        parsed['roi'] = roi

    return quantity_key(
        parsed['name'],
        index=parsed['index'],
        compartment=parsed['compartment'],
        roi=parsed['roi'],
    )

def increment_quantity(k, increment=1):
    """Return a quantity key with its index increased.

    If the key has an index, it is increased by `increment`. If the
    key has no index, it is given the index 1, regardless of the value
    of `increment`.

    Args:
        k (str): Quantity key, e.g. 'F_1_b'.
        increment (int, optional): Amount to add to the existing
            index. Defaults to 1.

    Returns:
        str: The key with the new index.

    Raises:
        ValueError: If the compartment or ROI in the key is not valid
            (raised by :func:`modify_quantity`).

    Example:
        >>> increment_quantity('F_1_b')
        'F_2_b'
        >>> increment_quantity('F_1_b', increment=2)
        'F_3_b'
        >>> increment_quantity('F_b')
        'F_1_b'
    """
    parsed = parse_quantity(k)
    if parsed['index'] is None:
        new_index = 1
    else:
        new_index = parsed['index'] + increment
    return modify_quantity(k, index=new_index)

def popindex_quantity(k):
    """Return a quantity key with its index removed.

    The key is parsed with :func:`parse_quantity` and rebuilt with
    :func:`quantity_key` without the index. The name, compartment and
    ROI are kept. If the key has no index, it is returned unchanged.

    Args:
        k (str): Quantity key, e.g. 'F_1_b'.

    Returns:
        str: The key without the index.

    Raises:
        ValueError: If the compartment or ROI in the key is not valid
            (raised by :func:`quantity_key`).

    Example:
        >>> popindex_quantity('F_1_b')
        'F_b'
        >>> popindex_quantity('F_b')
        'F_b'
    """
    parsed = parse_quantity(k)
    return quantity_key(
        parsed['name'],
        compartment=parsed['compartment'],
        roi=parsed['roi'],
    )


def quantity_desc(k: str, quantities=None):
    """Return the description of a quantity.

    The key can be the name of a quantity in the registry (e.g. 'dt'),
    or a name with qualifiers added by :func:`quantity_key` (e.g.
    'T1_e_gm' or 'F_2_b'). For a qualified key, the description of the
    base quantity is copied, and its human-readable name is extended
    with the index, compartment and ROI. The initial value and bounds
    are replaced by ROI- or compartment-specific values where these
    are available.

    Args:
        k (str): Quantity key.
        quantities (dict, optional): Additional quantities, in the
            same format as the registry returned by
            :func:`all_quantities`. These are added to the registry
            for this call, and replace registry entries with the same
            key. Defaults to None.

    Returns:
        dict: A copy of the metadata dictionary of the quantity, with
        the keys described in :func:`all_quantities` (``init``,
        ``bounds``, ``name``, ``unit``, ``group``, ``dicom_key``,
        ``osipi_key`` and, for some quantities, ``bounds_type``).
        For a qualified key, ``name`` is the extended description
        (e.g. 'second bolus arrival time in the aorta'), and
        ``init`` and ``bounds`` may be specific to the ROI and
        compartment. The result is independent of the registry, so
        it can be modified freely.

    Raises:
        ValueError: If the base name of the key is not a known
            quantity.

    Example:
        >>> quantity_desc('dt')['unit']
        'sec'
    """
    lexicon = QUANTITIES
    if quantities is not None:
        lexicon |= quantities

    if k in lexicon:
        return deepcopy(lexicon[k])

    parsed = parse_quantity(k)
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
    data_key = quantity_key(name, compartment=compartment, roi=roi)
    if data_key in QDATA_ROIS:
        quant['init'] = QDATA_ROIS[data_key]['init']
        quant['bounds'] = QDATA_ROIS[data_key]['bounds']
    elif compartment is not None and roi is not None:
        fallback_key = quantity_key(name, compartment=compartment)
        if fallback_key in QDATA_ROIS:
            quant['init'] = QDATA_ROIS[fallback_key]['init']
            quant['bounds'] = QDATA_ROIS[fallback_key]['bounds']

    return quant


def print_quantities(q: set | dict, title=None, decimals:int=None, digits:int=None, as_dict=False):
    """Print a table of quantities with their descriptions.

    Args:
        q (set or dict): The quantities to print. A set holds quantity
            keys (see :func:`quantity_key`). A dict maps quantity keys
            to values.
        title (str, optional): Title printed above the table. Defaults
            to None, which prints no title.
        decimals (int, optional): Number of decimal places used to
            format numerical values. Defaults to None, which applies no
            rounding.
        digits (int, optional): Number of significant digits used to
            format numerical values. Defaults to None. It is an
            alternative to `decimals`.
        as_dict (bool, optional): If True, print the table contents
            as a dictionary instead of a table. Defaults to
            False.
    """
    # Possible input formats:
    # q = set of keys
    # q = dict with key: value
    # q = dict with key: dict of quantity
    
    if isinstance(q, set):
        q = {k: quantity_desc(k) for k in q}
    else:
        template = {}
        for k, v in q.items():
            if isinstance(v, dict):
                if 'name' in v: # v is full quantity description
                    template[k] = v
                    continue
            # v is value alone - read rest of description from dictionary
            template[k] = quantity_desc(k)
            template[k]['init'] = v
        q = template

    if as_dict:
        _print_quantities_as_dict(q)
        return

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

def quantity_to_json(path, dct, indent=2):
    """Save a (nested) dictionary as a JSON file.

    Numpy arrays and other values that JSON cannot represent directly
    are converted by the encoder, so that :func:`quantity_from_json`
    can restore them.

    Args:
        path (str or Path): Path of the JSON file to write. An existing
            file is overwritten.
        dct (dict): The dictionary to save. It may be nested and may
            contain numpy arrays.
        indent (int, optional): Number of spaces used to indent the
            file. Defaults to 2.

    Raises:
        TypeError: If the dictionary contains a value that neither JSON
            nor the encoder can handle.

    Example:
        >>> quantity_to_json('params.json', {'dt': 0.5, 'tmax': 240})
    """
    with open(path, 'w') as f:
        json.dump(dct, f, default=_encode, indent=indent)

def quantity_from_json(path):
    """Read a JSON file written by :func:`quantity_to_json`.

    Numpy arrays that were encoded when the file was written are
    restored as arrays.

    Args:
        path (str or Path): Path of the JSON file to read.

    Returns:
        dict: The dictionary stored in the file, with numpy arrays
        restored.

    Example:
        >>> quantity_from_json('params.json')
        {'dt': 0.5, 'tmax': 240}
    """
    with open(path) as f:
        return json.load(f, object_hook=_decode)

def quantity_to_csv(path, dct):
    """Save a flat dictionary of scalar values as a two-column CSV file.

    The file has a header row ``key,value`` followed by one row per
    item of the dictionary.

    Args:
        path (str or Path): Path of the CSV file to write. An existing
            file is overwritten.
        dct (dict): A flat dictionary. All values must be scalars:
            numbers, strings, booleans or None.

    Raises:
        TypeError: If any value is not a scalar, for example a list, a
            dictionary or a numpy array. The check is done before the
            file is opened, so nothing is written in that case.

    Example:
        >>> quantity_to_csv('params.csv', {'dt': 0.5, 'tmax': 240})
    """
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