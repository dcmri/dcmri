import dcmri.core.values as values
import dcmri.core.quantities as quantities


def ca_conc(agent: str) -> float:
    """Contrast agent concentration

    Args:
        agent (str): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.

    Raises:
        ValueError: If no data are available for the agent.

    Returns:
        float: concentration in mmol/mL

    Sources:
        - `mri questions <https://mriquestions.com/so-many-gd-agents.html>`_
        - `gadoxetate <https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf>`_
        - `gadobutrol <https://www.medicines.org.uk/emc/product/2876/smpc#gref>`_

    Example:

        Print the concentration of the agents gadobutrol and gadoterate:

    .. exec_code::

        import dcmri as dc

        print('gadobutrol is available in a solution of', dc.ca_conc('gadobutrol'), 'mmol/mL')
        print('gadoterate is available in a solution of', dc.ca_conc('gadoterate'), 'mmol/mL')
    """
    try:
        return values.CA_CONC[agent]
    except KeyError:
        raise ValueError(
            f"No concentration data for contrast agent {agent}. "
            f"Currently values are only available for {values.CA_CONC.keys()}. "
            f"Please extend the dictionary with literature values."
        ) from None # No traceback to KeyError shown


def ca_std_dose(agent: str) -> float:
    """Standard injection volume (dose) in mL per kg body weight.

    Args:
        agent (str): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.

    Raises:
        ValueError: If no data are available for the agent.

    Returns:
        float: Standard injection volume in mL/kg.

    Note:

        Available agents:
            - gadoxetate
            - gadobutrol
            - gadopiclenol
            - gadopentetate
            - gadobenate
            - gadodiamide
            - gadoterate
            - gadoteridol

        Sources:
            - `mri questions <https://mriquestions.com/so-many-gd-agents.html>`_
            - `gadoxetate <https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf>`_
            - `gadobutrol <https://www.medicines.org.uk/emc/product/2876/smpc#gref>`_

    Example:

        >>> import dcmri as dc
        >>> print('The standard clinical dose of gadobutrol is', dc.ca_std_dose('gadobutrol'), 'mL/kg')
        The standard clinical dose of gadobutrol is 0.1 mL/kg
    """
    # """Standard dose in mL/kg""" # better in mmol/kg, or offer it as an option
    try:
        return values.CA_DOSE[agent]
    except KeyError:
        raise ValueError(
            f"No data available for contrast agent {agent}."
            f"Currently values are only available for {values.CA_DOSE.keys()}. "
            f"Please extend the dictionary with literature values."
        ) from None



def relaxivity(field_strength=3.0, tissue='plasma', agent='gadoxetate') -> dict:
    """Contrast agent relaxivity values in units of Hz/M"""
    return {
        'r1': r1(field_strength, tissue, agent),
        'r2': r2(field_strength, tissue, agent),
        'r2s': r2s(field_strength, tissue, agent),
        'r2sq': 1e3,
    }

def r1(field_strength=3.0, tissue='plasma', agent='gadoxetate', force=False) -> float:
    """Longitudinal contrast agent relaxivity values in units of Hz/M

    Args:
        field_strength (float, optional): Field strength in Tesla. Defaults to 3.0.
        tissue (str, optional): Tissue type - options are 'plasma', 'hepatocytes'. Defaults to 'plasma'.
        agent (str, optional): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.. Defaults to 'gadoxetate'.

    Returns:
        float: relaxivity in Hz/M or 1/(sec*M)

    Note:

        This library is in construction and not all known values are currently available through this function.

        Available contrasts:
            - T1

        Available tissues:
            - blood
            - plasma
            - hepatocytes

        Available agents:
            - gadopentetate
            - gadobutrol
            - gadoteridol
            - gadobenade
            - gadoterate
            - gadodiamide
            - mangafodipir
            - gadoversetamide
            - ferucarbotran
            - ferumoxide
            - gadoxetate
            - gadopiclenol

        Available field strengths:
            - 0.47
            - 1.5
            - 3
            - 4.7
            - 7.0 (gadoxetate)
            - 9.0 (gadoxetate)

        Sources:
            - Rohrer M, et al. Comparison of Magnetic Properties of MRI Contrast Media Solutions at Different Magnetic Field Strengths. Investigative Radiology 40(11):p 715-724, November 2005. DOI: `10.1097/01.rli.0000184756.66360.d3 <https://journals.lww.com/investigativeradiology/FullText/2005/11000/Comparison_of_Magnetic_Properties_of_MRI_Contrast.5.aspx>`_
            - Szomolanyi P, et al. Comparison of the Relaxivities of Macrocyclic Gadolinium-Based Contrast Agents in Human Plasma at 1.5, 3, and 7 T, and Blood at 3 T. Invest Radiol. 2019 Sep;54(9):559-564. doi: `10.1097/RLI.0000000000000577 <https://pubmed.ncbi.nlm.nih.gov/31124800/>`_

    Example:

        >>> import dcmri as dc
        >>> print('The plasma relaxivity of gadobutrol at 3T is', 1e-3*dc.relaxivity(3.0, 'plasma', 'gadobutrol'), 'Hz/mM')
        The plasma relaxivity of gadobutrol at 3T is 5.0 Hz/mM
    """

    val = values.R1_RELAXIVITY
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in val else map.get(tissue)

    if tissue_key in val and agent in val[tissue_key] and field_strength in val[tissue_key][agent]:
        return 1000 * val[tissue_key][agent][field_strength]

    if force:
        raise ValueError(f'No r1 values for {agent} in {tissue} at {field_strength} T.')

    if agent in val['plasma'] and field_strength in val['plasma'][agent]:
        return 1000 * val['plasma'][agent][field_strength]

    raise ValueError(f"No plasma r1 values for agent {agent} at field strength {field_strength}")
    

def r2(field_strength=3.0, tissue='plasma', agent='gadoxetate', force=False) -> float:
    """Transverse contrast agent relaxivity values in units of Hz/M

    Args:
        field_strength (float, optional): Field strength in Tesla. Defaults to 3.0.
        tissue (str, optional): Tissue type - options are 'plasma', 'hepatocytes'. Defaults to 'plasma'.
        agent (str, optional): Generic contrast agent name, all lower case. Examples are 'gadobutrol', 'gadobenate', etc.. Defaults to 'gadoxetate'.

    Returns:
        float: relaxivity in Hz/M or 1/(sec*M)

    Note:

        This library is in construction and not all known values are currently available through this function.

        Available contrasts:
            - T1

        Available tissues:
            - blood
            - plasma
            - hepatocytes

        Available agents:
            - gadopentetate
            - gadobutrol
            - gadoteridol
            - gadobenade
            - gadoterate
            - gadodiamide
            - mangafodipir
            - gadoversetamide
            - ferucarbotran
            - ferumoxide
            - gadoxetate
            - gadopiclenol

        Available field strengths:
            - 0.47
            - 1.5
            - 3
            - 4.7
            - 7.0 (gadoxetate)
            - 9.0 (gadoxetate)

        Sources:
            - Rohrer M, et al. Comparison of Magnetic Properties of MRI Contrast Media Solutions at Different Magnetic Field Strengths. Investigative Radiology 40(11):p 715-724, November 2005. DOI: `10.1097/01.rli.0000184756.66360.d3 <https://journals.lww.com/investigativeradiology/FullText/2005/11000/Comparison_of_Magnetic_Properties_of_MRI_Contrast.5.aspx>`_
            - Szomolanyi P, et al. Comparison of the Relaxivities of Macrocyclic Gadolinium-Based Contrast Agents in Human Plasma at 1.5, 3, and 7 T, and Blood at 3 T. Invest Radiol. 2019 Sep;54(9):559-564. doi: `10.1097/RLI.0000000000000577 <https://pubmed.ncbi.nlm.nih.gov/31124800/>`_

    Example:

        >>> import dcmri as dc
        >>> print('The plasma relaxivity of gadobutrol at 3T is', 1e-3*dc.relaxivity(3.0, 'plasma', 'gadobutrol'), 'Hz/mM')
        The plasma relaxivity of gadobutrol at 3T is 5.0 Hz/mM
    """
    val = values.R2_RELAXIVITY
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in val else map.get(tissue)

    if tissue_key in val and agent in val[tissue_key] and field_strength in val[tissue_key][agent]:
        return 1000 * val[tissue_key][agent][field_strength]

    if force:
        raise ValueError(f'No r2 values for {agent} in {tissue} at {field_strength} T.')

    if agent in val['plasma'] and field_strength in val['plasma'][agent]:
        return 1000 * val['plasma'][agent][field_strength]

    raise ValueError(f"No plasma r2 values for agent {agent} at field strength {field_strength}")


def r2s(field_strength=3.0, tissue='blood', agent='gadoxetate', force=False) -> float:
    """R2*-relaxivity"""
    val = values.R2_STAR_RELAXIVITY
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in val else map.get(tissue)

    if tissue_key in val:
        return 1000 * val[tissue_key]

    if force:
        raise ValueError(f'No r2* values for {tissue}.')

    return 1000 * val['blood']


def r2sq(field_strength=3.0, tissue='blood', agent='gadoxetate', force=False) -> float:
    """R2*-relaxivity"""
    val = values.R2_STAR_RELAXIVITY_SQ
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in val else map.get(tissue)

    if tissue_key in val:
        return 1000 * val[tissue_key]

    if force:
        raise ValueError(f'No r2* values for {tissue}.')

    return 1000 * val['blood']


def T1(field_strength=3.0, tissue='blood', Hct=0.45, force=False) -> float:
    """T1 value of selected tissue types.

    Values are taken from literature, mostly from `Stanisz et al 2005 <https://doi.org/10.1002/mrm.20605>`_

    Args:
        field_strength (float, optional): Field strength in Tesla (see below for options). Defaults to 3.0.
        tissue (str, optional): Tissue type (see below for options). Defaults to 'blood'.
        Hct (float, optional): Hematocrit value - ignored when tissue is not blood. Defaults to 0.45.
        force: By default a blood value is returned for any compartments that don't have a dedicated value. 
            Set Force=True to override this behaviour and raise an exception instead.

    Raises:
        ValueError: If the requested T1 values are not available.

    Returns:
        float: T1 values in sec

    Note:

        This library is in construction and not all known values are currently available through this function.

        Available tissue types:
            - skin
            - bone marrow
            - csf
            - muscle
            - heart
            - cartilage
            - white matter
            - gray matter
            - optic nerve
            - spinal cord
            - blood
            - spleen
            - liver
            - kidney

        Available field strengths:
            - 1.5
            - 3
            - 4.7 (spleen, liver)
            - 7.0 (spleen, liver)
            - 9.0 (spleen, liver)

    Example:

        >>> import dcmri as dc
        >>> print('The T1 of liver at 1.5T is', 1e3*dc.const.T1(1.5, 'liver'), 'msec')
        The T1 of liver at 1.5T is 602.0 msec
    """
    # H. M. Gach, C. Tanase and F. Boada, "2D & 3D Shepp-Logan Phantom
    # Standards for MRI," 2008 19th International Conference on Systems
    # Engineering, Las Vegas, NV, USA, 2008, pp. 521-526, doi:
    # 10.1109/ICSEng.2008.15

    if field_strength==3 and tissue=='blood':
        return 1 / (0.52 * Hct + 0.38)  # Lu MRM 2004

    T1VAL = values.T1_RELAXATION_TIMES
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in T1VAL else map.get(tissue)

    if tissue_key in T1VAL and field_strength in T1VAL[tissue_key]:
        return T1VAL[tissue_key][field_strength]

    if force:
        raise ValueError(f'No T1 values for {tissue} at {field_strength} T.')

    if field_strength in T1VAL['blood']:
        return T1VAL['blood'][field_strength]

    raise ValueError(f"No blood T1 values for field strength {field_strength}")


def T2(field_strength=3.0, tissue='gray matter', force=False) -> float:
    """T2 value of selected tissue types.

    Values are taken from `Gach et al 2008 <https://ieeexplore.ieee.org/document/4616690>`_

    Args:
        field_strength (float, optional): Field strength in Tesla. Defaults to 3.0.
        tissue (str, optional): Tissue type. Defaults to 'gray matter'.

    Raises:
        ValueError: If the requested T2 values are not available.

    Returns:
        float: T2 values in sec

    Note:

        This library is in construction and not all known values are currently available through this function.

        Available tissue types:
            - skin
            - bone marrow
            - csf
            - white matter
            - gray matter

        Available field strengths:
            - 1.5
            - 3

    Example:

        >>> import dcmri as dc
        >>> print('The T2 of skin at 1.5T is', 1e3*dc.T2(1.5, 'skin'), 'msec')
        The T2 of skin at 1.5T is 70.0 msec
    """
    # H. M. Gach, C. Tanase and F. Boada, "2D & 3D Shepp-Logan Phantom
    # Standards for MRI," 2008 19th International Conference on Systems
    # Engineering, Las Vegas, NV, USA, 2008, pp. 521-526, doi:
    # 10.1109/ICSEng.2008.15.

    VAL = values.T2_RELAXATION_TIMES
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in VAL else map.get(tissue)

    if tissue_key in VAL and field_strength in VAL[tissue_key]:
        return VAL[tissue_key][field_strength]

    if force:
        raise ValueError(f'No T2 values for {tissue} at {field_strength} T.')

    if field_strength in VAL['blood']:
        return VAL['blood'][field_strength]

    raise ValueError(f"No blood T2 values for field strength {field_strength}")


def T2s(field_strength=3.0, tissue='gray matter', force=False) -> float:
    """T2* value of selected tissue types.

    Note these qre quick estimates - need more in-depth research.
    """

    VAL = values.T2_STAR_RELAXATION_TIMES
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in VAL else map.get(tissue)

    if tissue_key in VAL and field_strength in VAL[tissue_key]:
        return VAL[tissue_key][field_strength]

    if force:
        raise ValueError(f'No T2* values for {tissue} at {field_strength} T.')

    if field_strength in VAL['blood']:
        return VAL['blood'][field_strength]

    raise ValueError(f"No blood T2* values for field strength {field_strength}")


def PD(tissue='gray matter', force=False) -> float:
    """Relative proton density (PD) value of selected tissue types.

    Values are taken from `Gach et al 2008 <https://ieeexplore.ieee.org/document/4616690>`_

    Args:
        tissue (str, optional): Tissue type. Defaults to 'gray matter'.

    Raises:
        ValueError: If the requested PD values are not available.

    Returns:
        float: PD values (dimensionless, range 0-1)

    Note:

        This library is in construction and not all known values are currently available through this function.

        Available tissue types:
            - skin
            - bone marrow
            - csf
            - white matter
            - gray matter

    Example:

        >>> import dcmri as dc
        >>> print('The PD of skin is', dc.PD('skin'))
        The PD of skin is 0.8
    """
    VAL = values.PROTON_DENSITY
    map = quantities.COMPS | quantities.ROIS

    tissue_key = tissue if tissue in VAL else map.get(tissue)

    if tissue_key in VAL:
        return VAL[tissue_key]

    if force:
        raise ValueError(f'No proton-density values for {tissue}.')

    return VAL['blood']


def perfusion(parameter='Fb', tissue='gray matter') -> float:
    """perfusion parameters of selected tissue types.

    Args:
        parameter (str, optional): perfusion parameter. Options are 'Fb' (blood flow), 'vb' (Blood volume), 'PS' (permeability-surface area product) and 'vi' (interstitial volume). Defaults to 'Fb'.
        tissue (str, optional): Tissue type. Defaults to 'gray matter'.

    Raises:
        ValueError: If the requested parameter values are not available.

    Returns:
        float: parameter value in standard units

    Note:

        This library is in construction and not all known values are currently available through this function.

        Available tissue types:
            - skin
            - bone marrow
            - csf
            - white matter
            - gray matter

    Example:

        >>> import dcmri as dc
        >>> print('The BF of gray matter is', dc.perfusion('Fb', 'gray matter'), 'mL/sec/mL')
        The BF of gray matter is 0.01 mL/sec/mL
    """
    try:
        return values.PERFUSION[parameter][tissue]
    except KeyError:
        raise ValueError(
            f"No {parameter} values for {tissue}"
        ) from None
