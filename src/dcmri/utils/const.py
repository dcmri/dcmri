
import copy


AGENTS = [
    'gadoxetate',
    'gadobutrol',
    'gadopentetate',
    'gadobenate',
    'gadodiamide',
    'gadoterate',
    'gadoteridol',
    'gadopiclenol',
]

# -----------------------------------------------------------------------------
# Global Relaxivity Databases (Hz/mM)
# -----------------------------------------------------------------------------

_R1_RELAXIVITY = {
    'plasma': {
        'gadopiclenol': {1.5: 12.8, 3.0: 11.6},
        'gadopentetate': {0.47: 3.8, 1.5: 4.1, 3.0: 3.7, 4.7: 3.8}, # Magnevist
        'gadobutrol': {0.47: 6.1, 1.5: 5.2, 3.0: 5.0, 4.7: 4.7}, # Gadovist
        'gadoteridol': {0.47: 4.8, 1.5: 4.1, 3.0: 3.7, 4.7: 3.7}, # Prohance
        'gadobenade': {0.47: 9.2, 1.5: 6.3, 3.0: 5.5, 4.7: 5.2}, # Multihance
        'gadoterate': {0.47: 4.3, 1.5: 3.6, 3.0: 3.5, 4.7: 3.3}, # Dotarem
        'gadodiamide': {0.47: 4.4, 1.0: 4.35, 1.5: 4.3, 3.0: 4.0, 4.7: 3.9}, # Omniscan
        'mangafodipir': {0.47: 3.6, 1.5: 3.6, 3.0: 2.7, 4.7: 2.2}, # Teslascan
        'gadoversetamide': {0.47: 5.7, 1.5: 4.7, 3.0: 4.5, 4.7: 4.4}, # Optimark
        'ferucarbotran': {0.47: 15.0, 1.5: 7.4, 3.0: 3.3, 4.7: 1.7},  # Resovist
        'ferumoxide': {1.5: 4.5, 3.0: 2.7, 4.7: 1.2}, # Feridex
        'gadoxetate': {0.47: 8.7, 1.5: 8.1, 3.0: 6.4, 4.7: 6.4, 7.0: 6.2, 9.0: 6.1}, # Primovist
    }
}

# Deep-copy plasma parameters over for hepatocytes as base configuration
_R1_RELAXIVITY['hepatocytes'] = copy.deepcopy(_R1_RELAXIVITY['plasma'])
_R1_RELAXIVITY['hepatocytes']['gadoxetate'] = {
    1.5: 14.6, 3.0: 9.8, 4.7: 7.6, 7.0: 6.0, 9.0: 6.1
}

# Known literature value mapping for pure r2 (spin-echo / CPMG sequence data).
# Missing entries are entirely omitted to leave out unknown combinations.
_R2_RELAXIVITY = {
    'plasma': {
        'gadopiclenol': {1.5: 13.2, 3.0: 15.4},
        'gadopentetate': {0.47: 4.6, 1.5: 4.6, 3.0: 4.8, 4.7: 5.0},
        'gadobutrol': {0.47: 7.3, 1.5: 6.1, 3.0: 7.4, 4.7: 6.1},
        'gadoteridol': {0.47: 5.6, 1.5: 5.0, 3.0: 4.9, 4.7: 5.1},
        'gadobenade': {0.47: 10.9, 1.5: 8.4, 3.0: 8.1, 4.7: 8.3},
        'gadoterate': {0.47: 5.1, 1.5: 4.4, 3.0: 4.7, 4.7: 4.5},
        'gadodiamide': {0.47: 5.2, 1.5: 5.1, 3.0: 5.0, 4.7: 5.2},
        'mangafodipir': {0.47: 4.2, 1.5: 4.4, 3.0: 3.6, 4.7: 3.1},
        'gadoversetamide': {0.47: 6.8, 1.5: 5.9, 3.0: 6.0, 4.7: 6.2},
        'gadoxetate': {0.47: 10.8, 1.5: 10.1, 3.0: 8.8, 4.7: 9.1},
    }
}
_R2_RELAXIVITY['hepatocytes'] = copy.deepcopy(_R2_RELAXIVITY['plasma'])


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

        print('gadobutrol is available in a solution of', dc.ca_conc('gadobutrol'), 'M')
        print('gadoterate is available in a solution of', dc.ca_conc('gadoterate'), 'M')
    """
    if agent == 'gadoxetate':
        return 0.25     # mmol/mL
    if agent == 'gadobutrol':
        return 1.0      # mmol/mL
    if agent in [
        'gadopentetate',
        'gadobenate',
        'gadodiamide',
        'gadoterate',
        'gadoteridol',
        'gadopiclenol',
    ]:
        return 0.5  # mmol/mL
    raise ValueError(
        f"No concentration data for contrast agent {agent}."
        f"Possible values are {AGENTS}."
    )


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
    if agent == 'gadoxetate':
        # https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf
        return 0.1  # mL/kg
    if agent == 'gadobutrol':
        return 0.1      # mL/kg
    if agent == 'gadopiclenol':
        return 0.1      # mL/kg
    if agent in [
            'gadopentetate',
            'gadobenate',
            'gadodiamide',
            'gadoterate',
            'gadoteridol',
    ]:
        return 0.2      # mL/kg  # 0.5 mmol/mL = 0.1 mmol/kg
    raise ValueError(
        f"No data available for contrast agent {agent}."
        f"Possible values are {AGENTS}."
    )



def relaxivity(field_strength=3.0, tissue='plasma', agent='gadoxetate') -> dict:
    """Contrast agent relaxivity values in units of Hz/M"""
    return {
        'r1': r1(field_strength, tissue, agent),
        'r2': r2(field_strength, tissue, agent),
        'r2s': r2s(field_strength, tissue, agent),
    }

def r1(field_strength=3.0, tissue='plasma', agent='gadoxetate') -> float:
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
    if tissue == 'blood':
        tissue = 'plasma'
        
    try:
        # Values in dictionary are stored in Hz/mM; convert to Hz/M (multiply by 1000)
        return 1000.0 * _R1_RELAXIVITY[tissue][agent][field_strength]
    except KeyError:
        raise ValueError(
            f"No r1 relaxivity data available for {agent} in {tissue} at {field_strength} T."
        )
    

def r2(field_strength=3.0, tissue='plasma', agent='gadoxetate') -> float:
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
    if tissue == 'blood':
        tissue = 'plasma'
        
    try:
        # Values in dictionary are stored in Hz/mM; convert to Hz/M (multiply by 1000)
        return 1000.0 * _R2_RELAXIVITY[tissue][agent][field_strength]
    except KeyError:
        raise ValueError(
            f"No r1 relaxivity data available for {agent} in {tissue} at {field_strength} T."
        )


def r2s(field_strength=3.0, tissue='blood', agent='gadoxetate') -> float:
    """R2*-relaxivity"""
    if tissue=='blood':
        # Estimated from the range [0, 5mM] in data by van Osch MJ, Vonken EJ, Viergever MA, van der Grond J, Bakker CJ. Measuring the arterial input function with gradient echo sequences. Magn Reson Med 2003;49:1067–1076
        return 10e3
    else:
        # TODO: Look up literature values
        return 20e3


def T1(field_strength=3.0, tissue='blood', Hct=0.45) -> float:
    """T1 value of selected tissue types.

    Values are taken from literature, mostly from `Stanisz et al 2005 <https://doi.org/10.1002/mrm.20605>`_

    Args:
        field_strength (float, optional): Field strength in Tesla (see below for options). Defaults to 3.0.
        tissue (str, optional): Tissue type (see below for options). Defaults to 'blood'.
        Hct (float, optional): Hematocrit value - ignored when tissue is not blood. Defaults to 0.45.

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
    # 10.1109/ICSEng.2008.15.

    T1val = {
        'skin': {  # Gach 2008 (scalp)
            1.5: 0.324 * (1.5**0.137),
            3.0: 0.324 * (3.0**0.137),
        },
        'bone marrow': {  # Gach 2008
            1.5: 0.533 * (1.5**0.088),
            3.0: 0.533 * (3.0**0.088),
        },
        'csf': {  # Gach 2008
            1.5: 4.20,
            3.0: 4.20,
        },
        'muscle': {
            1.5: 1.008,
            3.0: 1.412,
        },
        'heart': {
            1.5: 1.030,
            3.0: 1.471,
        },
        'cartilage': {
            1.5: 1.024,
            3.0: 1.168,
        },
        'white matter': {
            1.5: 0.884,
            3.0: 1.084,
        },
        'gray matter': {
            1.5: 1.124,
            3.0: 1.820,
        },
        'optic nerve': {
            1.5: 0.815,
            3.0: 1.083,
        },
        'spinal cord': {
            1.5: 0.745,
            3.0: 0.993,
        },
        'blood': {
            1.0: 1.378,  # Extrapolated
            1.5: 1.441,
            3.0: 1 / (0.52 * Hct + 0.38),  # Lu MRM 2004
            4.7: 1 / 1.70,  # https://cds.ismrm.org/ismrm-2002/PDF4/1048.PDF
            7.0: 1 / 2.29,  # 10.1016/j.mri.2012.08.008
        },
        'spleen': {
            4.7: 1 / 0.631,
            7.0: 1 / 0.611,
            9.0: 1 / 0.600,
        },
        'liver': {
            1.5: 0.602,  # liver R1 in 1/sec (Waterton 2021)
            3.0: 0.752,  # liver R1 in 1/sec (Waterton 2021)
            # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
            4.7: 1 / 1.281,
            # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
            7.0: 1 / 1.109,
            # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)
            9.0: 1 / 0.920,
        },
        'kidney': {
            # Reference values average over cortex and medulla from Cox et al
            # https://academic.oup.com/ndt/article/33/suppl_2/ii41/5078406
            1.0: 1.017,  # Extrapolated
            1.5: (1.024 + 1.272) / 2,
            3.0: (1.399 + 1.685) / 2,
        },
    }
    try:
        return T1val[tissue][field_strength]
    except BaseException:
        msg = 'No T1 values for ' + tissue + \
            ' at ' + str(field_strength) + ' T.'
        raise ValueError(msg)


def T2(field_strength=3.0, tissue='gray matter') -> float:
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

    T2val = {
        'skin': {  # Gach 2008 (scalp)
            1.5: 0.07,
            3.0: 0.07,
        },
        'bone marrow': {  # Gach 2008
            1.5: 0.05,
            3.0: 0.05,
        },
        'csf': {  # Gach 2008
            1.5: 1.99,
            3.0: 1.99,
        },
        'white matter': {
            1.5: 0.08,
            3.0: 0.08,
        },
        'gray matter': {
            1.5: 0.1,
            3.0: 0.1,
        },
    }
    try:
        return T2val[tissue][field_strength]
    except BaseException:
        msg = 'No T2 values for ' + tissue + \
            ' at ' + str(field_strength) + ' T.'
        raise ValueError(msg)


def PD(tissue='gray matter') -> float:
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
    # H. M. Gach, C. Tanase and F. Boada, "2D & 3D Shepp-Logan Phantom
    # Standards for MRI," 2008 19th International Conference on Systems
    # Engineering, Las Vegas, NV, USA, 2008, pp. 521-526, doi:
    # 10.1109/ICSEng.2008.15.

    PDval = {
        'skin': 0.8,
        'bone marrow': 0.12,
        'csf': 0.98,
        'white matter': 0.617,
        'gray matter': 0.745,
    }
    try:
        return PDval[tissue]
    except BaseException:
        msg = 'No PD values for ' + tissue
        raise ValueError(msg)


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
    if parameter == 'Fb':
        val = {
            'skin': 0.005,  # 5 kg/s/m**3 = 5000mL/sec/100*100*100 mL = 0.005mL/sec/mL
            'bone marrow': 0.0013,  # 0.08 ml/ml/min
            'csf': 0.0,
            'white matter': 0.0033,
            'gray matter': 0.01,
        }
    elif parameter == 'vb':
        val = {
            'skin': 0.03,
            'bone marrow': 0.25,
            'csf': 0.0,
            'white matter': 0.02,
            'gray matter': 0.05,
        }
    elif parameter == 'PS':  # Needs verification
        val = {
            'skin': 0.001,
            'bone marrow': 0.0002,
            'csf': 0.0,
            'white matter': 0.0,
            'gray matter': 0.0,
        }
    elif parameter == 'vi':  # Needs verification
        val = {
            'skin': 0.03,
            'bone marrow': 0.2,
            'csf': 0.0,
            'white matter': 0.3,
            'gray matter': 0.35,
        }
    try:
        return val[tissue]
    except BaseException:
        msg = 'No ' + parameter + ' values for ' + tissue
        raise ValueError(msg)
