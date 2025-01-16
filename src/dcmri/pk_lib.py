import numpy as np

import dcmri.pk as pk



def aif_parker(t, BAT: float = 0.0) -> np.ndarray:
    """Population AIF model as defined by `Parker et al (2006) <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21066>`_

    Args:
        t (array_like): time points in units of sec.
        BAT (float, optional): Time in seconds before the bolus arrives. Defaults to 0 sec (no delay).

    Returns:
        np.ndarray: Concentrations in M for each time point in t. If t is a scalar, the return value is a scalar too.

    References:
        Adapted from a contribution by the QBI lab of the University of Manchester to the `OSIPI code repository <https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection>`_.

    Example:

        >>> import numpy as np
        >>> import dcmri as dc

        Create an array of time points covering 20sec in steps of 1.5sec, which rougly corresponds to the first pass of the Paeker AIF:

        >>> t = np.arange(0, 20, 1.5)

        Calculate the Parker AIF at these time points, and output the result in units of mM:

        >>> 1000*dc.aif_parker(t)
        array([0.08038467, 0.23977987, 0.63896354, 1.45093969,
        2.75255937, 4.32881325, 5.6309778 , 6.06793854, 5.45203828,
        4.1540079 , 2.79568217, 1.81335784, 1.29063036, 1.08751679])
    """

    # Check input types
    if not np.isscalar(BAT):
        raise ValueError('BAT must be a scalar')

    # Convert from secs to units used internally (mins)
    t_offset = np.array(t) / 60 - BAT / 60

    # A1/(SD1*sqrt(2*PI)) * exp(-(t_offset-m1)^2/(2*var1))
    # A1 = 0.833, SD1 = 0.055, m1 = 0.171
    gaussian1 = 5.73258 * np.exp(
        -1.0 * (t_offset - 0.17046) * (t_offset - 0.17046) / (2.0 * 0.0563 * 0.0563))

    # A2/(SD2*sqrt(2*PI)) * exp(-(t_offset-m2)^2/(2*var2))
    # A2 = 0.336, SD2 = 0.134, m2 = 0.364
    gaussian2 = 0.997356 * np.exp(
        -1.0 * (t_offset - 0.365) * (t_offset - 0.365) / (2.0 * 0.132 * 0.132))
    # alpha*exp(-beta*t_offset) / (1+exp(-s(t_offset-tau)))
    # alpha = 1.064, beta = 0.166, s = 37.772, tau = 0.482
    sigmoid = 1.050 * np.exp(-0.1685 * t_offset) / \
        (1.0 + np.exp(-38.078 * (t_offset - 0.483)))

    pop_aif = gaussian1 + gaussian2 + sigmoid

    return pop_aif / 1000  # convert to M


def aif_tristan_rat(t, BAT=4.6 * 60, duration=30) -> np.ndarray:
    """Population AIF model for rats measured with a standard dose of 
    gadoxetate.

    Args:
        t (array_like): time points in units of sec.
        BAT (float, optional): Time in seconds before the bolus arrives. 
          Defaults to 4.6 min.
        duration (float, optional): Duration of the injection. Defaults to 30s.

    Returns:
        np.ndarray: Blood concentrations in M for each time point in t. If 
        t is a scalar, the return value is a scalar too.

    References:

        - Melillo N, Scotcher D, Kenna JG, Green C, Hines CDG, Laitinen I, 
          et al. Use of In Vivo Imaging and Physiologically-Based Kinetic 
          Modelling to Predict Hepatic Transporter Mediated Drug-Drug 
          Interactions in Rats. 
          `Pharmaceutics 2023;15(3):896 <https://doi.org/10.3390/pharmaceutics15030896>`_.

        - Gunwhy, E. R., & Sourbron, S. (2023). TRISTAN-RAT (v3.0.0). 
          `Zenodo <https://doi.org/10.5281/zenodo.8372595>`_

    Example:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Create an array of time points over 30 minutes

        >>> t = np.arange(0, 30*60, 0.1)

        Generate the rat input function for these time points:

        >>> cb = dc.aif_tristan_rat(t)

        Plot the result:

        >>> plt.plot(t/60, 1000*cb, 'r-')
        >>> plt.title('TRISTAN rat AIF')
        >>> plt.xlabel('Time (min)')
        >>> plt.ylabel('Blood concentration (mM)')
        >>> plt.show()
    """
    # Constants

    dose = 0.0075   # mmol

    Fb = 2.27 / 60    # mL/sec
    # https://doi.org/10.1021/acs.molpharmaceut.1c00206
    # (Changed from 3.61/60 on 07/03/2022)
    # From Brown the cardiac output of rats is
    # 110.4 mL/min (table 3-1) ~ 6.62L/h
    # From table 3-4, sum of hepatic artery and portal vein
    # blood flow is 17.4% of total cardiac output ~ 1.152 L/h
    # Mass of liver is 9.15g, with density of 1.08 kg/L,
    # therefore ~8.47mL
    #  9.18g refers to the whole liver, i.e. intracellular tissue
    # + extracellular space + blood
    # Dividing 1.152L/h for 8.47mL we obtain ~2.27 mL/min/mL liver
    # Calculation done with values in Table S2 of our article
    # lead to the same results
    Hct = 0.418     # Cremer et al, J Cereb Blood Flow Metab 3, 254-256 (1983)
    VL = 8.47       # mL
    # Scotcher et al 2021, DOI: 10.1021/acs.molpharmaceut.1c00206
    # Supplementary material, Table S2
    GFR = 1.38 / 60   # mL/sec (=1.38mL/min)
    # https://doi.org/10.1152/ajprenal.1985.248.5.F734
    P = 0.172       # mL/sec
    # Estimated from rat repro study data using PBPK model
    # 0.62 L/h
    # Table 3 in Scotcher et al 2021
    # DOI: 10.1021/acs.molpharmaceut.1c00206
    VB = 15.8       # mL
    # 0.06 X BW + 0.77, Assuming body weight (BW) = 250 g
    # Lee and Blaufox. Blood volume in the rat.
    # J Nucl Med. 1985 Jan;26(1):72-6.
    VE = 30         # mL
    # All tissues, including liver.
    # Derived from Supplementary material, Table S2
    # Scotcher et al 2021
    # DOI: 10.1021/acs.molpharmaceut.1c00206
    E = 0.4         # Liver extraction fraction, estimated from TRISTAN data
    # Using a model with free E, then median over population of controls

    # Derived constants
    VP = (1 - Hct) * VB
    Fp = (1 - Hct) * Fb
    K = GFR + E * Fp * VL  # mL/sec

    # Influx in mmol/sec
    J = np.zeros(np.size(t))
    Jmax = dose / duration  # mmol/sec
    J[(t > BAT) & (t < BAT + duration)] = Jmax

    # Model parameters
    TP = VP / (K + P)
    TE = VE / P
    E = P / (K + P)

    Jp = pk.flux_2cxm(J, [TP, TE], E, t)
    cp = Jp / K

    return cp * (1 - Hct)
