import numpy as np


def aif_parker(t, BAT:float=0.0)->np.ndarray:
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
    t_offset = np.array(t)/60 - BAT/60

    #A1/(SD1*sqrt(2*PI)) * exp(-(t_offset-m1)^2/(2*var1))
    #A1 = 0.833, SD1 = 0.055, m1 = 0.171
    gaussian1 = 5.73258 * np.exp(
        -1.0 *
        (t_offset - 0.17046) * (t_offset - 0.17046) /
        (2.0 * 0.0563 * 0.0563) )
    
    #A2/(SD2*sqrt(2*PI)) * exp(-(t_offset-m2)^2/(2*var2))
    #A2 = 0.336, SD2 = 0.134, m2 = 0.364
    gaussian2 = 0.997356 * np.exp(
        -1.0 *
        (t_offset - 0.365) * (t_offset - 0.365) /
        (2.0 * 0.132 * 0.132))
    # alpha*exp(-beta*t_offset) / (1+exp(-s(t_offset-tau)))
    # alpha = 1.064, beta = 0.166, s = 37.772, tau = 0.482
    sigmoid = 1.050 * np.exp(-0.1685 * t_offset) / (1.0 + np.exp(-38.078 * (t_offset - 0.483)))

    pop_aif = gaussian1 + gaussian2 + sigmoid
    
    return pop_aif/1000 # convert to M


