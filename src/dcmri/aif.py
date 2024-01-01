import numpy as np


def aif_parker(t:np.ndarray, BAT:float=30.0)->np.ndarray:
    """AIF model as defined by Parker et al (2005)

    Args:
        t (np.ndarray): array of time points in units of sec. 
        BAT (float, optional): Time in seconds before the bolus arrives. Defaults to 30sec. 

    Returns:
        np.ndarray: Concentrations in M for each time point in t.

    Example:

        Create an array of time points covering 6min in steps of 1sec, calculate the Parker AIF at these time points and plot the results.

        Import packages:

        >>> import matplotlib.pyplot as plt
        >>> import dcmri

        Calculate AIF and plot

        >>> t = np.arange(0, 6*60, 0.1)
        >>> ca = dcmri.aif_parker(t)
        >>> plt.plot(t,ca)
        >>> plt.show()
    """

    # Convert from OSIPI units (sec) to units used internally (mins)
    t_min = t/60
    bat_min = BAT/60

    t_offset = t_min - bat_min

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


