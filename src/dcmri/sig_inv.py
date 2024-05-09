import math
import numpy as np


def conc_spgress(S, S0, T10, FA, TR, r1):
    """
    Calculates the tracer concentration from a spoiled gradient-echo signal.

    Arguments
    ---------
        S: Signal S(C) at concentration C
        S0: Precontrast signal S(C=0)
        FA: Flip angle in degrees
        TR: Repetition time TR in msec (=time between two pulses)
        T10: Precontrast T10 in msec
        r1: Relaxivity in Hz/mM

    Returns
    -------
        Concentration in mM
    """
    E = math.exp(-TR/T10)
    c = math.cos(FA*math.pi/180)
    Sn = (S/S0)*(1-E)/(1-c*E)	#normalized signal
    R1 = -np.log((1-Sn)/(1-c*Sn))/TR	#relaxation rate in 1/msec
    return 1000*(R1 - 1/T10)/r1 


def conc_lin(S, S0, T10, r1):
    """
    Calculates the tracer concentration from a signal that is linear in the concentration.

    Arguments
    ---------
        S: Signal S(C) at concentration C
        S0: Precontrast signal S(C=0)
        T10: Precontrast T10 in msec
        r1: Relaxivity in Hz/mM

    Returns
    -------
        Concentration in mM
    """
    R10 = 1/T10
    R1 = R10*S/S0	#relaxation rate in 1/msec
    return 1000*(R1 - R10)/r1 