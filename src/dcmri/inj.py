import numpy as np

def inj_step(t, weight, conc, dose1, rate, start1, dose2=None, start2=None):
    """dose injected per unit time (mM/sec)"""

    duration = weight*dose1/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J1 = np.interp(t-start1, t, J, left=0)
    if start2 is None:
        return J1
    duration = weight*dose2/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J2 = np.interp(t-start2, t, J, left=0)
    return J1 + J2