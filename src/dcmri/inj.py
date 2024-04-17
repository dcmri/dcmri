import numpy as np

def inj_step(t, weight, conc, dose1, rate, start1, dose2=None, start2=None):
    """dose injected per unit time (mmol/sec)"""

    duration = weight*dose1/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    dt = np.amin(t[1:]-t[:-1])
    # Check consistency of inputs
    if duration==0:
        msg = 'Invalid input variables. \n' 
        msg = 'The injection duration is zero.'
        raise ValueError(msg)
    if dt >= duration:
        msg = 'Invalid input variables. \n' 
        msg = 'The smallest time step dt ('+dt+' sec) is larger than the injection duration 1 (' + duration + 'sec). \n'
        msg = 'We would recommend dt to be at least 5 times smaller.'
        raise ValueError(msg)
    
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J1 = np.interp(t-start1, t, J, left=0)
    if start2 is None:
        return J1
    duration = weight*dose2/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    # Check consistency of inputs
    if duration==0:
        msg = 'Invalid input variables. \n' 
        msg = 'The injection duration is zero.'
        raise ValueError(msg)
    if dt >= duration:
        msg = 'Invalid input variables. \n' 
        msg = 'The smallest time step dt ('+dt+' sec) is larger than the injection duration 2 (' + duration + 'sec). \n'
        msg = 'We would recommend dt to be at least 5 times smaller.'
        raise ValueError(msg)
    t_inject = (t > 0) & (t < duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    J2 = np.interp(t-start2, t, J, left=0)
    return J1 + J2