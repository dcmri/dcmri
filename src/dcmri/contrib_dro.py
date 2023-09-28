"""
Digital reference objects
"""

import numpy as np
from contrib_aux import quad


def sample(t, S, ts, dts): 
    """Sample the signal"""
    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 


def step_inject(t, weight=70, conc=0.5, dose=0.2, rate=2, start=0):
    """
    Injected flux as a steo function

    weight (kg)
    conc (mmol/mL)
    dose mL/kg
    rate mL/sec

    returns dose injected per unit time (mmol/sec)"""

    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > start) & (t < start+duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    return J


def flow_organ_1d(
        length = 25,    # cm
        nr = 25,        # Nr of voxels
        vmax = 20,      # cm/sec
        vmin = 5,       # cm/sec
        flow = 4,       # mL/sec/cm^2
        ):
    dx = length/nr
    # voxel centers
    c = np.arange(dx/2, length+dx/2, dx)
    # voxel boundaries
    b = np.arange(0, length+dx, dx)
    # velocity = f/v
    u = quad(b, [vmax,vmin,vmax])
    # volume fractions
    v = flow/u
    v = (v[1:]+v[:-1])/2
    return {
        'voxel centers (cm)': c,
        'voxel boundaries (cm)': b,
        'velocity (cm/sec)': u,
        'volume fraction': v,
        }

def organ_perf_1d(
        length = 25,
        nr = 30,
        ):
    pass



######### TESTS ############

tmax = 120 # sec
dt = 0.01 # sec
MTT = 20 # sec

weight = 70.0           # Patient weight in kg
conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
dose = 0.025            # mL per kg bodyweight (quarter dose)
rate = 1                # Injection rate (mL/sec)
start = 20.0        # sec
dispersion = 90    # %


def test_step_inject():
    t = np.arange(0, tmax, dt)
    J = step_inject(t, weight, conc, dose, rate, start)
    assert np.round(np.sum(J)*dt,1) == np.round(weight*dose*conc, 1)

if __name__ == "__main__":

    test_step_inject()