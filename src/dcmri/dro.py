"""
Digital reference objects
"""

import numpy as np
from helper import quad


def step_inject(t: np.ndarray,
                weight=70,
                conc=0.5,
                dose=0.2,
                rate=2,
                start=0
                ) -> np.ndarray:
    """
    Calculates injected flux as a step function.

    Args:
        t: numpy array of time series, t.
        weight: integer value of subject body weight [kg].
        conc: integer value of injected tracer concentration [mmol/mL].
        dose: integer value of injected tracer dose [mL/kg].
        rate: integer value of rate of tracer injection [mL/sec].
        start: integer value of injection start time [s].

    Returns:
        Dose injected per unit time, i.e., flux, J [mmol/sec].
    """
    duration = weight*dose/rate     # sec = kg * (mL/kg) / (mL/sec)
    Jmax = conc*rate                # mmol/sec = (mmol/ml) * (ml/sec)
    t_inject = (t > start) & (t < start+duration)
    J = np.zeros(t.size)
    J[np.nonzero(t_inject)[0]] = Jmax
    return J


def flow_organ_1d(length = 25,
                  nr = 25,
                  vmax = 20,
                  vmin = 5,
                  flow = 4,
                  ) -> dict[int,
                            int,
                            int,
                            int]:
    """
    Propagates organ flow in 1d.

    Args:
        length: integer value of length [cm].
        nr: integer value of number of voxels.
        vmax: integer value of vmax [cm/sec].
        vmin: integer value of vmin [cm/sec].
        flow: integer value of flow [mL/sec/cm^2].

    Returns:
        Dictionary of values for voxel centers, c [cm],
        voxel boundaries, b [cm], velocity, u [cm/sec],
        and volume fraction, v.
    """
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