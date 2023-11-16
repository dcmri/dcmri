"""Spatially distributed tissue models"""

import numpy as np
import contrib_rt as rt

def conc_nephc(
            # Nephron model with constant properties 
            t, # time points (sec)
            J, # influx per unit volume (mmol/sec/mL tissue)
            r, # total reabsorption fraction 0<r<1
            omega0, # dimensionless velocity at the inlet (1/sec)
            delta0, # dimensionless diffusion coefficient (1/sec)
            n = 20, # nr of numerical voxels
            ):
    alpha = np.linspace(0, 1, n+1) # dimensionless locations of numerical voxel boundaries.
    lamda = -np.log(1-r)
    omega = omega0*np.exp(-lamda*alpha) # dimensionless velocity everywhere (1/sec)
    delta = delta0*np.ones(n+1) # dimensionless diffusivity everywhere (1/sec)
    # Compartmental rate constants
    d_alpha = alpha[1] # dimensionless size of numerical voxels
    Jt = J/d_alpha
    Kp, Kn = rt.K_flowdiff_1d(d_alpha, omega, delta) # 1/sec
    # High resolution time points
    dth = 0.9/(2*delta0/d_alpha**2 + omega0/d_alpha)
    tacq = np.amax(t)
    nth = 1 + np.ceil(tacq/dth).astype(np.int32)
    th = np.linspace(0, tacq, nth)
    # Upsample influx
    Jh = np.interp(th, t, Jt)
    # Calculate concentrations
    Ch = rt.conc_1d1c(th, Jh, Jh*0, Kp, Kn)
    # Sum concentrations over all voxels
    Cth = d_alpha*np.sum(Ch, axis=1)
    # Downsample concentrations to measured time resolution
    Ct = np.interp(t, th, Cth)
    return Ct # mmol/mL tissue




