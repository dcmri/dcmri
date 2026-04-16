import numpy as np
from scipy.integrate import trapezoid

import dcmri.kinetics.lib as pk



def flux_aorta(J_vena: np.ndarray,
        t=None, dt=1.0, E=0.1, FFkl=0.0, FFk=0.5,
        heartlung=['pfcomp', (10, 0.2)],
        organs=['2cxm', ([20, 120], 0.15)],
        kidneys=['comp', (10,)],
        liver=['pfcomp', (10, 0.2)],
        tol=0.001,
        max_it=None,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rk = FFk*FFkl*(1-E)
    Rl = (1-FFk)*FFkl*(1-E)
    Ro = (1-FFkl)*(1-E)

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    it=0
    while True:
      
        # Aorta flux of the current pass
        J_aorta = pk.flux(
            J_vena, *heartlung[1], t=t, dt=dt, model=heartlung[0])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = Ro * pk.flux(
            J_aorta, *organs[1], t=t, dt=dt, model=organs[0])
        if np.sum(Rl) > 0:
            J_vena += Rl * pk.flux(
                J_aorta, *liver[1], t=t, dt=dt, model=liver[0])
        if np.sum(Rk) > 0:
            J_vena += Rk * pk.flux(
                J_aorta, *kidneys[1], t=t, dt=dt, model=kidneys[0])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total
