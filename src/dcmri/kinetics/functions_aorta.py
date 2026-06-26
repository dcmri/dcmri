import numpy as np
from scipy.integrate import trapezoid
from dcmri.kinetics.functions_blocks import flux


def flux_aorta(
        J_vena: np.ndarray,
        t=None, 
        dt=1.0, 
        heartlung=None,
        organs=None,
        tol=0.1,
        max_it=None,
        
    ):
    if heartlung is None:
        heartlung = {
            'model': 'pfcomp', 
            'params': {'T':10, 'D':0.2}
        }
    if organs is None:
        organs = [
            {
            'vr': 1.0,
            'model': 'pfcomp', 
            'params': {'T':30, 'D':0.5}
            }  
        ]      
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol * dose

    # # Residuals of each pathway
    # FFo = 1 - (FFlk + FFrk)
    # vr_lk = FFlk * (1 - Elk)
    # vr_rk = FFrk * (1 - Erk)
    # vr_o = FFo * (1 - El)

    # Initialize output
    nt = J_vena.size
    J_aorta_total = np.zeros(nt)

    it=0
    while True:
      
        # Aorta flux of the current pass
        J_aorta = flux(heartlung['model'], J_vena, t=t, dt=dt, **heartlung['params'])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = np.zeros(nt)

        for organ in organs:

            if isinstance(organ, tuple):
                # Propagate through a series of organs
                # Only the last one has a venous return associated (vr)
                vr_series = organ[-1]['vr']
                if vr_series > 0:
                    J_series = J_aorta
                    for single_organ in organ:
                        J_series = flux(single_organ['model'], J_series, t=t, dt=dt, **single_organ['params'])
                    J_vena += vr_series * J_series

            elif organ['vr'] > 0:
                # Propagate through a single organ with a given vr
                J_vena += organ['vr'] * flux(organ['model'], J_aorta, t=t, dt=dt, **organ['params'])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total