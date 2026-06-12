import numpy as np
from scipy.integrate import trapezoid
from dcmri.kinetics.blocks import flux


def conc_aorta(J_vena: np.ndarray,
        t=None, dt=1.0, E=0.1, Fo=60, Fk=20, Fl=20,
        heartlung=['pfcomp', {'T':10, 'D':0.2}],
        organs=['2cxm', {'T':[20, 120], 'E':0.15}],
        kidneys=['comp', {'T':10}],
        liver=['pfcomp', {'T':10, 'D':0.2}],
        tol=0.001,
        max_it=500,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    CO = Fo + Fk + Fl
    FFk = Fk / CO
    FFl = Fl / CO
    FFo = Fo / CO

    # Initialize output
    nt = J_vena.size
    J_aorta_total = np.zeros(nt)

    it=0
    while True:
      
        # Aorta flux of the current pass
        J_aorta = flux(heartlung[0], J_vena, t=t, dt=dt, **heartlung[1])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = np.zeros(nt)

        if FFo > 0:
            J_vena += flux(organs[0], FFo * J_aorta, t=t, dt=dt, **organs[1])

        if FFl > 0:
            J_vena += flux(liver[0], FFl * J_aorta, t=t, dt=dt, **liver[1])

        if FFk > 0:
            J_vena += flux(kidneys[0], FFk * J_aorta, t=t, dt=dt, **kidneys[1])

        # Account for indicator loss
        J_vena = (1 - E) * J_vena

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total / CO