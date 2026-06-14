import numpy as np
from scipy.integrate import trapezoid
from dcmri.kinetics.blocks import flux


def flux_aorta(J_vena: np.ndarray,
        t=None, dt=1.0, E=0.1, FFkl=0.0, FFk=0.5,
        heartlung=['pfcomp', {'T':10, 'D':0.2}],
        organs=['2cxm', {'T':[20, 120], 'E':0.15}],
        kidneys=['comp', {'T':10}],
        liver=['pfcomp', {'T':10, 'D':0.2}],
        tol=0.001,
        max_it=None,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rk = FFk * FFkl * (1 - E)
    Rl = (1 - FFk) * FFkl * (1 - E)
    Ro = (1 - FFkl) * (1 - E)

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    it=0
    while True:
      
        # Aorta flux of the current pass
        J_aorta = flux(heartlung[0], J_vena, t=t, dt=dt, **heartlung[1])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = Ro * flux(organs[0], J_aorta, t=t, dt=dt, **organs[1])

        if np.sum(Rl) > 0:
            J_vena += Rl * flux(liver[0], J_aorta, t=t, dt=dt, **liver[1])

        if np.sum(Rk) > 0:
            J_vena += Rk * flux(kidneys[0], J_aorta, t=t, dt=dt, **kidneys[1])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total


def flux_aorta_hlo(J_vena: np.ndarray,
        t=None, dt=1.0, E=0.1, 
        heartlung=['pfcomp', {'T':10, 'D':0.2}],
        organs=['2cxm', {'T':[20, 120], 'E':0.15}],
        veins=['pass', {}],
        tol=0.001,
        max_it=None,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Ro = 1-E

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    it=0
    while True:
      
        # Pass through heart and lungs
        J_aorta = flux(heartlung[0], J_vena, t=t, dt=dt, **heartlung[1])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Pass through organs
        J_vena = Ro * flux(organs[0], J_aorta, t=t, dt=dt, **organs[1])

        # Pass through the venous return
        J_vena = flux(veins[0], J_vena, t=t, dt=dt, **veins[1])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total


def flux_aorta_hlol(J_vena: np.ndarray,
        t=None, dt=1.0, El=0.1, Ek=0.1, FFl=0.0,
        heartlung=['pfcomp', {'T':10, 'D':0.2}],
        organs=['2cxm', {'T':[20, 120], 'E':0.15}],
        liver=['pfcomp', {'T':10, 'D':0.2}],
        tol=0.001,
        max_it=None,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rl = FFl * (1 - El)
    Ro = (1 - FFl) * (1 - Ek)

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    it=0
    while True:
      
        # Aorta flux of the current pass
        J_aorta = flux(heartlung[0], J_vena, t=t, dt=dt, **heartlung[1])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = Ro * flux(organs[0], J_aorta, t=t, dt=dt, **organs[1])
        J_vena += Rl * flux(liver[0], J_aorta, t=t, dt=dt, **liver[1])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total


def flux_aorta_hlok(J_vena: np.ndarray,
        t=None, dt=1.0, El=0.1, Ek=0.1, FFk=0.0,
        heartlung=['pfcomp', {'T':10, 'D':0.2}],
        organs=['2cxm', {'T':[20, 120], 'E':0.15}],
        kidney=['comp', {'T':10}],
        tol=0.001,
        max_it=None,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rk = FFk * (1 - Ek)
    Ro = (1 - FFk) * (1 - El)

    # Initialize output
    J_aorta_total = np.zeros(J_vena.size)

    it=0
    while True:
      
        # Aorta flux of the current pass
        J_aorta = flux(heartlung[0], J_vena, t=t, dt=dt, **heartlung[1])

        # Add to the total aorta flux
        J_aorta_total += J_aorta

        # Venous flux of the current pass
        J_vena = Ro * flux(organs[0], J_aorta, t=t, dt=dt, **organs[1])
        J_vena += Rk * flux(kidney[0], J_aorta, t=t, dt=dt, **kidney[1])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total

def flux_aorta_hlokk(J_vena: np.ndarray,
        t=None, dt=1.0, 
        El=0.1, Elk=0.1, Erk=0.1, FFlk=0.1, FFrk=0.1,
        heartlung=['pfcomp', {'T':10, 'D':0.2}],
        organs=['2cxm', {'T':[20, 120], 'E':0.15}],
        left_kidney=['comp', {'T':10}],
        right_kidney=['comp', {'T':10}],
        tol=0.001,
        max_it=None,
    ):
    dose = trapezoid(J_vena, x=t, dx=dt)
    min_dose = tol*dose

    # Residuals of each pathway
    Rlk = FFlk * (1 - Elk)
    Rrk = FFrk * (1 - Erk)
    FFo = 1 - (FFlk + FFrk)
    Ro = (1 - FFo) * (1 - El)

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

        if Ro > 0:
            J_vena += Ro * flux(organs[0], J_aorta, t=t, dt=dt, **organs[1])

        if Rrk > 0:
            J_vena += Rrk * flux(right_kidney[0], J_aorta, t=t, dt=dt, **right_kidney[1])

        if Rlk > 0:
            J_vena += Rlk * flux(left_kidney[0], J_aorta, t=t, dt=dt, **left_kidney[1])

        # Get residual dose in current pass
        dose = trapezoid(J_vena, x=t, dx=dt)

        if dose <= min_dose:
            break
        
        it += 1
        if max_it is not None:
            if it > max_it:
                break

    return J_aorta_total



