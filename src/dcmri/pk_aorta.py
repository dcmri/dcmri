import numpy as np

import dcmri

# 4 parameters
def aorta_flux_4(J_lungs,
        T_lh, D_lh,
        Tp_organs,
        E_extr,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    dose0 = np.trapz(J_lungs, x=t, dx=dt)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_lungs, T_lh, D_lh, t=t, dt=dt, solver=solver)
        # Propagate through the other organs
        J_lungs = (1-E_extr)*dcmri.flux_comp(J_aorta, Tp_organs, t=t, dt=dt)
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, x=t, dx=dt)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_chain(J_lungs_total, T_lh, D_lh, t=t, dt=dt, solver=solver)
    # Return total flux into aorta
    return J_aorta_total

# 5 parameters

def aorta_flux_5(J_lungs, 
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        E_extr,
        t=None, dt=1.0, tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, x=t, dx=dt)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_comp(J_lungs, T_lh, t=t, dt=dt)
        # Propagate through the other organs
        J_aorta = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_lungs = (1-E_extr)*dcmri.flux_2comp(J_aorta, [Tp_organs, Te_organs], E_o, t=t, dt=dt)[0,0,:]
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, x=t, dx=dt)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_comp(J_lungs_total, T_lh, t=t, dt=dt)
    # Return total flux into aorta
    return J_aorta_total


# 6 parameters
def aorta_flux_6(J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        E_extr,
        t=None, dt=1.0, tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, x=t, dx=dt)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_pfcomp(J_lungs, T_lh, D_lh, t=t, dt=dt, solver='interp')
        # Propagate through the other organs
        J_aorta = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_lungs = (1-E_extr)*dcmri.flux_2comp(J_aorta, [Tp_organs, Te_organs], E_o, t=t, dt=dt)[0,0,:]
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, x=t, dx=dt)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_pfcomp(J_lungs_total, T_lh, D_lh, t=t, dt=dt, solver='interp')
    # Return total flux into aorta
    return J_aorta_total


# 6 parameters
def aorta_flux_6b(J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        E_extr,
        t=None, dt=1.0, tol = 0.001, solver='step'):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, x=t, dx=dt)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_lungs, T_lh, D_lh, t=t, dt=dt, solver=solver)
        # Propagate through the other organs
        J_aorta = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_lungs = (1-E_extr)*dcmri.flux_2comp(J_aorta, [Tp_organs, Te_organs], E_o, t=t, dt=dt)[0,0,:]
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, x=t, dx=dt)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_chain(J_lungs_total, T_lh, D_lh, t=t, dt=dt, solver=solver)
    # Return total flux into aorta
    return J_aorta_total


# 9 parameters
def aorta_flux_9(J_lungs,
        T_lh, D_lh,
        R_organs, E_organs, Tp_organs, Te_organs,
        R_liver, Te_liver, De_liver,
        t=None, dt=1.0, tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, x=t, dx=dt)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_pfcomp(J_lungs, T_lh, D_lh, t=t, dt=dt)
        # Propagate through liver and other organs
        # R_liver = (1-E_liver)*FF_liver
        # R_organs = (1-E_kidneys)*(1-FF_liver)
        J_liver = R_liver*dcmri.flux_pfcomp(J_aorta, Te_liver, De_liver, t=t, dt=dt)
        J_aorta = np.stack((J_aorta, np.zeros(J_aorta.size)))
        J_organs = R_organs*dcmri.flux_2comp(J_aorta, [Tp_organs, Te_organs], E_o, t=t, dt=dt)[0,0,:]
        # Add up outfluxes from liver and other organs
        J_lungs = J_liver + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, x=t, dx=dt)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_pfcomp(J_lungs_total, T_lh, D_lh, t=t, dt=dt)
    # Return total flux into aorta
    return J_aorta_total

# 8 parameters
# flux_wb_kidneys 
def aorta_flux_8(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, 
        tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_comp(J_lungs, T_lh, t)
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, 1/Ke_kidneys, t)
        J_organs = (1-E_liver)*dcmri.flux_2comp(J_organs, [Tp_organs, Te_organs], E_o, t)[0,0,:]
        # Add up outfluxes from liver and other organs
        J_lungs = J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_comp(J_lungs_total, T_lh, t)
    # Return total flux into aorta
    return J_aorta_total

# # 10 parameters
def aorta_flux_10(t, J_lungs,
        T_l, T_h,
        E_organs, Tp_organs, Te_organs,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, 
        T_venous,
        tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_comp(J_lungs, T_l, t)
        J_aorta = dcmri.flux_comp(J_aorta, T_h, t)
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, 1/Ke_kidneys, t)
        J_organs = (1-E_liver)*dcmri.flux_2comp(J_organs, [Tp_organs, Te_organs], E_o, t)[0,0,:]
        # Add up outfluxes from liver and other organs
        J_lungs = J_kidneys + J_organs
        # Propgate through venous return
        J_lungs = dcmri.flux_plug(J_lungs, T_venous, t)
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_comp(J_lungs_total, T_l, t)
    J_aorta_total = dcmri.flux_comp(J_aorta_total, T_h, t)
    # Return total flux into aorta
    return J_aorta_total

# 10 params kidneys
def aorta_flux_10b(t, J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        E_liver,
        FF_kidneys, E_kidneys, Ke_kidneys, Ta_kidneys,
        tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_lungs, T_lh, D_lh, dt=t[1]-t[0])
        # Split into liver and other organs
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_kidneys)*J_aorta
        # Propagate through liver and other organs
        J_kidneys = dcmri.flux_plug(J_kidneys, Ta_kidneys, t)
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, 1/Ke_kidneys, t)
        J_organs = (1-E_liver)*dcmri.flux_2comp(J_organs, [Tp_organs, Te_organs], E_o, t)[0,0,:]
        # Add up outfluxes from liver and other organs
        J_lungs = J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_chain(J_lungs_total, T_lh, D_lh, dt=t[1]-t[0])
    # Return total flux into aorta
    return J_aorta_total

# 11 params liver kidneys
def aorta_flux_11(t, J_lungs,
        T_lh, 
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, 
        FF_kidneys, E_kidneys, Kp_kidneys,
        tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_comp(J_lungs, T_lh, t)
        # Split into liver, kidneys and other organs
        J_liver = FF_liver*J_aorta
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_liver-FF_kidneys)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = dcmri.flux_comp(J_liver, T_gut, t)
        J_liver = (1-E_liver)*dcmri.flux_nscomp(J_liver, 1/Ke_liver, t)
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, Kp_kidneys, t)
        J_organs = dcmri.flux_2comp(J_organs, [Tp_organs, Te_organs], E_o, t)[0,0,:]
        # Add up outfluxes from liver, kidneys and other organs
        J_lungs = J_liver + J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_comp(J_lungs_total, T_lh, t)
    # Return total flux into aorta
    return J_aorta_total

# 12 params liver kidneys
def aorta_flux_12(t, J_lungs,
        T_lh, D_lh,
        E_organs, Tp_organs, Te_organs,
        T_gut, FF_liver, E_liver, Ke_liver, 
        FF_kidneys, E_kidneys, Kp_kidneys,
        tol = 0.001):
    E_o = [[1-E_organs,1],[E_organs,0]]
    dose0 = np.trapz(J_lungs, t)
    dose = dose0
    min_dose = tol*dose0
    J_lungs_total = J_lungs
    while dose > min_dose:
        # Propagate through the lungs and heart
        J_aorta = dcmri.flux_chain(J_lungs, T_lh, D_lh, dt=t[1]-t[0])
        # Split into liver, kidneys and other organs
        J_liver = FF_liver*J_aorta
        J_kidneys = FF_kidneys*J_aorta
        J_organs = (1-FF_liver-FF_kidneys)*J_aorta
        # Propagate through liver, kidneys and other organs
        J_liver = dcmri.flux_comp(J_liver, T_gut, t)
        J_liver = (1-E_liver)*dcmri.flux_nscomp(J_liver, 1/Ke_liver, t)
        J_kidneys = (1-E_kidneys)*dcmri.flux_nscomp(J_kidneys, Kp_kidneys, t)
        J_organs = dcmri.flux_2comp(J_organs, [Tp_organs, Te_organs], E_o, t)[0,0,:]
        # Add up outfluxes from liver, kidneys and other organs
        J_lungs = J_liver + J_kidneys + J_organs
        # Add to the total flux into the lungs
        J_lungs_total += J_lungs
        # Get residual dose
        dose = np.trapz(J_lungs, t)
    # Propagate total flux through lungs
    J_aorta_total = dcmri.flux_chain(J_lungs_total, T_lh, D_lh, t[1]-t[0])
    # Return total flux into aorta
    return J_aorta_total