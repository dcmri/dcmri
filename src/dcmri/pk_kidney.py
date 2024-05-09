import numpy as np

import dcmri

def kidney_conc_9(
        ca, # Arterial plasma concentration (mmol/mL)
        Fp, # Arterial plasma flow (mL/sec or mL/sec/mL)
        Eg, # Glomerular extraction fraction
        fc, # Cortical fraction of the peritubular capillaries
        Tg, # Glomerular mean transit time (sec)
        Tv, # Mean transit time of peritubular capillaries & venous system (sec)
        Tpt, # Proximal tubuli mean transit time (sec)
        Tlh, # Lis of Henle mean transit time (sec)
        Tdt, # Distal tubuli mean transit time (sec)
        Tcd, # Collecting duct mean transit time (sec)
        t=None, dt=1.0):
    
    # Flux out of the glomeruli and arterial tree
    Jg = dcmri.flux_comp(Fp*ca, Tg, t=t, dt=dt)

    # Flux out of the peritubular capillaries and venous system
    Jv = dcmri.flux_comp((1-Eg)*Jg, Tv, t=t, dt=dt)

    # Flux out of the proximal tubuli
    Jpt = dcmri.flux_comp(Eg*Jg, Tpt, t=t, dt=dt)

    # Flux out of the lis of Henle
    Jlh = dcmri.flux_comp(Jpt, Tlh, t=t, dt=dt)

    # Flux out of the distal tubuli
    Jdt = dcmri.flux_comp(Jlh, Tdt, t=t, dt=dt)

    # Flux out of the collecting ducts
    Jcd = dcmri.flux_comp(Jdt, Tcd, t=t, dt=dt)

    # Cortex = arteries/glomeruli 
    # + part of the peritubular capillaries
    # + proximal and distal tubuli 
    Ccor = Tg*Jg + fc*Tv*Jv + Tpt*Jpt + Tdt*Jdt 

    # Medulla = lis of Henle 
    # + part of the peritubular capillaries
    # + collecting ducts 
    Cmed = (1-fc)*Tv*Jv + Tlh*Jlh + Tcd*Jcd

    return Ccor, Cmed
    