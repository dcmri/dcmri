import numpy as np

import dcmri as dc

def kidney_conc_pf(
        ca, # Arterial plasma concentration (mmol/mL) 
        Fp,
        Tp,
        Ft,
        h,
        TT = [15,30,60,90,150,300,600],
        t=None, dt=1.0):

    Cp = dc.conc_plug(Fp*ca, Tp, t=t, dt=dt) 
    vp = Tp*(Fp+Ft)
    cp = Cp/vp
    Ct = dc.conc_free(Ft*cp, h, dt=dt, TT=TT, solver='step')   
    return Cp, Ct


def kidney_conc_cm9(
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
    Jg = dc.flux_comp(Fp*ca, Tg, t=t, dt=dt)

    # Flux out of the peritubular capillaries and venous system
    Jv = dc.flux_comp((1-Eg)*Jg, Tv, t=t, dt=dt)

    # Flux out of the proximal tubuli
    Jpt = dc.flux_comp(Eg*Jg, Tpt, t=t, dt=dt)

    # Flux out of the lis of Henle
    Jlh = dc.flux_comp(Jpt, Tlh, t=t, dt=dt)

    # Flux out of the distal tubuli
    Jdt = dc.flux_comp(Jlh, Tdt, t=t, dt=dt)

    # Flux out of the collecting ducts
    Jcd = dc.flux_comp(Jdt, Tcd, t=t, dt=dt)

    # Cortex = arteries/glomeruli 
    # + part of the peritubular capillaries
    # + proximal and distal tubuli 
    Ccor = Tg*Jg + fc*Tv*Jv + Tpt*Jpt + Tdt*Jdt 

    # Medulla = lis of Henle 
    # + part of the peritubular capillaries
    # + collecting ducts 
    Cmed = (1-fc)*Tv*Jv + Tlh*Jlh + Tcd*Jcd

    return Ccor, Cmed
    