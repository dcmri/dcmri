import numpy as np
import dcmri

def aorta_signal_6(xdata,
        BAT, CO, T_hl, D_hl, Tp_o, E_b,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        R10 = 1.0,                  # Precontrast relaxation rate (1/sec)
        sample = True,              # If false return the pseudocontinuous signal
        return_conc = False,
    ):

    # Calculate
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    conc = dcmri.ca_conc(agent)
    Ji = dcmri.inj_step(t, weight, conc, dose, rate, BAT) #mmol/sec
    Jb = dcmri.aorta_flux_4(Ji, # mmol/sec
            T_hl, D_hl, Tp_o, E_b, 
            dt=dt, tol=dose_tolerance)
    cb = Jb/CO  # M = (mmol/sec) / (mL/sec) 
    if return_conc:
        return t, cb
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*cb
    signal = dcmri.signal_spgress(TR, FA, R1, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[2]-xdata[1])

def aorta_signal_7(xdata,
        BAT, CO, T_hl, E_o, Tp_o, Te_o, E_b,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        R10 = 1.0,                  # Precontrast relaxation rate (1/sec)
        sample = True,              # If false return the pseudocontinuous signal
        return_conc = False,
    ):

    # Calculate
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    conc = dcmri.ca_conc(agent)
    Ji = dcmri.inj_step(t, weight, conc, dose, rate, BAT) #mmol/sec
    Jb = dcmri.aorta_flux_5(Ji, # mmol/sec
            T_hl,
            E_o, Tp_o, Te_o,
            E_b, 
            dt=dt, tol=dose_tolerance)
    cb = Jb/CO  # M = (mmol/sec) / (mL/sec) 
    if return_conc:
        return t, cb
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*cb
    signal = dcmri.signal_spgress(TR, FA, R1, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[2]-xdata[1])


def aorta_signal_8(xdata,
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        R10 = 1.0,                  # Precontrast relaxation rate (1/sec)
        sample = True,              # If false return the pseudocontinuous signal
        return_conc = False,
    ):

    # Calculate
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    conc = dcmri.ca_conc(agent)
    Ji = dcmri.inj_step(t, weight, conc, dose, rate, BAT) #mmol/sec
    Jb = dcmri.aorta_flux_6(Ji, # mmol/sec
            T_hl, D_hl,
            E_o, Tp_o, Te_o,
            E_b, 
            dt=dt, tol=dose_tolerance)
    cb = Jb/CO  # M = (mmol/sec) / (mL/sec) 
    if return_conc:
        return t, cb
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*cb
    signal = dcmri.signal_spgress(TR, FA, R1, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[2]-xdata[1])


def aorta_signal_8b(xdata,
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        TD = 180/1000.0,            # Delay time (sec)
        R10 = 1.0,                  # Precontrast relaxation rate (1/sec)
        sample = True,              # If false return the pseudocontinuous signal
        return_conc = False,
    ):

    # Calculate
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    conc = dcmri.ca_conc(agent) #mmol/mL
    Ji = dcmri.inj_step(t, weight, conc, dose, rate, BAT) #mmol/sec
    Jb = dcmri.aorta_flux_6(Ji, #mmol/sec
            T_hl, D_hl,
            E_o, Tp_o, Te_o,
            E_b, 
            dt=dt, tol=dose_tolerance)
    cb = Jb/CO  # M #mmol/sec / (mL/sec)
    if return_conc:
        return t, cb
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*cb
    signal = dcmri.signal_sr(R1, S0, TD)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def aorta_signal_9(xdata,
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b, TI,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        R10 = 1.0,                  # Precontrast relaxation rate (1/sec)
        sample = True,              # If false return the pseudocontinuous signal
        return_conc = False,
    ):

    # Calculate
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    conc = dcmri.ca_conc(agent) #mmol/mL
    Ji = dcmri.inj_step(t, weight, conc, dose, rate, BAT) #mmol/sec
    Jb = dcmri.aorta_flux_6(Ji, #mmol/sec
            T_hl, D_hl,
            E_o, Tp_o, Te_o,
            E_b, 
            dt=dt, tol=dose_tolerance)
    cb = Jb/CO  # M #mmol/sec / (mL/sec)
    if return_conc:
        return t, cb
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*cb
    signal = dcmri.signal_eqspgre(R1, S0, TR, FA, TI)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[2]-xdata[1])


def tissue_signal_3(xdata,
        vp, Ktrans, ve,
        R10 = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        aif = None,                 # High-res AIF - tuple (t, cb)
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        sample = True,
        return_conc = False,
        ):
    t = aif[0]
    ca = aif[1]/(1-Hct)
    C = dcmri.conc_etofts(ca, vp, Ktrans, Ktrans/ve, t)
    if return_conc:
        return t, C
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*C
    signal = dcmri.signal_spgress(TR, FA, R1, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[2]-xdata[1])

def tissue_signal_4(xdata,
        Fp, vp, PS, ve,
        R10 = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        aif = None,                 # High-res AIF - tuple (t, cb)
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        sample = True,
        return_conc = False,
        ):
    t = aif[0]
    ca = aif[1]/(1-Hct)
    C = dcmri.conc_2cxm(ca, Fp, vp, PS, ve, t)
    if return_conc:
        return t, C
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*C
    signal = dcmri.signal_spgress(TR, FA, R1, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_signal_4(xdata,
        Fp, Tp, Ft, Tt,
        R10 = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,                  # AIF conc in M
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        return_conc = False,
        sample = True,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(cb))
    vp = Tp*(Fp+Ft)
    ca = cb/(1-Hct)
    cp = dcmri.conc_comp(Fp*ca, Tp, t)
    Cp = vp*cp
    Ct = dcmri.conc_comp(Ft*cp, Tt, t)
    if return_conc:
        return t, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1k = R10 + rp*Cp + rp*Ct
    signal = dcmri.signal_spgress(TR, FA, R1k, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_signal_4b(xdata,
        Fp, Tp, Ft, Tt,
        R10 = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        Tsat = 0,                   # time before start of readout
        TD = 85/1000,               # Time to the center of the readout pulse
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        return_conc = False,
        sample = True,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(cb))
    vp = Tp*(Fp+Ft)
    ca = cb/(1-Hct)
    cp = dcmri.conc_comp(Fp*ca, Tp, t)
    Cp = vp*cp
    Ct = dcmri.conc_comp(Ft*cp, Tt, t)
    if return_conc:
        return t, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*Cp + rp*Ct
    signal = dcmri.signal_srspgre(R1, S0, TR, FA, Tsat, TD)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_signal_5(xdata,
        Ta, FF_k, F_k, Tp, Tt,
        R10k = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        J_aorta = None,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        kidney_volume = None, 
        return_conc = False,
        sample = True,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(J_aorta))
    E_k = F_k/(1+F_k)
    J_kidneys = FF_k*J_aorta
    J_kidneys = dcmri.flux_plug(J_kidneys, Ta, t, solver='interp')
    Np = dcmri.conc_comp(J_kidneys, Tp, t)
    Nt = dcmri.conc_comp(E_k*Np/Tp, Tt, t)
    Cp = Np/kidney_volume # 
    Ct = Nt/kidney_volume # M
    if return_conc:
        return t, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1k = R10k + rp*Cp + rp*Ct
    signal = dcmri.signal_spgress(TR, FA, R1k, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_signal_5b(xdata,
        Fp, Tp, Ft, Tt, Ta,
        R10 = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        Tsat = 0,                   # time before start of readout
        TD = 85/1000,               # Time to the center of the readout pulse
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        return_conc = False,
        sample = True,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(cb))
    vp = Tp*(Fp+Ft)
    ca = cb/(1-Hct)
    ca = dcmri.flux_plug(ca, Ta, t)
    cp = dcmri.conc_comp(Fp*ca, Tp, t)
    Cp = vp*cp
    Ct = dcmri.conc_comp(Ft*cp, Tt, t)
    if return_conc:
        return t, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*Cp + rp*Ct
    signal = dcmri.signal_srspgre(R1, S0, TR, FA, Tsat, TD)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_signal_6(xdata,
        Fp, Tp, Ft, Tt, Ta, S0,
        R10 = 1,
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        Tsat = 0,                   # time before start of readout
        TD = 85/1000,               # Time to the center of the readout pulse
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        return_conc = False,
        sample = True,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(cb))
    vp = Tp*(Fp+Ft)
    ca = cb/(1-Hct)
    ca = dcmri.flux_plug(ca, Ta, t)
    cp = dcmri.conc_comp(Fp*ca, Tp, t)
    Cp = vp*cp
    Ct = dcmri.conc_comp(Ft*cp, Tt, t)
    if return_conc:
        return t, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1 = R10 + rp*Cp + rp*Ct
    signal = dcmri.signal_srspgre(R1, S0, TR, FA, Tsat, TD)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_signal_9(xdata,
        FF_k, F_k, Tv, h0,h1,h2,h3,h4,h5,
        R10 = 1,
        S0 = 1,                     # Baseline signal (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        J_aorta = None,             # mmol/sec
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        kidney_volume = None,
        return_conc = False,
        sample = True,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(J_aorta))
    E_k = F_k/(1+F_k)
    Kvp = 1/Tv
    Kp = Kvp/(1-E_k)
    H = [h0,h1,h2,h3,h4,h5]
    TT = [15,30,60,90,150,300,600]
    J_kidneys = FF_k*J_aorta
    Np = dcmri.conc_plug(J_kidneys, Tv, t, solver='interp') 
    Nt = dcmri.conc_free(E_k*Kp*Np, H, dt=t[1]-t[0], TT=TT, solver='step')
    Cp = Np/kidney_volume # M
    Ct = Nt/kidney_volume # M
    if return_conc:
        return t, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1k = R10 + rp*Cp + rp*Ct
    signal = dcmri.signal_spgress(TR, FA, R1k, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def kidney_cm_signal_9(tacq,
        Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd,
        R10c = 1,
        R10m = 1,
        S0c = 1,                    # Baseline signal cortex (a.u.)
        S0m = 1,                    # Baseline signal medulla (a.u.)
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FAc = 15.0,                  # Flip angle cortex (degrees)
        FAm = 15.0,                  # Flip angle medulla (degrees)
        Tsat = 0,                   # time before start of readout
        TD = 85/1000,               # Time to the center of the readout pulse
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        Hct = 0.45, 
        return_conc = False,
        sample = True,
        ):
    t = dt*np.arange(len(cb))
    ca = cb/(1-Hct)
    Cc, Cm = dcmri.kidney_conc_9(
        ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, dt=dt)
    if return_conc:
        return t, Cc, Cm
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1c = R10c + rp*Cc
    R1m = R10m + rp*Cm
    Sc = dcmri.signal_srspgre(R1c, S0c, TR, FAc, Tsat, TD)
    Sm = dcmri.signal_srspgre(R1m, S0m, TR, FAm, Tsat, TD)
    if not sample:
        return t, Sc, Sm
    nt = int(len(tacq)/2)
    Sc_meas = dcmri.sample(t, Sc, tacq[:nt], tacq[1]-tacq[0])
    Sm_meas = dcmri.sample(t, Sm, tacq[nt:], tacq[1]-tacq[0])
    return np.concatenate((Sc_meas, Sm_meas))


def liver_signal_5(xdata,
        Te, De, ve, k_he, Th, 
        S0 = 1,
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,
        Hct = 0.45,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        R10 = 1,
        field_strength = 3.0,       # Field strength (T)
        sample = True,
        return_conc = False,
        ):
    # Propagate through the gut
    ca = dcmri.flux_pfcomp(cb, Te, De, dt=dt, solver='interp')
    #ca = dcmri.flux_comp(cb, Te, t)
    #ca = dcmri.flux_plug(ca, Td, t, solver='interp')
    # Tissue concentration in the extracellular space
    Ce = ve*ca/(1-Hct)
    # Tissue concentration in the hepatocytes
    Ch = dcmri.conc_comp(k_he*ca, Th, dt=dt)
    if return_conc:
        return t, Ce, Ch
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', 'Primovist')
    rh = dcmri.relaxivity(field_strength, 'hepatocytes', 'Primovist')
    R1 = R10 + rp*Ce + rh*Ch
    t = dt*np.arange(len(cb))
    signal = dcmri.signal_spgress(TR, FA, R1, S0)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


def liver_signal_9(xdata,
        Te, De, ve, k_he_i, k_he_f, Th_i, Th_f, S01, S02,
        dt = 0.5,                   # Internal time resolution (sec)
        cb = None,
        Hct = 0.45,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        R10 = 1,
        tR12 = 1,
        sample = True,
        return_conc = False,
        ):
    #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    t = dt*np.arange(len(cb))
    k_he = dcmri.interp(t, [k_he_i, k_he_f])
    # Interpolating Kbh here for consistency with original model
    Kbh = dcmri.interp(t, [1/Th_i, 1/Th_f])
    # Propagate through the gut
    ca = dcmri.flux_pfcomp(cb, Te, De, dt=dt, solver='interp')
    # Tissue concentration in the extracellular space
    Ce = ve*ca/(1-Hct)
    # Tissue concentration in the hepatocytes
    Ch = dcmri.conc_nscomp(k_he*ca, 1/Kbh, t)
    if return_conc:
        return t, Ce, Ch
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', 'Primovist')
    rh = dcmri.relaxivity(field_strength, 'hepatocytes', 'Primovist')
    R1 = R10 + rp*Ce + rh*Ch
    signal = dcmri.signal_spgress(TR, FA, R1, S01)
    t2 = (t >= tR12)
    signal[t2] = dcmri.signal_spgress(TR, FA, R1[t2], S02)
    if not sample:
        return t, signal
    return dcmri.sample(t, signal, xdata, xdata[1]-xdata[0])


# Modified ad-hoc, needs checking
def aorta_kidney_signal_17(xdata,
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_l,
        FF_k, F_k, Tv, Ta, h0,h1,h2,h3,h4,h5,
        R10b = 1,
        R10k = 1,
        S0k = 1,                     # Baseline signal (a.u.)
        S0b = 1,
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        kidney_volume = None, 
        return_conc = False,
        sample = True,
        ):
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    # Aorta
    conc = dcmri.ca_conc(agent)
    Ji = dcmri.inj_step(t, # mmol/sec
        weight, conc, dose, rate, BAT)
    J_aorta = dcmri.aorta_flux_10b(t, Ji,
        T_hl, D_hl,
        E_o, Tp_o, Te_o,
        E_l,
        FF_k, E_k, Kp, Ta,
        tol=dose_tolerance)
    cb = J_aorta/CO # M
    # Kidney
    E_k = F_k/(1+F_k)
    Kvp = 1/Tv
    Kp = Kvp/(1-E_k)
    H = [h0,h1,h2,h3,h4,h5]
    TT = [15,30,60,90,150,300,600]
    J_kidneys = FF_k*J_aorta
    J_kidneys = dcmri.flux_plug(J_kidneys, Ta, t)
    Np = dcmri.conc_comp(J_kidneys, 1/Kp, t)
    Nt = dcmri.conc_free(E_k*Kp*Np, H, dt=t[1]-t[0], TT=TT, solver='step')
    Cp = Np/kidney_volume # M
    Ct = Nt/kidney_volume # M
    if return_conc:
        return t, cb, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', agent)
    R1k = R10k + rp*Cp + rp*Ct
    R1b = R10b + rp*cb
    Sb = dcmri.signal_spgress(TR, FA, R1b, S0b)
    Sk = dcmri.signal_spgress(TR, FA, R1k, S0k)
    if not sample:
        return t, Sb, Sk
    Sb = dcmri.sample(t, Sb, xdata, xdata[1]-xdata[0])
    Sk = dcmri.sample(t, Sk, xdata, xdata[1]-xdata[0])
    return np.concatenate([Sb, Sk])

def aorta_liver_signal_14(xdata,
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, 
        FF_l, E_l, E_k, 
        Te, De, Tel, Th,
        R10b = 1,
        R10l = 1,
        S0l = 1,                    # Baseline signal (a.u.)
        S0b = 1,
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        agent = 'Primovist',
        dose_tolerance = 0.1,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        Hct = 0.45, 
        field_strength = 3.0,       # Field strength (T)
        liver_volume = None, 
        return_conc = False,
        sample = True,
        ):
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    # Aorta
    conc = dcmri.ca_conc(agent)
    Ji = dcmri.inj_step(t, weight, conc, dose, rate, BAT) # mmol/sec
    J_aorta = dcmri.aorta_flux_9(Ji,
        T_hl, D_hl,
        (1-E_k)*(1-FF_l), E_o, Tp_o, Te_o,
        (1-E_l)*FF_l, Te, De,
        dt=dt, tol=dose_tolerance)
    # Liver
    cb = J_aorta/CO # M
    #ca = pkmods.flux_pfcomp(cb, Te, De, dt=dt, solver='interp')
    #Ce = ve*ca/(1-Hct)
    #Ch = dcmri.conc_comp(k_he*ca, Th, t)
    J_aorta = dcmri.flux_pfcomp(J_aorta, Te, De, dt=dt, solver='interp')
    Ne = Tel*FF_l*J_aorta
    Nh = dcmri.conc_comp(E_l*FF_l*J_aorta, Th, t)
    Ce, Ch = Ne/liver_volume, Nh/liver_volume
    if return_conc:
        return t, cb, Ce, Ch
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', 'Primovist')
    rh = dcmri.relaxivity(field_strength, 'hepatocytes', 'Primovist')
    R1l = R10l + rp*Ce + rh*Ch
    R1b = R10b + rp*cb
    Sb = dcmri.signal_spgress(TR, FA, R1b, S0b)
    Sl = dcmri.signal_spgress(TR, FA, R1l, S0l)
    if not sample:
        return t, Sb, Sl
    Sb = dcmri.sample(t, Sb, xdata, xdata[1]-xdata[0])
    Sl = dcmri.sample(t, Sl, xdata, xdata[1]-xdata[0])
    return np.concatenate([Sb, Sl])


def aorta_kidney_liver_signal(xdata,
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o,
        T_g, FF_l, E_l, Te, Th,
        FF_k, F_k, Tv, Ta, h0,h1,h2,h3,h4,h5,
        R10l = 1,
        R10b = 1,
        R10k = 1,
        S0l = 1,
        S0k = 1,                     # Baseline signal (a.u.)
        S0b = 1,
        dt = 0.5,                   # Internal time resolution (sec)
        weight = 70.0,              # Patient weight in kg
        dose = 0.025,               # mL per kg bodyweight (quarter dose)
        rate = 1,                   # Injection rate (mL/sec)
        dose_tolerance = 0.1,
        TR = 3.71/1000.0,           # Repetition time (sec)
        FA = 15.0,                  # Nominal flip angle (degrees)
        field_strength = 3.0,       # Field strength (T)
        agent = 'Dotarem',
        kidney_volume = None, 
        liver_volume = None,
        return_conc = False,
        sample = True,
        ):
    t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
    # Derived params
    E_k = F_k/(1+F_k)
    Kvp = 1/Tv
    Kp = Kvp/(1-E_k)
    # Aorta
    conc = dcmri.ca_conc(agent)
    Ji = dcmri.inj_step(t, # mmol/sec
        weight, conc, dose, rate, BAT)
    J_aorta = dcmri.aorta_flux_12(t, Ji,
        T_hl, D_hl,
        E_o, Tp_o, Te_o,
        T_g, FF_l, E_l, 1/Te, 
        FF_k, E_k, Kp,
        tol=dose_tolerance)
    cb = J_aorta/CO # M
    # Liver
    J_liver = FF_l*J_aorta
    J_liver = dcmri.flux_comp(J_liver, T_g, t)
    Ne = Te*FF_l*J_aorta
    Nh = dcmri.conc_comp(E_l*FF_l*J_aorta, Th, t)
    Ce_l = Ne/liver_volume # M
    Ch_l = Nh/liver_volume # M
    # Kidney
    H = [h0,h1,h2,h3,h4,h5]
    TT = [15,30,60,90,150,300,600]
    J_kidneys = FF_k*J_aorta
    J_kidneys = dcmri.flux_plug(J_kidneys, Ta, t)
    Np = dcmri.conc_comp(J_kidneys, 1/Kp, t)
    Nt = dcmri.conc_free(E_k*Kp*Np, H, dt=t[1]-t[0], TT=TT, solver='step')
    Cp = Np/kidney_volume # M
    Ct = Nt/kidney_volume # M
    if return_conc:
        return t, cb, Ce_l, Ch_l, Cp, Ct
    # Return R
    rp = dcmri.relaxivity(field_strength, 'plasma', 'Primovist')
    rh = dcmri.relaxivity(field_strength, 'hepatocytes', 'Primovist')
    R1l = R10l + rp*Ce_l + rh*Ch_l
    R1k = R10k + rp*Cp + rp*Ct
    R1b = R10b + rp*cb
    Sb = dcmri.signal_spgress(TR, FA, R1b, S0b)
    Sl = dcmri.signal_spgress(TR, FA, R1l, S0l)
    Sk = dcmri.signal_spgress(TR, FA, R1k, S0k)
    if not sample:
        return t, Sb, Sl, Sk
    Sb = dcmri.sample(t, Sb, xdata, xdata[1]-xdata[0])
    Sl = dcmri.sample(t, Sl, xdata, xdata[1]-xdata[0])
    Sk = dcmri.sample(t, Sk, xdata, xdata[1]-xdata[0])
    return np.concatenate([Sb, Sl, Sk])

