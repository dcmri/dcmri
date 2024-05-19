import numpy as np
import dcmri as dc

def make_tissue_2cm(
        tacq = 180, 
        dt = 1.5,
        BAT = 20,
        Fp = 0.1, 
        vp = 0.05, 
        PS = 0.003, 
        ve = 0.2,
        field_strength = 3.0,
        agent = 'gadodiamide', 
        Hct = 0.45,
        R10b = 1/dc.T1(3.0, 'blood'),
        R10 = 1/dc.T1(3.0,'muscle'),
        S0b = 100,
        S0 = 150,
        TR = 0.005,
        FA = 20,
        CNR = np.inf,
        dt_sim = 0.1,
    ):
    """Synthetic data generated using Parker's AIF, a two-compartment exchange tissue and a steady-state sequence.

    Args:
        tacq (int, optional): Duration of the acquisition in sec. Defaults to 180.
        dt (float, optional): Sampling inteval in sec. Defaults to 1.5.
        BAT (int, optional): Bolus arrival time in sec. Defaults to 20.
        Fp (float, optional): Plasma flow in mL/sec/mL. Defaults to 0.1.
        vp (float, optional): Plasma volume fraction. Defaults to 0.05.
        PS (float, optional): Permeability-surface area product in 1/sec. Defaults to 0.003.
        ve (float, optional): Extravascular, exctracellular volume fraction. Defaults to 0.3.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        Hct (float, optional): Hematocrit. Defaults to 0.45.
        R10b (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.T1(3.0, 'blood').
        R10 (int, optional): Precontrast relaxation rate for tissue in 1/sec. Defaults to 1.
        S0b (int, optional): Signal scaling factor for blood (arbitrary units). Defaults to 100.
        S0 (int, optional): Signal scaling factor for tissue (arbitrary units). Defaults to 150.
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 20.
        CNR (float, optional): Contrast-to-noise ratio, define as the ratio of signal-enhancement in the AIF to noise. Defaults to np.inf.
        dt_sim (float, optional): Sampling inteval of the forward modelling in sec. Defaults to 0.1.

    Returns: 
        tuple: time, aif, roi, gt

        - **time**: array of time points.
        - **aif**: array of AIF signals.
        - **roi**: array of ROI signals.
        - **gt**: dictionary with ground truth values for concentrations and tissue parameters.
    """
    t = np.arange(0, tacq+dt, dt_sim)
    cp = dc.aif_parker(t, BAT)
    C = dc.conc_2cxm(cp, Fp, vp, PS, ve, dt=dt_sim)
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1b = R10b + rp*cp*(1-Hct)
    R1 = R10 + rp*C
    aif = dc.signal_spgress(R1b, S0b, TR, FA)
    roi = dc.signal_spgress(R1, S0, TR, FA)
    time = np.arange(0, tacq, dt)
    aif = dc.sample(t, aif, time, dt)
    roi = dc.sample(t, roi, time, dt)
    sdev = (np.amax(aif)-aif[0])/CNR
    aif = dc.add_noise(aif, sdev)
    roi = dc.add_noise(roi, sdev)
    gt = {'t':t, 'cp':cp, 'C':C, 
          'Fp':Fp, 'vp':vp, 'PS':PS, 've':ve, 
          'TR':TR, 'FA':FA, 'S0':S0}
    return time, aif, roi, gt
    

def dro_aif_1(
        tacq = 180, 
        dt = 1.5, 
        BAT = 20, 
        field_strength = 3.0, 
        agent = 'gadoterate', 
        S0 = 1, 
        TR = 0.005, 
        FA = 20,
        CNR = np.inf,
    ):
    t = np.arange(0, tacq, dt)
    cb = dc.aif_parker(t, BAT)*(1-0.45)
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1 = 1/dc.T1(field_strength, 'blood') + rp*cb
    signal = dc.signal_spgress(R1, S0, TR, FA)
    sdev = (np.amax(signal)-signal[0])/CNR
    signal = dc.add_noise(signal, sdev)
    return t, signal, cb


def dro_aif_2(
        tacq = 180, 
        dt = 1.5, 
        BAT = 20, 
        field_strength = 3.0, 
        agent = 'gadoterate', 
        S0 = 1, 
        TD = 0.180, 
        CNR = np.inf,
    ):
    t = np.arange(0, tacq, dt)
    cb = dc.aif_parker(t, BAT)*(1-0.45)
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1 = 1/dc.T1(field_strength, 'blood') + rp*cb
    signal = dc.signal_sr(R1, S0, TD)
    sdev = (np.amax(signal)-signal[0])/CNR
    signal = dc.add_noise(signal, sdev)
    return t, signal, cb


def dro_aif_3(
        tacq = 180, 
        tbreak = 60,
        dt = 1.5, 
        BAT = 20, 
        field_strength = 3.0, 
        agent = 'gadoterate', 
        S01 = 1,
        S02 = 2,
        TR = 0.005, 
        FA = 20.0,
        CNR = np.inf,
    ):
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R10 = 1/dc.T1(field_strength, 'blood')

    t1 = np.arange(0, tacq, dt)
    cb1 = dc.aif_parker(t1, BAT)*(1-0.45) 
    R1 = R10 + rp*cb1
    S = dc.signal_spgress(R1, S01, TR, FA)
    sdev = (np.amax(S)-S[0])/CNR
    S1 = dc.add_noise(S, sdev)

    t2 = np.arange(tacq+tbreak, 2*tacq+tbreak, dt)
    cb2 = dc.aif_parker(t2, BAT)*(1-0.45)
    cb2 += dc.aif_parker(t2, tacq+tbreak+BAT)*(1-0.45)
    R1 = R10 + rp*cb2
    S = dc.signal_spgress(R1, S02, TR, FA)
    sdev = (np.amax(S)-S[0])/CNR
    S2 = dc.add_noise(S, sdev)
    
    return ( 
        np.concatenate((t1,t2)),
        np.concatenate((S1,S2)), 
        np.concatenate((cb1,cb2)), 
    )


