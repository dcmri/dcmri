from tqdm import tqdm
import numpy as np

import dcmri
from dcmri import const
from dcmri import phantoms
from dcmri.utils.misc import sample, add_noise
from dcmri.kinetics import ConcTissueX, ConcLiver, ConcCortMed
import dcmri.kinetics.lib as pk
from dcmri.bloch import Signal
import dcmri.bloch.lib as sig


def aif(
        tacq=180.0,
        dt=1.5,
        BAT=20,
        field_strength=3.0,
        agent='gadodiamide',
        H=0.45,
        R10a=1/const.T1(3.0, 'blood'),
        S0=150,
        model='3D-SPGR-SS',
        TR=0.005,
        FA=15,
        B1corr=1,
        TC=0.2,
        CNR=np.inf,
        dt_sim=0.1,
    ):
    """Synthetic AIF signal with noise.

    Args:
        tacq (float, optional): Duration of the acquisition in sec. Defaults to 180.
        dt (float, optional): Sampling interval in sec. Defaults to 1.5.
        BAT (int, optional): Bolus arrival time in sec. Defaults to 20.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        H (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.const.T1(3.0, 'blood').
        S0 (int, optional): Signal scaling factor in blood (arbitrary units). Defaults to 150.
        model (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 20.
        B1corr (float, optional): B1-correction factor. Defaults to 1.
        TC (float, optional): time to center of k-space in SR sequence. This is ignored when sequence='SS'. efaults to 0.2 sec.
        CNR (float, optional): Contrast-to-noise ratio, define as the ratio of signal-enhancement in the AIF to noise. Defaults to np.inf.
        dt_sim (float, optional): Sampling inteval of the forward modelling in sec. Defaults to 0.1.

    Returns: 
        tuple: time, aif, gt

        - **time**: array of time points.
        - **aif**: array of AIF signals.
        - **gt**: dictionary with ground truth values.
    """
    t = np.arange(0, tacq+dt, dt_sim)
    cp = dcmri.aif.parker(t, BAT)
    rp = const.r1(field_strength, 'plasma', agent)
    r2s = const.r1(field_strength, 'blood', agent)
    R1b = R10a + rp*cp*(1-H)
    R20sa = 20
    R2sb = R20sa + r2s * cp * (1-H)
    R1b = R1b.reshape(1, -1)
    if model == '3D-SPGR-SS':
        Mz = sig.Mz_spgr_in_ss(R1b, 1, 0, np.zeros_like(R1b), 1, TR, FA * B1corr)
    elif model == '3D-SR-SPGR-SS':
        Mz = sig.Mz_pr_spgr_in_ss(R1b, 1, 0, np.zeros_like(R1b), 1, TC, TR, FA * B1corr, 0, 2 * TC, 90) 
    aif_ = sig.mz_readout(Mz, np.zeros_like(Mz), S0, FA * B1corr, 0, 0)
    time = np.arange(0, tacq, dt)
    aif_ = sample(time, t, aif_, dt)
    sdev = (np.amax(aif_)-aif_[0])/CNR
    aif_ = add_noise(aif_, sdev)
    gt = {'t': t, 'cp': cp, 'cb': cp*(1-H),
          'TR': TR, 'FA': FA, 'B1corr':B1corr, 'S0b': S0}
    return time, aif_, gt


# TODO include DSC
def brain(
        n=192,
        tacq=180.0,
        dt=1.5,
        BAT=20,
        Tav=8,
        field_strength=3.0,
        agent='gadodiamide',
        H=0.45,
        R10a=1/const.T1(3.0, 'blood'),
        R20sa=20,
        S0=150,
        model='3D-SPGR-SS',
        TR=0.005,
        TE=0.002,
        FA=15,
        TC=0.2,
        CNR=np.inf,
        dt_sim=0.1,
        verbose=0,
    ):
    """Synthetic brain images data generated using Parker's AIF, the Shepp-Logan phantom and a two-compartment exchange tissue.

    Args:
        n (int, optional): matrix size. Defaults to 192.
        tacq (float, optional): Duration of the acquisition in sec. Defaults to 180.
        dt (float, optional): Sampling inteval in sec. Defaults to 1.5.
        BAT (int, optional): Bolus arrival time in sec. Defaults to 20.
        Tav (float, optional): Arterio-venous transit time. Defaults to 8 sec.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        H (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.const.T1(3.0, 'blood').
        S0 (int, optional): Signal scaling factor for tissue (arbitrary units). Defaults to 150.
        model (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 20.
        TC (float, optional): time to center of k-space in SR sequence. This is ignored when sequence='SS'. efaults to 0.2 sec.
        CNR (float, optional): Contrast-to-noise ratio, define as the ratio of signal-enhancement in the AIF to noise. Defaults to np.inf.
        dt_sim (float, optional): Sampling inteval of the forward modelling in sec. Defaults to 0.1.
        verbose (int, optional): if set to 0, no feedback is given during the computation. With verbose=1, an progress bar is shown. Defaults to 0.

    Returns: 
        tuple: time, signal, gt

        - **time**: array of time points.
        - **signal**: 3D array of pixel sigals, with dimensions (x,y,t).
        - **gt**: dictionary with ground truth values for concentrations and tissue parameters.

    Example:

        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.animation import ArtistAnimation
        >>> import dcmri as dc

        Generate data:

        >>> time, signal, aif, gt = dc.fake.brain(n=64, CNR=100)

        Display as animation:

        >>> fig, ax = plt.subplots()
        >>> ims = []
        >>> for i in range(time.size):
        >>>     im = ax.imshow(signal[:,:,i], cmap='magma', animated=True, vmin=0, vmax=30)
        >>>     ims.append([im])
        >>> ani = ArtistAnimation(
        >>>     fig, ims, interval=50, blit=True,
        >>>     repeat_delay=0)
        >>> plt.show()

        .. image:: ../_static/animations/fake_brain.gif
            :alt: Example animation

    """
    # Parameter maps
    roi = phantoms.shepp_logan(n=n)
    im = phantoms.shepp_logan('T1', 'PD', 'Fb', 'vb', 'PS', 'vi', n=n)

    # Input
    t = np.arange(0, tacq+dt, dt_sim)
    cp = dcmri.aif.parker(t, BAT)

    # Arterial signal
    rp = const.r1(field_strength, 'plasma', agent)
    r2s = const.r1(field_strength, 'blood', agent)
    R1b = R10a + rp * cp * (1-H)
    R2sb = R20sa + r2s * cp * (1-H)

    R1b = R1b.reshape(1, -1)
    if model == '3D-SPGR-SS':
        Mz = sig.Mz_spgr_in_ss(R1b, 1, 0, np.zeros_like(R1b), 1, TR, FA)
    elif model == '3D-SR-SPGR-SS':
        Mz = sig.Mz_pr_spgr_in_ss(R1b, 1, 0, np.zeros_like(R1b), 1, TC, TR, FA, 0, 2 * TC, 90) 
    aif_ = sig.mz_readout(Mz, R2sb, S0, FA, TE, 0)

    sdev = (np.amax(aif_)-aif_[0])/CNR
    time = np.arange(0, tacq, dt)
    aif_ = sample(time, t, aif_, dt)

    # Output
    conc = np.zeros((n, n, t.size))
    signal = np.zeros((n, n, time.size))
    signal_noisefree = np.zeros((n, n, time.size))

    # TissueX signal
    if verbose == 0:
        iter = range(n)
    else:
        iter = tqdm(range(n), desc='Generating fake brain data')

    for i in iter:
        for j in range(n):

            if roi['background'][i, j]:
                continue

            # Pixel concentration
            if roi['anterior artery'][i, j]:
                C = cp.copy()
            elif roi['sagittal sinus'][i, j]:
                C = pk.flux_comp(cp, Tav, dt=dt_sim)
            else:
                Fb = im['Fb'][i, j]
                vb = im['vb'][i, j]
                vi = im['vi'][i, j]
                PS = im['PS'][i, j]
                C = ConcTissueX('2CX')(
                    cp*(1-H), dt=dt_sim, 
                    H=H, Fb=Fb, vb=vb, vi=vi, PS=PS
                ).sum(axis=0)

            # Pixel signal
            R1 = 1/im['T1'][i, j] + rp * C
            R2s = R20sa + r2s * C

            R1 = R1.reshape(1, -1)
            if model == '3D-SPGR-SS':
                Mz = sig.Mz_spgr_in_ss(R1, 1, 0, np.zeros_like(R1), 1, TR, FA)
            elif model == '3D-SR-SPGR-SS':
                Mz = sig.Mz_pr_spgr_in_ss(R1, 1, 0, np.zeros_like(R1), 1, TC, TR, FA, 0, 2 * TC, 90) 
            s = sig.mz_readout(Mz, R2s, S0*im['PD'][i, j], FA, TE, 0)

            sig_noisefree = sample(time, t, s, dt)
            s = add_noise(sig_noisefree, sdev)

            # Save results
            conc[i, j, :] = C
            signal[i, j, :] = s
            signal_noisefree[i, j, :] = sig_noisefree

    gt = {'t': t, 'cp': cp, 'C': conc, 'cb': cp*(1-H),
          'signal': signal_noisefree,
          'Fp': im['Fb']*(1-H), 'vp': im['vb']*(1-H),
          'Fb': im['Fb'], 'vb': im['vb'],
          'PS': im['PS'], 'vi': im['vi'],
          'T1': im['T1'], 'PD': im['PD'],
          'TR': TR, 'FA': FA, 
          'S0': S0*im['PD']}
    
    gt['ve'] = np.where(gt['PS'] > 0, gt['vp'] + gt['vi'], gt['vp'])

    return time, signal, aif_, gt


def tissue(
    tacq=180.0,
    dt=1.5,
    BAT=20,
    Fb=0.02,  
    vb=0.1,
    PS=0.003,  
    vi=0.2, 
    field_strength=3.0,
    agent='gadodiamide',
    H=0.45,
    R10a=1/const.T1(3.0, 'blood'),
    R10=1/const.T1(3.0, 'muscle'),
    S0b=100,
    S0=150,
    model='3D-SPGR-SS',
    TR=0.005,
    FA=15,
    TC=0.2,
    CNR=np.inf,
    dt_sim=0.1,
):
    """Synthetic data generated using Parker's AIF and a two-compartment exchange tissue.

    Args:
        tacq (float, optional): Duration of the acquisition in sec. Defaults to 180.
        dt (float, optional): Sampling interval in sec. Defaults to 1.5.
        BAT (int, optional): Bolus arrival time in sec. Defaults to 20.
        Fb (float, optional): Blood flow in mL/sec/mL. Defaults to 0.1.
        vb (float, optional): Blood volume fraction. Defaults to 0.05.
        PS (float, optional): Permeability-surface area product in 1/sec. Defaults to 0.003.
        vi (float, optional): Interstitial volume fraction. Defaults to 0.3.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        H (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.const.T1(3.0, 'blood').
        R10 (int, optional): Precontrast relaxation rate for tissue in 1/sec. Defaults to 1.
        S0b (int, optional): Signal scaling factor for blood (arbitrary units). Defaults to 100.
        S0 (int, optional): Signal scaling factor for tissue (arbitrary units). Defaults to 150.
        model (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 20.
        TC (float, optional): time to center of k-space in SR sequence. This is ignored when sequence='SS'. efaults to 0.2 sec.
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
    cp = dcmri.aif.parker(t, BAT)
    C = ConcTissueX('2CX')(cp*(1-H), dt=dt_sim, H=H, Fb=Fb, vb=vb, PS=PS, vi=vi)
    rp = const.r1(field_strength, 'plasma', agent)
    r2s = const.r1(field_strength, 'blood', agent)
    R1b = R10a + rp*cp*(1-H)
    R20sa = 20
    R2sb = R20sa + r2s * cp * (1-H)
    R1 = R10 + rp * C.sum(axis=0)
    R2s = R20sa + r2s * C.sum(axis=0)
    if model == '3D-SPGR-SS':
        aif_ = Signal(model)(S0=S0b, R1=R1b, R2s=R2sb, TR=TR, FA=FA, TE=0)
        roi = Signal(model)(S0=S0, R1=R1, R2s=R2s, TR=TR, FA=FA, TE=0)
    elif model == '2D-SR-SPGR-SS':
        aif_ = Signal('3D-SR-SPGR-SS')(S0=S0b, R1=R1b, R2s=R2sb, TA=TR, PA=FA, TE=0) # TODO: Add a TC parameter to match the readout time wit tissue
        roi = Signal('2D-SR-SPGR-SS')(S0=S0b, R1=R1b, R2s=R2s, TC=TC, TR=TR, FA=FA, TE=0)

    time = np.arange(0, tacq, dt)
    aif_ = sample(time, t, aif_, dt)
    roi = sample(time, t, roi, dt)
    sdev = (np.amax(aif_) - aif_[0])/CNR
    aif_ = add_noise(aif_, sdev)
    roi = add_noise(roi, sdev)
    gt = {'t': t, 'cp': cp, 'C': C, 'cb': cp*(1-H),
          'Fb': Fb, 'vb': vb, 
          'Fp': Fb*(1-H), 'vp': vb*(1-H), 'PS': PS, 'vi': vi,
          'Ktrans': Fb*(1-H)*PS/(Fb*(1-H)+PS),
          'TR': TR, 'FA': FA, 'S0': S0}
    return time, aif_, roi, gt


def liver(
    tacq = 180.0,
    dt = 1.5,
    BAT = 20,
    Tg = 10,
    H = 0.45,
    ve = 0.3,
    Fp = 0.01,
    fa = 0.2,
    Ta = 2,
    E = 0.23,
    Th = 20*60,  
    field_strength = 3.0,
    agent = 'gadoxetate',
    R10a = 1/const.T1(3.0, 'blood'),
    R10 = 1/const.T1(3.0, 'liver'),
    S0b = 100,
    S0 = 150,
    sequence = '3D-SPGR-SS',
    TR = 0.005,
    FA = 15,
    TC = 0.5,
    CNR = np.inf,
    dt_sim = 0.1,
):
    """Synthetic data for liver tissue.

    Args:
        tacq (float, optional): Duration of the acquisition in sec. 
          Defaults to 180.
        dt (float, optional): Sampling interval in sec. Defaults to 1.5.
        BAT (float, optional): Bolus arrival time in sec. Defaults to 20.
        Tg (float, optional): Gut mean transit time. Defaults to 10 sec.
        H (float, optional): Hematocrit. Defaults to 0.45.
        ve (float, optional): Extracellular volume fraction. Defaults to 0.3.
        Fp (float, optional): Plasma flow in mL/sec/mL. Defaults to 0.01.
        fa (float, optional): Arterial flow fraction. Defaults to 0.2.
        Ta (float, optional): Arterial mean transit time. Defaults to 2 sec.
        E (float, optional): Hepatocellular extraction fraction. Defaults to 0.23.
        Th (float, optional): Hepatocyte mean transit time. Defaults to 20 
          mins.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 
          'gadoxetate'.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 
          1/sec. Defaults to 1/dc.const.T1(3.0, 'blood').
        R10 (int, optional): Precontrast relaxation rate for tissue in 1/sec. 
          Defaults to 1.
        S0b (int, optional): Signal scaling factor for blood (arbitrary 
          units). Defaults to 100.
        S0 (int, optional): Signal scaling factor for tissue (arbitrary 
          units). Defaults to 150.
        sequence (str, optional): Scanning sequences, either steady-state 
          ('SS') or Steady-state with inflow effects ('SSI').
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 20.
        TC (float, optional): Aortic inflow time for SSI sequence model. 
          This is ignored when sequence='SS'. Defaults to 0.2 sec.
        CNR (float, optional): Contrast-to-noise ratio, define as the ratio 
          of signal-enhancement in the AIF to noise. Defaults to np.inf.
        dt_sim (float, optional): Sampling inteval of the forward modelling 
          in sec. Defaults to 0.1.

    Returns: 
        tuple: time, aif, roi, gt

        - **time**: array of time points.
        - **aif**: array of AIF signals.
        - **vif**: array of VIF signals.
        - **roi**: array of ROI signals.
        - **gt**: dictionary with ground truth values for concentrations and 
          tissue parameters.
    """
    t = np.arange(0, tacq+dt, dt_sim)
    cp = dcmri.aif.parker(t, BAT)
    cv = pk.flux_comp(cp, Tg, t)
    ci = (cp*(1-H), cv*(1-H))
    # C = liver.conc_liver(
    #     ci, dt=dt_sim, sum=False, kinetics='2I-IC',
    #     ve=ve, Fp=Fp, fa=fa, Ta=Ta, E=E, Th=Th)
    C = ConcLiver('2I-IC')(
        ci, dt=dt_sim, sum=False,
        ve=ve, Fp=Fp, fa=fa, Ta=Ta, E=E, Th=Th)    
    rp = const.r1(field_strength, 'plasma', agent)
    rh = const.r1(field_strength, 'hepatocytes', agent)
    r2s = const.r1(field_strength, 'blood', agent)
    R20sa = 20
    R1a = R10a + rp*cp*(1-H)
    R1v = R10a + rp*cv*(1-H)
    R1 = R10 + rp*C[0, :] + rh*C[1, :]
    R2sa = R20sa + r2s*cp*(1-H)
    R2sv = R20sa + r2s*cv*(1-H)
    R2s = R20sa + r2s*C.sum(axis=0)

    if sequence == '3D-SPGR-SS':
        aif_ = Signal(sequence)(S0=S0b, R1=R1a, R2s=R2sa, TR=TR, FA=FA, TE=0)
        vif = Signal(sequence)(S0=S0b, R1=R1v, R2s=R2sv, TR=TR, FA=FA, TE=0)
        roi = Signal(sequence)(S0=S0, R1=R1, R2s=R2s, TR=TR, FA=FA, TE=0)
    elif sequence == '3D-SPGR-SSI':
        aif_ = Signal(sequence)(S0=S0b, R1=R1a, R2s=R2sa, TR=TR, FA=FA, TF=TC, TE=0)
        vif = Signal('3D-SPGR-SS')(S0=S0b, R1=R1v, R2s=R2sv, TR=TR, FA=FA, TE=0)
        roi = Signal('3D-SPGR-SS')(S0=S0, R1=R1, R2s=R2s, TR=TR, FA=FA, TE=0)

    time = np.arange(0, tacq, dt)
    aif_ = sample(time, t, aif_, dt)
    vif = sample(time, t, vif, dt)
    roi = sample(time, t, roi, dt)
    sdev = (np.amax(aif_)-aif_[0])/CNR
    aif_ = add_noise(aif_, sdev)
    vif = add_noise(vif, sdev)
    roi = add_noise(roi, sdev)
    gt = {'t': t, 'cp': cp, 'cv':cv*(1-H), 
          'C': np.sum(C,axis=0), 'cb': cp*(1-H),
          've': ve, 'Fp': Fp, 
          'Fb': Fp/(1-H), 'fa': fa, 'T_a': Ta, 'E': E, 'Th': Th,
          'TR': TR, 'FA': FA, 'S0': S0}
    return time, aif_, vif, roi, gt


def tissue2scan(
    tacq=180.0,
    tbreak=60,
    dt=1.5,
    BAT=20,
    Fb=0.1,
    vb=0.05,
    PS=0.003,
    vi=0.2,
    field_strength=3.0,
    agent='gadodiamide',
    H=0.45,
    R10a=1/const.T1(3.0, 'blood'),
    R10=1/const.T1(3.0, 'muscle'),
    S0b1=100,
    S01=150,
    S0b2=200,
    S02=300,
    model='3D-SPGR-SS',
    TR=0.005,
    FA=15,
    TC=0.2,
    CNR=np.inf,
    dt_sim=0.1,
):
    """Synthetic data generated using Parker's AIF, a two-compartment exchange tissue and a steady-state sequence.

    Args:
        tacq (float, optional): Duration of the acquisition in sec. Defaults to 180.
        tbreak (float, optional): Break time between the two scans in sec. Defaults to 60.
        dt (float, optional): Sampling inteval in sec. Defaults to 1.5.
        BAT (int, optional): Bolus arrival time in sec. Defaults to 20.
        Fb (float, optional): Blood flow in mL/sec/mL. Defaults to 0.1.
        vb (float, optional): Blood volume fraction. Defaults to 0.05.
        PS (float, optional): Permeability-surface area product in 1/sec. Defaults to 0.003.
        vi (float, optional): Interstitial volume fraction. Defaults to 0.3.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        H (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.const.T1(3.0, 'blood').
        R10 (int, optional): Precontrast relaxation rate for tissue in 1/sec. Defaults to 1.
        S0b1 (int, optional): Signal scaling factor for blood in the first scan (arbitrary units). Defaults to 100.
        S01 (int, optional): Signal scaling factor for tissue in the first scan (arbitrary units). Defaults to 150.
        S0b2 (int, optional): Signal scaling factor for blood in the second scan (arbitrary units). Defaults to 100.
        S02 (int, optional): Signal scaling factor for tissue in the second scan (arbitrary units). Defaults to 150.
        model (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 15.
        TC (float, optional): time to center of k-space in SR sequence. This is ignored when sequence='SS'. Defaults to 0.2 sec.
        CNR (float, optional): Contrast-to-noise ratio, define as the ratio of signal-enhancement in the AIF to noise. Defaults to np.inf.
        dt_sim (float, optional): Sampling inteval of the forward modelling in sec. Defaults to 0.1.

    Returns: 
        tuple: time, aif, roi, gt

        - **time**: tuple with two arrays of time points.
        - **aif**: tuple with two arrays of AIF signals.
        - **roi**: tuple with two arrays of ROI signals.
        - **gt**: dictionary with ground truth values for concentrations and tissue parameters.
    """
    # Simulate relaxation rates over the full time range
    t = np.arange(0, 2*tacq+tbreak+dt, dt_sim)
    cp = dcmri.aif.parker(t, BAT)
    cp += dcmri.aif.parker(t, tacq+tbreak+BAT)
    C = ConcTissueX('2CX')(cp*(1-H), dt=dt_sim, H=H, Fb=Fb, vb=vb, PS=PS, vi=vi)
    rp = const.r1(field_strength, 'plasma', agent)
    r2s = const.r1(field_strength, 'blood', agent)
    R1b = R10a + rp*cp*(1-H)
    R1 = R10 + rp * C.sum(axis=0)
    R20sa = 20
    R2sb = R20sa + r2s * cp * (1-H)
    R2s = R20sa + r2s * C.sum(axis=0)

    # Generate the signals from the first scan
    if model == '3D-SPGR-SS':
        aif_ = Signal(model)(S0=S0b1, R1=R1b, R2s=R2sb, TR=TR, FA=FA, TE=0)
        roi = Signal(model)(S0=S01, R1=R1, R2s=R2s, TR=TR, FA=FA, TE=0)
    elif model == '2D-SR-SPGR-SS':
        aif_ = Signal('3D-SR-SPGR-SS')(S0=S0b1, R1=R1b, R2s=R2sb, TA=TR, PA=FA, TE=0) # TODO: Add a TC parameter to match the readout time wit tissue
        roi = Signal('2D-SR-SPGR-SS')(S0=S01, R1=R1, R2s=R2s, TC=TC, TR=TR, FA=FA, TE=0)
    time1 = np.arange(0, tacq, dt)
    aif1 = sample(time1, t, aif_, dt)
    roi1 = sample(time1, t, roi, dt)
    sdev = (np.amax(aif1)-aif1[0])/CNR
    aif1 = add_noise(aif1, sdev)
    roi1 = add_noise(roi1, sdev)

    # Generate the second signals
    if model == '3D-SPGR-SS':
        aif_ = Signal(model)(S0=S0b2, R1=R1b, R2s=R2sb, TR=TR, FA=FA, TE=0)
        roi = Signal(model)(S0=S02, R1=R1, R2s=R2s, TR=TR, FA=FA, TE=0)
    elif model == '2D-SR-SPGR-SS':
        aif_ = Signal('3D-SR-SPGR-SS')(S0=S0b2, R1=R1b, R2s=R2sb, TA=TR, PA=FA, TE=0) # TODO: Add a TC parameter to match the readout time wit tissue
        roi = Signal('2D-SR-SPGR-SS')(S0=S02, R1=R1, R2s=R2s, TC=TC, TR=TR, FA=FA, TE=0)
    time2 = np.arange(tacq+tbreak, 2*tacq+tbreak, dt)
    aif2 = sample(time2, t, aif_, dt)
    roi2 = sample(time2, t, roi, dt)
    sdev = (np.amax(aif2)-aif2[0])/CNR
    aif2 = add_noise(aif2, sdev)
    roi2 = add_noise(roi2, sdev)

    # Build return values
    time = (time1, time2)
    aif_ = (aif1, aif2)
    roi = (roi1, roi2)
    gt = {'t': t, 'cp': cp, 'C': C, 'cb': cp*(1-H),
          'Fb': Fb, 'vb': vb, 'PS': PS, 'vi': vi,
          'Fp': Fb*(1-H), 'vp':vb*(1-H),
          'TR': TR, 'FA': FA, 'S01': S01, 'S02': S02}
    return time, aif_, roi, gt


def kidney(
    tacq=180.0,
    dt=1.5,
    BAT=20,
    Fp=0.03,
    Eg=0.15,
    fc=0.8,
    Tglom=4,
    Tv=10,
    Tpt=60,
    Tlh=60,
    Tdt=30,
    Tcd=30,
    field_strength=3.0,
    agent='gadoterate',
    Hct=0.45,
    R10a=1/const.T1(3.0, 'blood'),
    R10c=1/const.T1(3.0, 'kidney'),
    R10m=1/const.T1(3.0, 'kidney'),
    S0b=100,
    S0=150,
    model='2D-SR-SPGR-SS',
    TC=0.2,
    TR=0.005,
    FA=15,
    CNR=np.inf,
    dt_sim=0.1,
):
    """Synthetic data generated using Parker's AIF, and a multicompartment kidney model.

    Args:
        tacq (float, optional): Duration of the acquisition in sec. Defaults to 180.
        dt (float, optional): Sampling inteval in sec. Defaults to 1.5.
        BAT (float, optional): Bolus arrival time in sec. Defaults to 20.
        Fp (float, optional): Plasma flow in mL/sec/mL. Defaults to 0.03.
        Eg (float, optional): Glomerular extraction fraction. Defaults to 0.15.
        fc (float, optional): Cortical flow fraction. Defaults to 0.8.
        Tg (float, optional): Glomerular mean transit time in sec. Defaults to 4.
        Tv (float, optional): Peritubular & venous mean transit time in sec. Defaults to 10.
        Tpt (float, optional): Proximal tubuli mean transit time in sec. Defaults to 60.
        Tlh (float, optional): Lis of Henle mean transit time in sec. Defaults to 60.
        Tdt (float, optional): Distal tubuli mean transit time in sec. Defaults to 30.
        Tcd (float, optional): Collecting duct mean transit time in sec. Defaults to 30.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        Hct (float, optional): Hematocrit. Defaults to 0.45.
        R10a (float, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.const.T1(3.0, 'blood').
        R10c (float, optional): Precontrast relaxation rate for cortex in 1/sec. Defaults to 1.
        R10m (float, optional): Precontrast relaxation rate for cortex in 1/sec. Defaults to 1.
        S0b (float, optional): Signal scaling factor for blood (arbitrary units). Defaults to 100.
        S0 (float, optional): Signal scaling factor for tissue (arbitrary units). Defaults to 150.
        TC (float, optional): Time to readout of the k-space center in sec. Defaults to 0.2.
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (float, optional): Flip angle. Defaults to 15.
        CNR (float, optional): Contrast-to-noise ratio, define as the ratio of signal-enhancement in the AIF to noise. Defaults to np.inf.
        dt_sim (float, optional): Sampling interval of the forward modelling in sec. Defaults to 0.1.

    Returns: 
        tuple: time, aif, roi, gt

        - **time**: array of time points.
        - **aif**: array of AIF signals.
        - **roi**: tuple with arrays of cortex and medulla signals.
        - **gt**: dictionary with ground truth values for concentrations.
    """
    t = np.arange(0, tacq+dt, dt_sim)
    cp = dcmri.aif.parker(t, BAT)
    Cc, Cm = ConcCortMed(kinetics='7C', Fp=Fp, Eg=Eg, fc=fc, Tglom=Tglom, Tv=Tv, Tpt=Tpt, Tlh=Tlh, Tdt=Tdt, Tcd=Tcd)(cp, dt=dt_sim)
    rp = const.r1(field_strength, 'plasma', agent)
    r2s = const.r1(field_strength, 'blood', agent)
    R1b = R10a + rp*cp*(1-Hct)
    R1c = R10c + rp*Cc.sum(axis=0)
    R1m = R10m + rp*Cm.sum(axis=0)
    R20sa = 20
    R2sb = R20sa + r2s * cp * (1-Hct)
    R2sc = R20sa + r2s * Cc.sum(axis=0)
    R2sm = R20sa + r2s * Cm.sum(axis=0)

    if model == '3D-SPGR-SS':
        aif_ = Signal(model)(S0=S0b, R1=R1b, R2s=R2sb, TR=TR, FA=FA, TE=0)
        roic = Signal(model)(S0=S0, R1=R1c, R2s=R2sc, TR=TR, FA=FA, TE=0)
        roim = Signal(model)(S0=S0, R1=R1m, R2s=R2sm, TR=TR, FA=FA, TE=0)
    elif model == '2D-SR-SPGR-SS':
        aif_ = Signal('3D-SR-SPGR-SS')(S0=S0b, R1=R1b, R2s=R2sb, TA=TR, PA=FA, TE=0) # TODO: Add a TC parameter to match the readout time wit tissue
        roic = Signal('2D-SR-SPGR-SS')(S0=S0, R1=R1c, R2s=R2sc, TC=TC, TR=TR, FA=FA, TE=0)
        roim = Signal('2D-SR-SPGR-SS')(S0=S0, R1=R1m, R2s=R2sm, TC=TC, TR=TR, FA=FA, TE=0)

    time = np.arange(0, tacq, dt)
    aif_ = sample(time, t, aif_, dt)
    roic = sample(time, t, roic, dt)
    roim = sample(time, t, roim, dt)
    sdev = (np.amax(aif_)-aif_[0])/CNR
    aif_ = add_noise(aif_, sdev)
    roic = add_noise(roic, sdev)
    roim = add_noise(roim, sdev)
    gt = {'t': t, 'cp': cp, 'Cc': Cc, 'Cm': Cm, 'cb': cp*(1-Hct)}
    return time, aif_, (roic, roim), gt
