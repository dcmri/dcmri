from tqdm import tqdm
import numpy as np
import dcmri as dc


def fake_aif(
    tacq=180.0,
    dt=1.5,
    BAT=20,
    field_strength=3.0,
    agent='gadodiamide',
    Hct=0.45,
    R10a=1/dc.T1(3.0, 'blood'),
    S0b=150,
    sequence='SS',
    TR=0.005,
    FA=15,
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
        Hct (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.T1(3.0, 'blood').
        S0b (int, optional): Signal scaling factor in blood (arbitrary units). Defaults to 150.
        sequence (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
        TR (float, optional): Repetition time in sec. Defaults to 0.005.
        FA (int, optional): Flip angle. Defaults to 20.
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
    cp = dc.aif_parker(t, BAT)
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1b = R10a + rp*cp*(1-Hct)
    if sequence == 'SS':
        aif = dc.signal_ss(R1b, S0b, TR, FA)
    elif sequence == 'SR':
        aif = dc.signal_src(R1b, S0b, TC)
    time = np.arange(0, tacq, dt)
    aif = dc.sample(time, t, aif, dt)
    sdev = (np.amax(aif)-aif[0])/CNR
    aif = dc.add_noise(aif, sdev)
    gt = {'t': t, 'cp': cp, 'cb': cp*(1-Hct),
          'TR': TR, 'FA': FA, 'S0b': S0b}
    return time, aif, gt


# TODO align API with Tissue
def fake_brain(
    n=192,
    tacq=180.0,
    dt=1.5,
    BAT=20,
    Tav=8,
    field_strength=3.0,
    agent='gadodiamide',
    Hct=0.45,
    R10a=1/dc.T1(3.0, 'blood'),
    S0=150,
    sequence='SS',
    TR=0.005,
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
        Hct (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.T1(3.0, 'blood').
        S0 (int, optional): Signal scaling factor for tissue (arbitrary units). Defaults to 150.
        sequence (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
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

        >>> time, signal, aif, gt = dc.fake_brain(n=64, CNR=100)

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

        .. image:: ../../_static/animations/fake_brain.gif
            :alt: Example animation

    """
    # Parameter maps
    roi = dc.shepp_logan(n=n)
    im = dc.shepp_logan('T1', 'PD', 'BF', 'BV', 'PS', 'IV', n=n)

    # Input
    t = np.arange(0, tacq+dt, dt_sim)
    cp = dc.aif_parker(t, BAT)

    # Arterial signal
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1b = R10a + rp*cp*(1-Hct)
    if sequence == 'SS':
        aif = dc.signal_ss(R1b, S0, TR, FA)
    elif sequence == 'SR':
        aif = dc.signal_src(R1b, S0, TC)
    sdev = (np.amax(aif)-aif[0])/CNR
    time = np.arange(0, tacq, dt)
    aif = dc.sample(time, t, aif, dt)

    # Output
    conc = np.zeros((n, n, t.size))
    signal = np.zeros((n, n, time.size))
    signal_noisefree = np.zeros((n, n, time.size))

    # Tissue signal
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
                C = dc.flux_comp(cp, Tav, dt=dt_sim)
            else:
                Fp = im['BF'][i, j]*(1-Hct)
                vp = im['BV'][i, j]*(1-Hct)
                vi = im['IV'][i, j]
                PS = im['PS'][i, j]
                C = dc.conc_tissue(
                    cp, dt=dt_sim, kinetics='2CX', Fp=Fp, vp=vp, vi=vi, PS=PS)

            # Pixel signal
            R1 = 1/im['T1'][i, j] + rp*C
            if sequence == 'SS':
                sig = dc.signal_ss(R1, S0*im['PD'][i, j], TR, FA)
            elif sequence == 'SR':
                sig = dc.signal_sr(R1, S0*im['PD'][i, j], TR, FA, TC)
            sig_noisefree = dc.sample(time, t, sig, dt)
            sig = dc.add_noise(sig_noisefree, sdev)

            # Save results
            conc[i, j, :] = C
            signal[i, j, :] = sig
            signal_noisefree[i, j, :] = sig_noisefree

    gt = {'t': t, 'cp': cp, 'C': conc, 'cb': cp*(1-Hct),
          'signal': signal_noisefree,
          'Fp': im['BF']*(1-Hct), 'vp': im['BV']*(1-Hct),
          'PS': im['PS'], 'vi': im['IV'],
          'T1': im['T1'], 'PD': im['PD'],
          'TR': TR, 'FA': FA, 'S0': S0*im['PD']}

    gt['ve'] = gt['vi'] + gt['vp']

    return time, signal, aif, gt


def fake_tissue(
    tacq=180.0,
    dt=1.5,
    BAT=20,
    Fp=0.01,  # TODO should be 0.01
    vp=0.05,
    PS=0.003,  # TODO replace Ktrans
    ve=0.2,  # TODO: rename vi
    field_strength=3.0,
    agent='gadodiamide',
    Hct=0.45,
    R10a=1/dc.T1(3.0, 'blood'),
    R10=1/dc.T1(3.0, 'muscle'),
    S0b=100,
    S0=150,
    sequence='SS',
    TR=0.005,
    FA=15,
    TC=0.2,
    CNR=np.inf,
    dt_sim=0.1,
):
    """Synthetic data generated using Parker's AIF and a two-compartment exchange tissue.

    Args:
        tacq (float, optional): Duration of the acquisition in sec. Defaults to 180.
        dt (float, optional): Sampling inteval in sec. Defaults to 1.5.
        BAT (int, optional): Bolus arrival time in sec. Defaults to 20.
        Fp (float, optional): Plasma flow in mL/sec/mL. Defaults to 0.1.
        vp (float, optional): Plasma volume fraction. Defaults to 0.05.
        PS (float, optional): Permeability-surface area product in 1/sec. Defaults to 0.003.
        ve (float, optional): Extravascular, exctracellular volume fraction. Defaults to 0.3.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        Hct (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.T1(3.0, 'blood').
        R10 (int, optional): Precontrast relaxation rate for tissue in 1/sec. Defaults to 1.
        S0b (int, optional): Signal scaling factor for blood (arbitrary units). Defaults to 100.
        S0 (int, optional): Signal scaling factor for tissue (arbitrary units). Defaults to 150.
        sequence (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
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
    cp = dc.aif_parker(t, BAT)
    C = dc.conc_tissue(cp, dt=dt_sim, kinetics='2CX',
                       Fp=Fp, vp=vp, PS=PS, vi=ve)
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1b = R10a + rp*cp*(1-Hct)
    R1 = R10 + rp*C
    if sequence == 'SS':
        aif = dc.signal_ss(R1b, S0b, TR, FA)
        roi = dc.signal_ss(R1, S0, TR, FA)
    elif sequence == 'SR':
        aif = dc.signal_src(R1b, S0b, TC)
        roi = dc.signal_sr(R1, S0, TR, FA, TC)
    time = np.arange(0, tacq, dt)
    aif = dc.sample(time, t, aif, dt)
    roi = dc.sample(time, t, roi, dt)
    sdev = (np.amax(aif)-aif[0])/CNR
    aif = dc.add_noise(aif, sdev)
    roi = dc.add_noise(roi, sdev)
    gt = {'t': t, 'cp': cp, 'C': C, 'cb': cp*(1-Hct),
          'Fp': Fp, 'vp': vp, 'PS': PS, 'vi': ve,
          'Ktrans': Fp*PS/(Fp+PS),
          'TR': TR, 'FA': FA, 'S0': S0}
    return time, aif, roi, gt


def fake_tissue2scan(
    tacq=180.0,
    tbreak=60,
    dt=1.5,
    BAT=20,
    Fp=0.1,
    vp=0.05,
    PS=0.003,
    ve=0.2,
    field_strength=3.0,
    agent='gadodiamide',
    Hct=0.45,
    R10a=1/dc.T1(3.0, 'blood'),
    R10=1/dc.T1(3.0, 'muscle'),
    S0b1=100,
    S01=150,
    S0b2=200,
    S02=300,
    sequence='SS',
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
        Fp (float, optional): Plasma flow in mL/sec/mL. Defaults to 0.1.
        vp (float, optional): Plasma volume fraction. Defaults to 0.05.
        PS (float, optional): Permeability-surface area product in 1/sec. Defaults to 0.003.
        ve (float, optional): Extravascular, exctracellular volume fraction. Defaults to 0.3.
        field_strength (float, optional): B0 field in T. Defaults to 3.0.
        agent (str, optional): Contrast agent generic name. Defaults to 'gadodiamide'.
        Hct (float, optional): Hematocrit. Defaults to 0.45.
        R10a (_type_, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.T1(3.0, 'blood').
        R10 (int, optional): Precontrast relaxation rate for tissue in 1/sec. Defaults to 1.
        S0b1 (int, optional): Signal scaling factor for blood in the first scan (arbitrary units). Defaults to 100.
        S01 (int, optional): Signal scaling factor for tissue in the first scan (arbitrary units). Defaults to 150.
        S0b2 (int, optional): Signal scaling factor for blood in the second scan (arbitrary units). Defaults to 100.
        S02 (int, optional): Signal scaling factor for tissue in the second scan (arbitrary units). Defaults to 150.
        sequence (str, optional): Scanning sequences, either steady-state ('SS') or saturation-recovery ('SR')
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
    cp = dc.aif_parker(t, BAT)
    cp += dc.aif_parker(t, tacq+tbreak+BAT)
    C = dc.conc_tissue(cp, dt=dt_sim, kinetics='2CX',
                       Fp=Fp, vp=vp, PS=PS, vi=ve)
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1b = R10a + rp*cp*(1-Hct)
    R1 = R10 + rp*C

    # Generate the signals from the first scan
    if sequence == 'SS':
        aif = dc.signal_ss(R1b, S0b1, TR, FA)
        roi = dc.signal_ss(R1, S01, TR, FA)
    elif sequence == 'SR':
        aif = dc.signal_src(R1b, S0b1, TC)
        roi = dc.signal_sr(R1, S01, TR, FA, TC)
    time1 = np.arange(0, tacq, dt)
    aif1 = dc.sample(time1, t, aif, dt)
    roi1 = dc.sample(time1, t, roi, dt)
    sdev = (np.amax(aif1)-aif1[0])/CNR
    aif1 = dc.add_noise(aif1, sdev)
    roi1 = dc.add_noise(roi1, sdev)

    # Generate the second signals
    if sequence == 'SS':
        aif = dc.signal_ss(R1b, S0b2, TR, FA)
        roi = dc.signal_ss(R1, S02, TR, FA)
    elif sequence == 'SR':
        aif = dc.signal_src(R1b, S0b2, TC)
        roi = dc.signal_sr(R1, S02, TR, FA, TC)
    time2 = np.arange(tacq+tbreak, 2*tacq+tbreak, dt)
    aif2 = dc.sample(time2, t, aif, dt)
    roi2 = dc.sample(time2, t, roi, dt)
    sdev = (np.amax(aif2)-aif2[0])/CNR
    aif2 = dc.add_noise(aif2, sdev)
    roi2 = dc.add_noise(roi2, sdev)

    # Build return values
    time = (time1, time2)
    aif = (aif1, aif2)
    roi = (roi1, roi2)
    gt = {'t': t, 'cp': cp, 'C': C, 'cb': cp*(1-Hct),
          'Fp': Fp, 'vp': vp, 'PS': PS, 'vi': ve,
          'TR': TR, 'FA': FA, 'S01': S01, 'S02': S02}
    return time, aif, roi, gt


def fake_kidney_cortex_medulla(
    tacq=180.0,
    dt=1.5,
    BAT=20,
    Fp=0.03,
    Eg=0.15,
    fc=0.8,
    Tg=4,
    Tv=10,
    Tpt=60,
    Tlh=60,
    Tdt=30,
    Tcd=30,
    field_strength=3.0,
    agent='gadoterate',
    Hct=0.45,
    R10a=1/dc.T1(3.0, 'blood'),
    R10c=1/dc.T1(3.0, 'kidney'),
    R10m=1/dc.T1(3.0, 'kidney'),
    S0b=100,
    S0=150,
    sequence='SR',
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
        R10a (float, optional): Precontrast relaxation rate for blood in 1/sec. Defaults to 1/dc.T1(3.0, 'blood').
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
    cp = dc.aif_parker(t, BAT)
    Cc, Cm = dc.conc_kidney_cortex_medulla(
        cp, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, dt=dt_sim, kinetics='7C')
    rp = dc.relaxivity(field_strength, 'plasma', agent)
    R1b = R10a + rp*cp*(1-Hct)
    R1c = R10c + rp*Cc
    R1m = R10m + rp*Cm
    if sequence == 'SS':
        aif = dc.signal_ss(R1b, S0b, TR, FA)
        roic = dc.signal_ss(R1c, S0, TR, FA)
        roim = dc.signal_ss(R1m, S0, TR, FA)
    elif sequence == 'SR':
        aif = dc.signal_src(R1b, S0b, TC)
        roic = dc.signal_sr(R1c, S0, TR, FA, TC)
        roim = dc.signal_sr(R1m, S0, TR, FA, TC)
    time = np.arange(0, tacq, dt)
    aif = dc.sample(time, t, aif, dt)
    roic = dc.sample(time, t, roic, dt)
    roim = dc.sample(time, t, roim, dt)
    sdev = (np.amax(aif)-aif[0])/CNR
    aif = dc.add_noise(aif, sdev)
    roic = dc.add_noise(roic, sdev)
    roim = dc.add_noise(roim, sdev)
    gt = {'t': t, 'cp': cp, 'Cc': Cc, 'Cm': Cm, 'cb': cp*(1-Hct)}
    return time, aif, (roic, roim), gt
