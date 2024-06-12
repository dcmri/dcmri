import numpy as np
import dcmri as dc
from scipy.stats import rv_histogram


class Kidney2CFXSR(dc.Model):
    """Delayed two-compartment filtration model with fast water exchange acquired with a saturation-recovery sequence.

        The free model parameters are:

        - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
        - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
        - **Tp** (Plasma mean transit time, sec): Transit time of the plasma compartment. 
        - **Ft** (Tubular flow, mL/sec/mL): Flow of fluid into the tubuli.
        - **Tt** (Tubular mean transit time, sec): Transit time of the tubular compartment. 
        - **Ta** (Arterial delay time, sec): Transit time through the arterial compartment.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Sampling interval of the AIF in sec. 
        Hct (float, optional): Hematocrit. 
        agent (str, optional): Contrast agent generic name.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        Tsat (float, optional): Time before start of readout (sec).
        TC (float, optional): Time to the center of the readout pulse
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec. 
        R10b (float, optional): Precontrast arterial relaxation rate in 1/sec. 
        t0 (float, optional): Baseline length (sec).
        vol (float, optional): Kidney volume in mL 

    See Also:
        `KidneyPFFXSS`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_sr` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_sr(R10=1/dc.T1(3.0,'kidney'))
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.Kidney2CFXSR(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TC = 0.200,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'kidney'),
        ...     R10b = 1/dc.T1(3.0,'blood'),
        ...     t0 = 15,
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> #
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
        >>> ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()
        >>> #
        >>> ax1.set_title('Reconstruction of concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
        >>> ax1.plot(time/60, 1000*model.conc(), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 

    dt = 0.5 
    Hct = 0.45 
    agent = 'gadoterate'
    field_strength = 3.0
    R10 = 1  
    R10b = 1
    t0 = 0
    vol = None

    TR = 0.005  
    FA = 15.0
    Tsat = 0
    TC = 0.085

    S0 = 1
    Fp = 200/6000
    Tp = 5
    Ft = 30/6000
    Tt = 120
    Ta = 0

    free = ['Fp','Tp','Ft','Tt','Ta']
    bounds = (
        0,
        [np.inf, 8, np.inf, np.inf, 3],
    )

    def __init__(self, aif, **kwargs):
        super().__init__(**kwargs)
        n0 = max([round(self.t0/self.dt),1])
        self.r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        cb = dc.conc_src(aif, self.TC, 1/self.R10b, self.r1, n0)
        self.t = self.dt*np.arange(np.size(aif))
        self.ca = cb/(1-self.Hct)        #: Arterial plasma concentration (M)

    def conc(self):
        """Tissue concentration

        Returns:
            numpy.ndarray: Concentration in M
        """
        ca = dc.flux_plug(self.ca, self.Ta, dt=self.dt)
        return dc.kidney_conc_2cm(ca, self.Fp, self.Tp, self.Ft, self.Tt, dt=self.dt)

    def predict(self, xdata):
        if np.amax(self.t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF.'
            raise ValueError(msg) 
        C = self.conc()
        R1 = self.R10 + self.r1*C
        signal = dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC, self.Tsat)
        return dc.sample(xdata, self.t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_sr(self.R10, 1, self.TR, self.FA, self.TC, self.Tsat)
        n0 = max([np.sum(xdata<self.t0), 1])
        self.S0 = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)

    def pars(self):
        pars = {}
        pars['S0'] = ['Signal scaling factor', self.S0, 'a.u.']
        pars['Fp'] = ['Plasma flow', self.Fp, 'mL/sec/mL']
        pars['Tp'] = ['Plasma mean transit time', self.Tp, 'sec']
        pars['Ft'] = ['Tubular flow', self.Ft, 'mL/sec/mL']
        pars['Tt'] = ['Tubular mean transit time', self.Tt, 'sec']
        pars['Ta'] = ['Arterial mean transit time', self.Ta, 'sec']
        pars['Fb'] = ['Blood flow',self.Fp/(1-self.Hct),'mL/sec/mL']
        pars['ve'] = ['Extracellular volume', self.Fp*self.Tp, '']
        pars['FF'] = ['Filtration fraction', self.Ft/self.Fp, '']
        pars['E'] = ['Extraction fraction', self.Ft/(self.Ft+self.Fp), '']
        if self.vol is None:
            return self.add_sdev(pars)
        pars['SK-GFR'] = ['Single-kidney glomerular filtration rate', self.Ft*self.vol, 'mL/sec']
        pars['SK-RBF'] = ['Single-kidney renal blood flow', self.Ft*self.vol/(1-self.Hct), 'mL/sec']
        return self.add_sdev(pars)



class Kidney2CFXSS(dc.Model):
    """Delayed two-compartment filtration model with fast water exchange acquired with a steady-state sequence.

        The free model parameters are:

        - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
        - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
        - **Tp** (Plasma mean transit time, sec): Transit time of the plasma compartment. 
        - **Ft** (Tubular flow, mL/sec/mL): Flow of fluid into the tubuli.
        - **Tt** (Tubular mean transit time, sec): Transit time of the tubular compartment. 
        - **Ta** (Arterial delay time, sec): Transit time through the arterial compartment.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Sampling interval of the AIF in sec. 
        Hct (float, optional): Hematocrit. 
        agent (str, optional): Contrast agent generic name.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec. 
        R10b (float, optional): Precontrast arterial relaxation rate in 1/sec. 
        t0 (float, optional): Baseline length (sec).
        vol (float, optional): Kidney volume in mL 

    See Also:
        `Kidney2CFXSR`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(R10 = 1/dc.T1(3.0,'kidney'))
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.Kidney2CFXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'kidney'),
        ...     R10b = 1/dc.T1(3.0,'blood'),
        ...     t0 = 15,
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> #
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
        >>> ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()
        >>> #
        >>> ax1.set_title('Reconstruction of concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
        >>> ax1.plot(time/60, 1000*model.conc(), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 

    dt = 0.5 
    Hct = 0.45 
    agent = 'gadoterate'
    field_strength = 3.0
    R10 = 1  
    R10b = 1
    t0 = 0
    vol = None

    TR = 0.005  
    FA = 15.0

    S0 = 1
    Fp = 200/6000
    Tp = 5
    Ft = 30/6000
    Tt = 120
    Ta = 0

    free = ['Fp','Tp','Ft','Tt','Ta']
    bounds = (
        0,
        [np.inf, 8, np.inf, np.inf, 3],
    )

    def __init__(self, aif, **kwargs):
        super().__init__(**kwargs)
        n0 = max([round(self.t0/self.dt),1])
        self.r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        cb = dc.conc_ss(aif, self.TR, self.FA, 1/self.R10b, self.r1, n0)
        self.t = self.dt*np.arange(np.size(aif))
        self.ca = cb/(1-self.Hct)        #: Arterial plasma concentration (M)

    def conc(self):
        """Tissue concentration

        Returns:
            numpy.ndarray: Concentration in M
        """
        ca = dc.flux_plug(self.ca, self.Ta, dt=self.dt)
        return dc.kidney_conc_2cm(ca, self.Fp, self.Tp, self.Ft, self.Tt, dt=self.dt)

    def predict(self, xdata):
        if np.amax(self.t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF.'
            raise ValueError(msg) 
        C = self.conc()
        R1 = self.R10 + self.r1*C
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, self.t, signal, xdata[1]-xdata[0])

    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata<self.t0), 1])
        self.S0 = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)

    def pars(self):
        pars = {}
        pars['S0'] = ['Signal scaling factor', self.S0, 'a.u.']
        pars['Fp'] = ['Plasma flow', self.Fp, 'mL/sec/mL']
        pars['Tp'] = ['Plasma mean transit time', self.Tp, 'sec']
        pars['Ft'] = ['Tubular flow', self.Ft, 'mL/sec/mL']
        pars['Tt'] = ['Tubular mean transit time', self.Tt, 'sec']
        pars['Ta'] = ['Arterial mean transit time', self.Ta, 'sec']
        pars['Fb'] = ['Blood flow',self.Fp/(1-self.Hct),'mL/sec/mL']
        pars['ve'] = ['Extracellular volume', self.Fp*self.Tp, '']
        pars['FF'] = ['Filtration fraction', self.Ft/self.Fp, '']
        pars['E'] = ['Extraction fraction', self.Ft/(self.Ft+self.Fp), '']
        if self.vol is None:
            return self.add_sdev(pars)
        pars['SK-GFR'] = ['Single-kidney glomerular filtration rate', self.Ft*self.vol, 'mL/sec']
        pars['SK-RBF'] = ['Single-kidney renal blood flow', self.Ft*self.vol/(1-self.Hct), 'mL/sec']
        return self.add_sdev(pars)



class KidneyPFFXSS(dc.Model):
    """Plug flow plasma compartment with model-free nephron in fast water exchange acquired with a steady-state spoiled gradient echo sequence.

        The free model parameters are:

        - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
        - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
        - **Tp** (Plasma mean transit time, sec): Transit time of the plasma compartment. 
        - **Ft** (Tubular flow, mL/sec/mL): Flow of fluid into the tubuli.
        - **h0** (Transit time frequency, in units of 1/sec): Frequency of the transit times in the first bin (default range 15-30s).
        - **h1** (Transit time frequency, in units of 1/sec): Frequency of the transit times in the second bin (default range 30-60s).
        - **h2** (Transit time frequency, in units of 1/sec): Frequency of the transit times in the third bin (default range 60-90s).
        - **h3** (Transit time frequency, in units of 1/sec): Frequency of the transit times in the fourth bin (default range 90-150s).
        - **h4** (Transit time frequency, in units of 1/sec): Frequency of the transit times in the fifth bin (default range 150-300s).
        - **h5** (Transit time frequency, in units of 1/sec): Frequency of the transit times in the sixth bin (default range 300-600s).

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Sampling interval of the AIF in sec. 
        Hct (float, optional): Hematocrit. 
        agent (str, optional): Contrast agent generic name.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec. 
        R10b (float, optional): Precontrast arterial relaxation rate in 1/sec. 
        t0 (float, optional): Baseline length (sec).
        TT (array-like, optional): 7-elements array with boundaries of the transit time histogram bins (sec).
        vol (float, optional): Kidney volume in mL 

    See Also:
        `Kidney2CFXSR`, `Kidney2CFXSS`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=100, R10 = 1/dc.T1(3.0,'kidney'))
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.KidneyPFFXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'kidney'),
        ...     R10b = 1/dc.T1(3.0,'blood'),
        ...     t0 = 15,
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> #
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
        >>> ax0.plot(time/60, model.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()
        >>> #
        >>> ax1.set_title('Reconstruction of concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
        >>> ax1.plot(time/60, 1000*model.conc(), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 

    dt = 0.5 
    Hct = 0.45 
    agent = 'gadoterate'
    field_strength = 3.0
    R10 = 1  
    R10b = 1
    t0 = 0
    vol = None

    TR = 0.005  
    FA = 15.0

    S0 = 1
    Fp = 200/6000
    Tp = 5
    Ft = 30/6000
    h0 = 1
    h1 = 1
    h2 = 1
    h3 = 1
    h4 = 1
    h5 = 1
    TT = [15,30,60,90,150,300,600]

    free = ['Fp','Tp','Ft','h0','h1','h2','h3','h4','h5']
    bounds = (
        0,
        [np.inf, 8, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    )
    
    def __init__(self, aif, **kwargs):
        super().__init__(**kwargs)
        n0 = max([round(self.t0/self.dt),1])
        self.r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        cb = dc.conc_ss(aif, self.TR, self.FA, 1/self.R10b, self.r1, n0)
        self.t = self.dt*np.arange(np.size(aif))
        self.ca = cb/(1-self.Hct)

    def conc(self):
        """Tissue concentration

        Returns:
            numpy.ndarray: Concentration in M
        """
        H = [self.h0,self.h1,self.h2,self.h3,self.h4,self.h5]
        return dc.kidney_conc_pf(self.ca, self.Fp, self.Tp, self.Ft, H, TT=self.TT, dt=self.dt)

    def predict(self, xdata):
        if np.amax(self.t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF.'
            raise ValueError(msg) 
        C = self.conc()
        R1 = self.R10 + self.r1*C
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, self.t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata<self.t0), 1])
        self.S0 = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)

    def pars(self):
        H = [self.h0,self.h1,self.h2,self.h3,self.h4,self.h5]
        TT = rv_histogram((H,self.TT), density=True)
        pars = {}
        pars['S0'] = ['Signal scaling factor', self.S0, 'a.u.']
        pars['Fp'] = ['Plasma flow', self.Fp, 'mL/sec/mL']
        pars['Tp'] = ['Plasma mean transit time', self.Tp, 'sec']
        pars['Ft'] = ['Tubular flow', self.Ft, 'mL/sec/mL']
        pars['h0'] = ["Transit time weight (15-30s)", self.h0, '1/sec']
        pars['h1'] = ["Transit time weight (30-60s)", self.h1, '1/sec']
        pars['h2'] = ["Transit time weight (60-90s)", self.h2, '1/sec']
        pars['h3'] = ["Transit time weight (90-150s)", self.h3, '1/sec']
        pars['h4'] = ["Transit time weight (150-300s)", self.h4, '1/sec']
        pars['h5'] = ["Transit time weight (300-600s)", self.h5, '1/sec']
        pars['Fb'] = ['Blood flow',self.Fp/(1-self.Hct),'mL/sec/mL']
        pars['ve'] = ['Extracellular volume', self.Fp*self.Tp, '']
        pars['FF'] = ['Filtration fraction', self.Ft/self.Fp, '']
        pars['E'] = ['Extraction fraction', self.Ft/(self.Ft+self.Fp), '']
        pars['Tt'] = ['Tubular mean transit time', TT.mean(), 'sec']
        pars['Dt'] = ['Tubular transit time dispersion', TT.std()/TT.mean(), '']
        if self.vol is None:
            return self.add_sdev(pars)
        pars['SK-GFR'] = ['Single-kidney glomerular filtration rate', self.Ft*self.vol, 'mL/sec']
        pars['SK-RBF'] = ['Single-kidney renal blood flow', self.Ft*self.vol/(1-self.Hct), 'mL/sec']
        return self.add_sdev(pars)


class KidneyCortMedSR(dc.Model):
    """
    Corticomedullary multi-compartment model in fast water exchange acquired with a saturation-recovery sequence.

        The free model parameters are:

        - **Fp** Plasma flow in mL/sec/mL. 
        - **Eg** Glomerular extraction fraction. 
        - **fc** Cortical flow fraction. 
        - **Tg** Glomerular mean transit time in sec. 
        - **Tv** Peritubular & venous mean transit time in sec. 
        - **Tpt** Proximal tubuli mean transit time in sec. 
        - **Tlh** Lis of Henle mean transit time in sec. 
        - **Tdt** Distal tubuli mean transit time in sec. 
        - **Tcd** Collecting duct mean transit time in sec.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Sampling interval of the AIF in sec. 
        Hct (float, optional): Hematocrit. 
        agent (str, optional): Contrast agent generic name.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        Tsat (float, optional): Time before start of readout (sec).
        TC (float, optional): Time to the center of the readout pulse
        R10c (float, optional): Precontrast cortex relaxation rate in 1/sec. 
        R10m (float, optional): Precontrast medulla relaxation rate in 1/sec.
        R10b (float, optional): Precontrast arterial relaxation rate in 1/sec. 
        S0c (float, optional): Signal scaling factor in the cortex (a.u.).
        S0m (float, optional): Signal scaling factor in the medulla (a.u.).
        t0 (float, optional): Baseline length (sec).
        vol (float, optional): Kidney volume in mL.

    See Also:
        `Kidney2CFXSR`, `Kidney2CFXSS`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_sr` to generate synthetic test data:

        >>> time, aif, roic, roim, gt = dc.make_kidney_cm_sr(CNR=100)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.KidneyCortMedSR(aif,
        ...     dt = time[1],
        ...     Hct = 0.45,
        ...     agent = 'gadoterate',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     TC = 0.2,
        ...     R10c = 1/dc.T1(3.0,'kidney'),
        ...     R10m = 1/dc.T1(3.0,'kidney'),
        ...     R10b = 1/dc.T1(3.0,'blood'),
        ...     t0 = 15,
        ...     )

        Train the model on the ROI data and predict signals and concentrations:

        >>> model.train(time, roic, roim)
        >>> roic_pred, roim_pred = model.predict(time)
        >>> concc, concm = model.conc()

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(time/60, roic, marker='o', linestyle='None', color='cornflowerblue', label='Cortex data')
        >>> ax0.plot(time/60, roic_pred, linestyle='-', linewidth=3.0, color='darkblue', label='Cortex prediction')
        >>> ax0.plot(time/60, roim, marker='x', linestyle='None', color='cornflowerblue', label='Medulla data')
        >>> ax0.plot(time/60, roim_pred, linestyle='--', linewidth=3.0, color='darkblue', label='Medulla prediction')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()

        >>> ax1.set_title('Reconstruction of concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['Cc'], marker='o', linestyle='None', color='cornflowerblue', label='Cortex ground truth')
        >>> ax1.plot(time/60, 1000*concc, linestyle='-', linewidth=3.0, color='darkblue', label='Cortex prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['Cm'], marker='x', linestyle='None', color='cornflowerblue', label='Medulla ground truth')
        >>> ax1.plot(time/60, 1000*concm, linestyle='--', linewidth=3.0, color='darkblue', label='Medulla prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()
    """ 

    dt = 0.5
    Hct = 0.45 
    agent = 'gadoterate'
    field_strength = 3.0
    TR = 0.005  
    FA = 15.0
    Tsat = 0
    TC = 0.085
    R10c = 1
    R10m = 1  
    R10b = 1
    S0c = 1
    S0m = 1
    t0 = 0
    vol = None

    Fp = 0.03
    Eg = 0.15
    fc = 0.8
    Tg = 4
    Tv = 10
    Tpt = 60
    Tlh = 60
    Tdt = 30
    Tcd = 30

    free = ['Fp','Eg','fc','Tg','Tv','Tpt','Tlh','Tdt','Tcd']
    bounds = (
        [0.01, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 10, 30, np.inf, np.inf, np.inf, np.inf],
    )

    def __init__(self, aif, **kwargs):
        super().__init__(**kwargs)
        n0 = max([round(self.t0/self.dt),1])
        self.r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        cb = dc.conc_src(aif, self.TC, 1/self.R10b, self.r1, n0)
        self.t = self.dt*np.arange(np.size(aif))
        self.ca = cb/(1-self.Hct)        #: Arterial plasma concentration (M)

    def conc(self):
        """Cortical and medullary concentration

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Concentration in cortex, concentration in medulla, in M
        """
        return dc.kidney_conc_cm9(self.ca, 
            self.Fp, self.Eg, self.fc, self.Tg, self.Tv, 
            self.Tpt, self.Tlh, self.Tdt, self.Tcd, dt=self.dt)
        
    def predict(self, xdata:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        Cc, Cm = self.conc()
        R1c = self.R10c + self.r1*Cc
        R1m = self.R10m + self.r1*Cm
        Sc = dc.signal_sr(R1c, self.S0c, self.TR, self.FA, self.TC, self.Tsat)
        Sm = dc.signal_sr(R1m, self.S0m, self.TR, self.FA, self.TC, self.Tsat)
        return (
            dc.sample(xdata, self.t, Sc, xdata[2]-xdata[1]), 
            dc.sample(xdata, self.t, Sm, xdata[2]-xdata[1]),
        ) 
    
    def train(self, xdata, ydata, **kwargs):
        n0 = max([np.sum(xdata<self.t0), 1])
        Scref = dc.signal_sr(self.R10c, 1, self.TR, self.FA, self.TC, self.Tsat)
        Smref = dc.signal_sr(self.R10m, 1, self.TR, self.FA, self.TC, self.Tsat)
        self.S0c = np.mean(ydata[0][:n0]) / Scref
        self.S0m = np.mean(ydata[1][:n0]) / Smref
        return super().train(xdata, ydata, **kwargs)
    
    def pars(self):
        pars = {}
        pars['Fp']=['Plasma flow',self.Fp,'mL/sec/mL']
        pars['Eg']=['Glomerular extraction fraction',self.Eg,'']
        pars['fc']=['Cortical flow fraction',self.fc,'']
        pars['Tg']=['Glomerular mean transit time',self.Tg,'sec']
        pars['Tv']=['Peritubular & venous mean transit time',self.Tv,'sec']
        pars['Tpt']=['Proximal tubuli mean transit time',self.Tpt,'sec'] 
        pars['Tlh']=['Lis of Henle mean transit time',self.Tlh,'sec'] 
        pars['Tdt']=['Distal tubuli mean transit time',self.Tdt,'sec'] 
        pars['Tcd']=['Collecting duct mean transit time',self.Tcd,'sec']
        pars['FF']=['Filtration fraction', self.Eg/(1-self.Eg), '']
        pars['Ft']=['Tubular flow', self.Fp*self.Eg/(1-self.Eg), 'mL/sec/mL']
        pars['CBF']=['Cortical blood flow', self.Fp/(1-self.Hct), 'mL/sec/mL']
        pars['MBF']=['Medullary blood flow', (1-self.fc)*(1-self.Eg)*self.Fp/(1-self.Hct), 'mL/sec/mL']
        if self.vol is None:   
            return self.add_sdev(pars)
        pars['SKGFR']=['Single-kidney glomerular filtration rate', self.vol*self.Fp*self.Eg/(1-self.Eg),'mL/sec']
        pars['SKBF']=['Single-kidney blood flow', self.vol*self.Fp/(1-self.Hct), 'mL/sec']
        pars['SKMBF']=['Single-kidney medullary blood flow', self.vol*(1-self.fc)*(1-self.Eg)*self.Fp/(1-self.Hct), 'mL/sec']

        return self.add_sdev(pars)
    
