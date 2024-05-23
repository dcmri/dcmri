import numpy as np
import dcmri as dc


class LiverPCC(dc.Model):
    """Single-inlet model for intracellular indicator measured with a steady-state sequence.

    The free model parameters are:

    - **Te**: mean transit time of the extracellular space.
    - **De**: Transit time dispersion of the extracellular space, in the range [0,1].
    - **ve**: volume faction of the extracellular space.
    - **khe**: rate constant for indicator transport from extracellular space to hepatocytes, in mL/sec/mL. 
    - **Th**: mean transit time of the hepatocytes (sec).

    Args:
        cb (array-like): Arterial blood concentrations measured separately.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `LiverSignalPCCNS`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=100, R10=1/dc.T1(3.0,'liver'))
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.LiverPCC(gt['cb'],
        ...     dt = gt['t'][1],
        ...     Hct = 0.45, 
        ...     agent = 'gadoxetate',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'liver'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(gt['t']/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 
    dt = 0.5                #: Sampling interval of the AIF in sec. 
    Hct = 0.45              #: Hematocrit. 
    agent = 'gadoxetate'    #: Contrast agent generic name.
    field_strength = 3.0    #: Magnetic field strength in T. 
    TR = 0.005              #: Repetition time, or time between excitation pulses, in sec. 
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1                 #: Precontrast tissue relaxation rate in 1/sec. 
    S0 = 1                  #: Signal scaling factor (a.u.).
    t0 = 0                  #: Baseline length (sec).
    vol = None              #: Liver volume in mL - optional constant only used to determine dependent parameters.

    def __init__(self, cb, pars='default', **attr):
        super().__init__(pars, **attr)
        self._rp = dc.relaxivity(self.field_strength, 'blood', self.agent)
        self._rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        self._ca = cb/(1-self.Hct)
        self._t = self.dt*np.arange(np.size(cb))

    def predict(self, xdata, return_conc=False, return_rel=False): 
        Te, De, ve, khe, Th = self.pars
        C = dc.liver_conc_pcc(self._ca, Te, De, ve, khe, Th, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        R1 = self.R10 + self._rp*C[0,:] + self._rh*C[1,:]
        if return_rel:
            return dc.sample(xdata, self._t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata<self.t0), 1])
        self.S0 = np.mean(ydata[:n0]) / Sref
        super().train(xdata, ydata, **kwargs)
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([30.0, 0.85, 0.3, 20/6000, 30*60])
        else:
            return np.zeros(5)

    def bounds(self, settings='default'):
        if settings == 'default':
            ub = [60, 1, 0.6, np.inf, 10*60*60]
            lb = [0.1, 0, 0.01, 0, 10*60]
        else:
            ub = [+np.inf, 1, 1, np.inf, np.inf] 
            lb = 0
        return (lb, ub)
    
    def pfree(self, units='standard'):
        pars = [
            # Inlets
            ['Te', "Extracellular transit time", self.pars[0], 'sec'], 
            ['De', "Extracellular dispersion", self.pars[1], ''],
            # Liver tissue
            ['ve', "Liver extracellular volume fraction", self.pars[2], 'mL/mL'],
            ['khe', "Hepatocellular uptake rate", self.pars[3], 'mL/sec/mL'],
            ['Th', "Hepatocellular transit time", self.pars[4], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['kbh', "Biliary excretion rate", (1-self.pars[2])/self.pars[4], 'mL/sec/mL'],
            ['Khe', "Hepatocellular tissue uptake rate", self.pars[3]/self.pars[2], 'mL/sec/mL'],
            ['Kbh', "Biliary tissue excretion rate", np.divide(1, self.pars[4]), 'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']

        if self.vol is None:
            return pars
        
        pars += [
            ['CL', 'Liver blood clearance', self.pars[3]*self.vol, 'mL/sec'],
        ]
        if units == 'custom':
            pars[3][2:] = [pars[3][2]*60, 'mL/min']

        return pars

    def aif_conc(self):
        """Reconstructed plasma concentrations in the arterial input.

        Returns:
            np.ndarray: Concentrations in M.
        """
        return self._ca
       

class LiverPCCNS(dc.Model):
    """Steady-state acquistion over two scans, with a non-stationary two-compartment model..

    The free model parameters are:

    - **Te**: mean transit time of the extracellular space.
    - **De**: Transit time dispersion of the extracellular space, in the range [0,1].
    - **ve**: volume faction of the extracellular space.
    - **khe_i**: Initial rate constant for indicator transport from extracellular space to hepatocytes, in mL/sec/mL. 
    - **khe_f**: Final rate constant for indicator transport from extracellular space to hepatocytes, in mL/sec/mL.
    - **Th_i**: Initial mean transit time of the hepatocytes (sec).
    - **Th_f**: Final mean transit time of the hepatocytes (sec).

    Args:
        cb (array-like): Arterial blood concentrations measured separately.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `LiverSignalPCC`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_2ss` to generate synthetic test data over 2 scans:

        >>> time, _, roi, gt = dc.make_tissue_2cm_2ss(CNR=100, R10=1/dc.T1(3.0,'liver'))
        >>> nt = int(time.size/2)
        >>> time1, time2 = time[:nt], time[nt:]
        >>> roi1, roi2 = roi[:nt], roi[nt:]
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.LiverPCCNS(gt['cb'],
        ...     dt = gt['t'][1],
        ...     Hct = 0.45, 
        ...     agent = 'gadoxetate',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'liver'),
        ...     R11 = 1/dc.T1(3.0,'liver'),
        ...     t0 = 15,
        ...     t1 = time2[0],
        ... )

        Train the model on the ROI data, fixing the baseline scaling factor, and predict the data:

        >>> model.train(time1, roi1, time2, roi2, pfix=7*[0]+[1,0])
        >>> sig = model.predict(time)
        >>> conc = model.predict(time, return_conc=True)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> #
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(time/60, roi, marker='o', linestyle='None', color='cornflowerblue', label='Data')
        >>> ax0.plot(time/60, sig, marker='x', linestyle='None', color='darkblue', label='Prediction')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()
        >>> #
        >>> ax1.set_title('Reconstruction of concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
        >>> ax1.plot(time/60, 1000*conc, linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(gt['t']/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 
    dt = 0.5                #: Sampling interval of the AIF in sec. 
    Hct = 0.45              #: Hematocrit. 
    agent = 'gadoxetate'    #: Contrast agent generic name.
    field_strength = 3.0    #: Magnetic field strength in T. 
    TR = 0.005              #: Repetition time, or time between excitation pulses, in sec. 
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1                 #: Precontrast tissue relaxation rate in 1/sec. 
    R11 = 1                 #: Second scan precontrast tissue relaxation rate in 1/sec.
    t0 = 0                  #: Baseline length (sec)
    t1 = 1                  #: Start of second scan (sec)
    vol = None              #: Liver volume in mL - optional constant only used to determine dependent parameters.

    def __init__(self, cb, pars='default', **attr):
        super().__init__(pars, **attr)
        self._rp = dc.relaxivity(self.field_strength, 'blood', self.agent)
        self._rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        self._ca = cb/(1-self.Hct)
        self._t = self.dt*np.arange(np.size(cb))
    
    def predict(self, xdata, return_conc=False, return_rel=False):
        # NOTE: Changed the parametrization here for consistency
        # from Kbh = [1/Th_i, 1/Th_f]
        # to Th = [Th_i, Th_f]
        # This creates a non-linear change in Kbh(t) over the time interval of the scan. THIS LIKELY HAS SOME EFFECT ON THE RESULTS IN TRISTAN EXP MED - CHECK!
        Te, De, ve, k_he_i, k_he_f, Th_i, Th_f, S01, S02 = self.pars
        khe = [k_he_i, k_he_f]
        Th = [Th_i, Th_f]
        C = dc.liver_conc_pcc_ns(self._ca, Te, De, ve, khe, Th, t=self._t, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        # Return R
        R1 = self.R10 + self._rp*C[0,:] + self._rh*C[1,:]
        if return_rel:
            return dc.sample(xdata, self._t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, S01, self.TR, self.FA)
        t2 = (self._t >= self.t1 - (xdata[1]-xdata[0]))
        signal[t2] = dc.signal_ss(R1[t2], S02, self.TR, self.FA)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata1, ydata1, xdata2, ydata2, **kwargs):
        n0 = max([np.sum(xdata1<self.t0), 1])

        # Estimate S01 from data
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.pars[-2] = np.mean(ydata1[:n0]) / Sref

        # Estimate S02 from data
        Sref = dc.signal_ss(self.R11, 1, self.TR, self.FA)
        self.pars[-1] = np.mean(ydata2[:n0]) / Sref

        xdata = np.concatenate((xdata1, xdata2))
        ydata = np.concatenate((ydata1, ydata2))
        super().train(xdata, ydata, **kwargs)

    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([30, 0.85, 0.3, 20/6000, 20/6000, 30*60, 30*60, 1, 1])
        else:
            return np.zeros(9)

    def bounds(self, settings='default'):
        if settings == 'default':
            ub = [60, 1, 0.6, np.inf, np.inf, 10*60*60, 10*60*60, np.inf, np.inf]
            lb = [0, 0, 0.01, 0, 0, 10*60, 10*60, 0, 0]
        else: 
            ub = [np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            # Inlets
            ['Te', "Extracellular transit time", self.pars[0], 'sec'],
            ['De', "Extracellular dispersion", self.pars[1], ''],  
            # Liver tissue
            ['ve', "Liver extracellular volume fraction", self.pars[2], 'mL/mL'],
            ['k_he_i', "Hepatocellular uptake rate (initial)", self.pars[3], 'mL/sec/mL'],
            ['k_he_f', "Hepatocellular uptake rate (final)", self.pars[4], 'mL/sec/mL'],
            ['Th_i', "Hepatocellular transit time (initial)", self.pars[5], 'sec'],
            ['Th_f', "Hepatocellular transit time (final)", self.pars[6], 'sec'],
            ['S01', "Signal amplitude S01", self.pars[7], "a.u."],
            ['S02', "Signal amplitude S02", self.pars[8], "a.u."],
        ]
        if units=='custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]*6000, 'mL/min/100mL']
            pars[5][2:] = [pars[5][2]/60, 'min']
            pars[6][2:] = [pars[6][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        k_he = [self.pars[3], self.pars[4]]
        Kbh = [1/self.pars[5], 1/self.pars[6]]
        k_he_avr = np.mean(k_he)
        Kbh_avr = np.mean(Kbh)
        k_he_var = (np.amax(k_he)-np.amin(k_he))/k_he_avr
        Kbh_var = (np.amax(Kbh)-np.amin(Kbh))/Kbh_avr 
        k_bh = np.mean((1-self.pars[2])*Kbh_avr)
        Th = np.mean(1/Kbh_avr)
        pars = [
            ['k_he', "Hepatocellular uptake rate", k_he_avr, 'mL/sec/mL'],
            ['k_he_var', "Hepatocellular uptake rate variance", k_he_var, ''],
            ['Kbh', "Biliary tissue excretion rate", Kbh_avr, 'mL/sec/mL'],
            ['Kbh_var', "Biliary tissue excretion rate variance", Kbh_var, ''],
            ['k_bh', "Biliary excretion rate", k_bh, 'mL/sec/mL'],
            ['k_bh_i', "Biliary excretion rate (initial)", (1-self.pars[2])/self.pars[5], 'mL/sec/mL'],
            ['k_bh_f', "Biliary excretion rate (final)", (1-self.pars[2])/self.pars[6], 'mL/sec/mL'],
            ['Khe', "Hepatocellular tissue uptake rate", k_he_avr/self.pars[2], 'mL/sec/mL'],
            ['Th', "Hepatocellular transit time", Th, 'sec'],
        ]
        if units=='custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*100, '%']
            pars[4][2:] = [pars[4][2]*6000, 'mL/min/100mL']
            pars[5][2:] = [pars[5][2]*6000, 'mL/min/100mL']
            pars[6][2:] = [pars[6][2]*6000, 'mL/min/100mL']
            pars[7][2:] = [pars[7][2]*6000, 'mL/min/100mL']
            pars[8][2:] = [pars[8][2]/60, 'min']
            
        if self.vol is None:
            return pars
        
        pars += [
            ['CL', 'Liver blood clearance', k_he_avr*self.vol, 'mL/sec'],
        ]
        if units=='custom': 
            pars[9][2:] = [pars[9][2]*60, 'mL/min']

        return pars
    
    def aif_conc(self):
        """Reconstructed plasma concentrations in the arterial input.

        Returns:
            np.ndarray: Concentrations in M.
        """
        return self._ca