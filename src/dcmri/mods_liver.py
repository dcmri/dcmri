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
        cb (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Sampling interval of the AIF in sec. 
        Hct (float, optional): Hematocrit. 
        agent (str, optional): Contrast agent generic name.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec. 
        S0 (float, optional): Signal scaling factor.
        t0 (float, optional): Baseline length (sec).
        vol (float, optional): Liver volume in mL.

    See Also:
        `LiverPCCNS`

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
        >>> ax1.plot(gt['t']/60, 1000*model.conc(), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(gt['t']/60, 1000*model.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 
    def __init__(self, cb, 
            pars = None,
            dt = 0.5,                
            Hct = 0.45,             
            agent = 'gadoxetate',    
            field_strength = 3.0,    
            TR = 0.005,  
            FA = 15.0,  
            R10 = 1,                 
            S0 = 1,               
            t0 = 1, 
            vol = None,
        ):
        self.pars = self.pars0(pars)
        self.ca = cb/(1-Hct)                #: Arterial plasma concentration (M)
        self._rp = dc.relaxivity(field_strength, 'blood', agent)
        self._rh = dc.relaxivity(field_strength, 'hepatocytes', agent)
        self._t = dt*np.arange(np.size(cb))
        self._dt = dt
        self._Hct = Hct
        self._TR = TR
        self._FA = FA
        self._R10 = R10
        self.S0 = S0
        self._t0 = t0
        self._vol = vol

    def conc(self, sum=True):
        """Tissue concentrations

        Args:
            sum (bool, optional): If True, returns the total concentrations. If False, returns concentration in both compartments separately. Defaults to True.

        Returns:
            numpy.ndarray: Concentration in M
        """
        return dc.liver_conc_pcc(self.ca, *self.pars, dt=self._dt, sum=sum)

    def _forward_model(self, xdata): 
        C = self.conc(sum=False)
        R1 = self._R10 + self._rp*C[0,:] + self._rh*C[1,:]
        signal = dc.signal_ss(R1, self.S0, self._TR, self._FA)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_ss(self._R10, 1, self._TR, self._FA)
        n0 = max([np.sum(xdata<self._t0), 1])
        self.S0 = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)
    
    def pars0(self, settings=None):
        if settings == None:
            return np.array([30.0, 0.85, 0.3, 20/6000, 30*60])
        else:
            return np.zeros(5)

    def bounds(self, settings=None):
        if settings == None:
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

        if self._vol is None:
            return pars
        
        pars += [
            ['CL', 'Liver blood clearance', self.pars[3]*self._vol, 'mL/sec'],
        ]
        if units == 'custom':
            pars[3][2:] = [pars[3][2]*60, 'mL/min']

        return pars
       

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
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Sampling interval of the AIF in sec. 
        Hct (float, optional): Hematocrit. 
        agent (str, optional): Contrast agent generic name.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec - first scan. 
        R11 (float, optional): Precontrast tissue relaxation rate in 1/sec - second scan. 
        t0 (float, optional): Baseline length (sec).
        t1 (float, optional): Start of the sccond scan (sec).
        vol (float, optional): Liver volume in mL.

    See Also:
        `LiverPCC`

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
        >>> ax1.plot(gt['t']/60, 1000*model.conc(), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(gt['t']/60, 1000*model.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()
    """ 

    def __init__(self, cb, 
            pars = None,
            dt = 0.5,                
            Hct = 0.45,             
            agent = 'gadoxetate',    
            field_strength = 3.0,    
            TR = 0.005,  
            FA = 15.0,  
            R10 = 1,  
            R11 = 1,                
            S0 = 1,               
            t0 = 1, 
            t1 = 1,
            vol = None,
        ):
        self.pars = self.pars0(pars)
        self.ca = cb/(1-Hct)                #: Arterial plasma concentration (M)
        self._rp = dc.relaxivity(field_strength, 'blood', agent)
        self._rh = dc.relaxivity(field_strength, 'hepatocytes', agent)
        self._t = dt*np.arange(np.size(cb))
        self._dt = dt
        self._Hct = Hct
        self._TR = TR
        self._FA = FA
        self._R10 = R10
        self._R11 = R11
        self.S0 = S0
        self._t0 = t0
        self._t1 = t1
        self._vol = vol

    def conc(self, sum=True):
        Te, De, ve, k_he_i, k_he_f, Th_i, Th_f, S01, S02 = self.pars
        khe = [k_he_i, k_he_f]
        Th = [Th_i, Th_f]
        return dc.liver_conc_pcc_ns(self.ca, Te, De, ve, khe, Th, t=self._t, dt=self._dt, sum=sum)   
    
    def _forward_model(self, xdata):
        # NOTE: Changed the parametrization here for consistency
        # from Kbh = [1/Th_i, 1/Th_f]
        # to Th = [Th_i, Th_f]
        # This creates a non-linear change in Kbh(t) over the time interval of the scan. THIS LIKELY HAS SOME EFFECT ON THE RESULTS IN TRISTAN EXP MED - CHECK!
        Te, De, ve, k_he_i, k_he_f, Th_i, Th_f, S01, S02 = self.pars
        C = self.conc(sum=False)
        R1 = self._R10 + self._rp*C[0,:] + self._rh*C[1,:]
        signal = dc.signal_ss(R1, S01, self._TR, self._FA)
        t2 = (self._t >= self._t1 - (xdata[1]-xdata[0]))
        signal[t2] = dc.signal_ss(R1[t2], S02, self._TR, self._FA)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata1, ydata1, xdata2, ydata2, **kwargs):
        n0 = max([np.sum(xdata1<self._t0), 1])

        # Estimate S01 from data
        Sref = dc.signal_ss(self._R10, 1, self._TR, self._FA)
        self.pars[-2] = np.mean(ydata1[:n0]) / Sref

        # Estimate S02 from data
        Sref = dc.signal_ss(self._R11, 1, self._TR, self._FA)
        self.pars[-1] = np.mean(ydata2[:n0]) / Sref

        xdata = np.concatenate((xdata1, xdata2))
        ydata = np.concatenate((ydata1, ydata2))
        super().train(xdata, ydata, **kwargs)

    def pars0(self, settings=None):
        if settings == None:
            return np.array([30, 0.85, 0.3, 20/6000, 20/6000, 30*60, 30*60, 1, 1])
        else:
            return np.zeros(9)

    def bounds(self, settings=None):
        if settings == None:
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
            
        if self._vol is None:
            return pars
        
        pars += [
            ['CL', 'Liver blood clearance', k_he_avr*self._vol, 'mL/sec'],
        ]
        if units=='custom': 
            pars[9][2:] = [pars[9][2]*60, 'mL/min']

        return pars
