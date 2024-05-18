import numpy as np
import dcmri as dc

class TissueSignal3(dc.Model):
    """Extended Tofts tissue in fast water exchange, acquired with a spoiled gradient echo sequence in steady state.
    """
    cb = None               #: Uniformly sampled blood concentrations at the inlet in M. 
    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec. 
    Hct = 0.45              #: Hematocrit. 
    agent = 'gadoterate'    #: Contrast agent generic name.
    field_strength = 3.0    #: Magnetic field strength in T. 
    TR = 5.0/1000.0         #: Repetition time, or time between excitation pulses, in sec. 
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1                 #: Precontrast relaxation rate in 1/sec. 
    S0 = 1                  #: Signal scaling factor in arbitrary units. 

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        t = self.dt*np.arange(len(self.cb))
        if np.amax(t) <= np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        vp, Ktrans, ve = self.pars
        ca = self.cb/(1-self.Hct)
        C = dc.conc_etofts(ca, vp, Ktrans, ve, dt=self.dt)
        if return_conc:
            return dc.sample(t, C, xdata, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*C
        ydata = dc.signal_spgress(R1, self.S0, self.TR, self.FA)
        return dc.sample(t, ydata, xdata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([0.1, 0.1/60, 0.2])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings='default'):
        if settings=='default':
            ub = [1, np.inf, 1]
            lb = [0, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
    
    def train(self, xdata, ydata, baseline=1, **kwargs):
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:baseline]) / Sref
        return super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        # vp, Ktrans, ve
        pars = [
            ['vp','Plasma volume',self.pars[0],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[1],'1/sec'],
            ['ve','Extravascular extracellular volume',self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*100, 'mL/100mL']
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Te','Extracellular mean transit time',self.pars[2]/self.pars[1],'sec'],
            ['kep','Extravascular transfer constant',self.pars[1]/self.pars[2],'1/sec'],
            ['v','Extracellular volume',self.pars[0]+self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]/60, 'min']
            pars[1][2:] = [pars[1][2]*60, '1/min']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
        return pars


class TissueSignal3b(TissueSignal3):
    """Extended Tofts tissue without water exchange, acquired with a spoiled gradient echo sequence in steady state.
    """

    def predict(self, xdata:np.ndarray)->np.ndarray:
        t = self.dt*np.arange(len(self.cb))
        if np.amax(t) <= np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        vp, Ktrans, ve = self.pars
        vb = vp/(1-self.Hct)
        ca = self.cb/(1-self.Hct)
        C = dc.conc_etofts(ca, vp, Ktrans, ve, dt=self.dt, sum=False)
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1b = self.R10 + rp*C[0,:]/vb
        R1e = self.R10 + rp*C[1,:]/ve
        R1c = self.R10 + np.zeros(C.shape[1])
        v = [vb, ve, 1-vb-ve]
        R1 = np.stack((R1b, R1e, R1c))
        ydata = dc.signal_spgress_nex(v, R1, self.S0, self.TR, self.FA)
        return dc.sample(t, ydata, xdata, xdata[2]-xdata[1])
    

class TissueSignal3c(dc.Model):
    """Extended Tofts tissue in fast water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    Probably the most common modelling approach for generic tiossues. The arterial concentrations are calculated by direct analytical inversion of the arterial signal 

    The 3 free model parameters are:

    - *Plasma volume*: Volume fraction of the plasma compartment. 
    - *Vascular transfer constant Ktrans* (mL/sec/mL): clearance of the plasma compartment per unit tissue.
    - *Extravascular, extracellular volume*: Blood flow through the loop.Volume fraction of the interstitial compartment.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `TissueSignal3`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `synth_1` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.synth_1(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.TissueSignal3c(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1.0,
        ...     R10a = 1/dc.T1(3.0,'blood'),
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
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=2, units='custom')
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Plasma volume (vp): 3.84 (0.0) mL/100mL
        Volume transfer constant (Ktrans): 16.64 (0.0) mL/min/100mL
        Extravascular extracellular volume (ve): 24.08 (0.01) mL/100mL
        ------------------
        Derived parameters
        ------------------
        Extracellular mean transit time: 1.45 min
        Extravascular transfer constant (kep): 0.69 1/min
        Extracellular volume (vp+ve): 27.91 mL/100mL
    """         

    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec. 
    Hct = 0.45              #: Hematocrit. 
    agent = 'gadoterate'    #: Contrast agent generic name.
    field_strength = 3.0    #: Magnetic field strength in T. 
    TR = 5.0/1000.0         #: Repetition time, or time between excitation pulses, in sec. 
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1                 #: Precontrast tissue relaxation rate in 1/sec. 
    R10a = 1                #: Precontrast arterial relaxation rate in 1/sec. 
    S0 = 1                  #: Signal scaling factor in arbitrary units. 
    t0 = 1                  #: Baseline length (sec).

    def __init__(self, aif, pars='default', **attr):
        super().__init__(pars, **attr)

        # Calculate constants
        self.n0 = max([round(self.t0/self.dt),1])
        self._r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        cb = dc.conc_spgress(aif, 
            TR=self.TR, FA=self.FA, T10=1/self.R10a, r1=self._r1, n0=self.n0)
        self._ca = cb/(1-self.Hct)
        self._t = self.dt*np.arange(np.size(aif))

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        C = dc.conc_etofts(self._ca, *self.pars, dt=self.dt)
        if return_conc:
            return dc.sample(self._t, C, xdata, xdata[2]-xdata[1])
        R1 = self.R10 + self._r1*C
        ydata = dc.signal_spgress(R1, self.S0, self.TR, self.FA)
        return dc.sample(self._t, ydata, xdata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([0.1, 0.1/60, 0.2])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings='default'):
        ub = [1, np.inf, 1]
        lb = [0, 0, 0]
        return (lb, ub)
    
    def train(self, xdata, ydata, **kwargs):
        
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:self.n0]) / Sref
        return super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        # vp, Ktrans, ve
        pars = [
            ['vp','Plasma volume',self.pars[0],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[1],'1/sec'],
            ['ve','Extravascular extracellular volume',self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*100, 'mL/100mL']
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Te','Extracellular mean transit time',self.pars[2]/self.pars[1],'sec'],
            ['kep','Extravascular transfer constant',self.pars[1]/self.pars[2],'1/sec'],
            ['v','Extracellular volume',self.pars[0]+self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]/60, 'min']
            pars[1][2:] = [pars[1][2]*60, '1/min']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
        return pars
    
    def aif_conc(self):
        """Reconstructed concentrations in the arterial input.

        Returns:
            np.ndarray: Concentrations in M.
        """
        return self._ca


class TissueSignal5(dc.Model):
    """Extended Tofts tissue with intermediate water exchange, acquired with a spoiled gradient echo sequence in steady state.
    """

    pars = np.zeros(5)

    cb = None
    dt = 0.5                    # Internal time resolution (sec)
    Hct = 0.45
    R10 = 1
    field_strength = 3.0        # Field strength (T)
    agent = 'gadoterate'    
    S0 = 1
    TR = 4.0/1000.0            # Repetition time (sec)
    FA = 15.0                   # Nominal flip angle (degrees)

    def predict(self, xdata:np.ndarray)->np.ndarray:
        t = self.dt*np.arange(len(self.cb))
        if np.amax(t) <= np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        vp, Ktrans, ve, PSbe, PSec = self.pars
        vb = vp/(1-self.Hct)
        ca = self.cb/(1-self.Hct)
        C = dc.conc_etofts(ca, vp, Ktrans, ve, dt=self.dt, sum=False)
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1b = self.R10 + rp*C[0,:]/vb
        R1e = self.R10 + rp*C[1,:]/ve
        R1c = self.R10 + np.zeros(C.shape[1])
        PS = np.array([[0,PSbe,0],[PSbe,0,PSec],[0,PSec,0]])
        v = [vb, ve, 1-vb-ve]
        R1 = np.stack((R1b, R1e, R1c))
        ydata = dc.signal_spgress_wex(PS, v, R1, self.S0, self.TR, self.FA)
        return dc.sample(t, ydata, xdata, xdata[2]-xdata[1])

    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([0.1, 0.1/60, 0.2, 10, 10])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings='default'):
        if settings == 'default':
            ub = [1, np.inf, 1, np.inf, np.inf]
            lb = [0, 0, 0, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
    
    def train(self, xdata, ydata, baseline=1, **kwargs):
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:baseline]) / Sref
        return super().train(xdata, ydata, **kwargs)