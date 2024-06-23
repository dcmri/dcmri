import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc


class Tissue(dc.Model):
    """Model for general vascular-interstitial tissues with bi-directional exchange of indicator.

        **Input function**

        - **aif** (array-like, default=None): Signal-time curve in a feeding artery. If AIF is set to None, then the parameter ca must be provided (arterial concentrations).
        - **ca** (array-like, default=None): Concentration (M) in the arterial input. Must be provided when aif = None, ignored otherwise. 

        **Acquisition parameters**

        - **t** (array-like, default=None): Time points (sec) of the aif. If t is not provided, the temporal sampling is uniform with interval dt.
        - **dt** (float, default=1.0): Sampling interval of the AIF in sec.
        - **field_strength** (float, default=3.0): Magnetic field strength in T.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **n0** (int, default=1): Baseline length in nr of acquisitions.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec.
        - **FA** (float, default=15): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence.
        - **TP** (float, default=0): Preparation delay in a saturation-recovery sequence.

        **Tracer-kinetic parameters**

        - **kinetics** (str, default='HF'): Tracer-kinetic model.
        - **Hct** (float, default=0.45): Hematocrit.
        - **Fp** (float, default=0.01): Plasma flow, or flow of plasma into the plasma compartment (mL/sec/mL).
        - **PS** (float, default=0.003): Permeability-surface area product: volume of plasma cleared of indicator per unit time and per unit tissue (mL/sec/mL).
        - **vp** (float, default=0.1): Plasma volume, or volume fraction of the plasma compartment (mL/mL). 
        - **ve** (float, default=0.5): Extravascular, extracellular volume: volume fraction of the interstitial compartment (mL/mL).

        **Water-kinetic parameters**

        - **water_exchange** (str, default='fast'): Water exchange regime ('fast', 'none' or 'any').
        - **PSe** (float, default=10): Transendothelial water permeability-surface area product: PS for water across the endothelium (mL/sec/mL).
        - **PSc** (float, default=10): Transcytolemmal water permeability-surface area product: PS for water across the cell wall (mL/sec/mL).

        **Signal parameters**

        - **sequence** (str, default='SS'): imaging sequence.
        - **R10** (float, default=1): Precontrast tissue relaxation rate in 1/sec.
        - **R10b** (float, default=1): Precontrast arterial relaxation rate in 1/sec. 
        - **S0** (float, default=1): Scale factor for the MR signal (a.u.).

        **Prediction and training parameters**

        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **bounds** (array-like): 2-element list with lower and upper bounds of the free parameters. The default depends on the kinetics parameter.

    Args:
        params (dict, optional): override defaults for any of the parameters.

    See Also:
        `Liver`, `Kidney`

    Example:

        Fit extended Tofts model to data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.Tissue(
        ...     aif = aif,
        ...     dt = time[1],
        ...     agent = 'gadodiamide',
        ...     TR = 0.005,
        ...     FA = 20,
        ...     n0 = 15,
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> model.plot(time, roi, testdata=gt)
    """ 

    def __init__(self, **params):

        # Input function
        self.aif = None
        self.ca = None
        
        # Acquisition parameters
        self.sequence = 'SS'
        self.field_strength = 3.0
        self.t = None
        self.dt = 1.0
        self.agent = 'gadoterate'
        self.n0 = 1
        self.TR = 0.005
        self.FA = 15
        self.TC = 0.1
        self.TP = 0
        
        # Tracer-kinetic parameters
        self.kinetics = 'HF'
        self.Hct = 0.45
        self.Fp = 0.01
        self.PS = 0.003
        self.vp = 0.1
        self.ve = 0.5

        # Water-kinetic parameters (TODO: what are typical values?)
        self.water_exchange = 'fast'
        self.PSe = 10
        self.PSc = 10

        # Signal parameters
        self.R10 = 1/dc.T1(3.0, 'muscle')
        self.R10b = 1/dc.T1(3.0, 'blood')
        self.S0 = 1

        # training parameters
        self.free = ['PS','vp','ve']
        self.bounds = [0, [np.inf, 1, 1]]

        # Preset parameters
        if 'kinetics' in params:
            if params['kinetics'] == '2CXM':
                self.free = ['Fp','PS','vp','ve']
                self.bounds = [0, [np.inf, np.inf, 1, 1]]
            elif params['kinetics'] == '2CUM':
                self.free = ['Fp','PS','vp']
                self.bounds = [0, [np.inf, np.inf, 1]]
            elif params['kinetics'] == 'HF':
                self.free = ['PS','vp','ve']
                self.bounds = [0, [np.inf, 1, 1]]
            elif params['kinetics'] == 'HFU':
                self.free = ['Fp','vp']
                self.bounds = [0, [np.inf, 1]]
            elif params['kinetics'] == 'WV': 
                self.free = ['Ktrans','ve']
                self.bounds = [0, [np.inf, 1]]
            elif params['kinetics'] == 'FX':
                self.free = ['Fp','v']
                self.bounds = [0, [np.inf, 1]] 
            elif params['kinetics'] == 'NX': 
                self.free = ['Fp','vp']
                self.bounds = [0, [np.inf, 1]]
            elif params['kinetics'] == 'U': 
                self.free = ['Fp']
                self.bounds = [0, np.inf]

        if 'water_exchange' in params:        
            if params['water_exchange'] == 'any':
                self.free += ['PSe', 'PSc']
                self.bounds[1] += [np.inf, np.inf]

        # Override defaults
        for k, v in params.items():
            setattr(self, k, v)

        # Check inputs
        if (self.aif is None) and (self.ca is None):
            raise ValueError('Please provide either an arterial sigal (aif) or an arterial concentration (ca).')  
        
        # Dependent parameters
        self.Ktrans = self.PS*self.Fp/(self.PS+self.Fp)
        self.v = self.vp+self.ve


    def conc(self, sum=True):
        """Tissue concentration

        Args:
            sum (bool, optional): If True, returns the total concentrations. Else returns the concentration in the individual compartments. Defaults to True.

        Returns:
            np.ndarray: Concentration in M
        """

        # Arterial concentrations
        if self.aif is not None:
            r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
            if self.sequence == 'SR':
                cb = dc.conc_src(self.aif, self.TC, 1/self.R10b, r1, self.n0)
            elif self.sequence == 'SS':
                cb = dc.conc_ss(self.aif, self.TR, self.FA, 1/self.R10b, r1, self.n0)
            else:
                raise NotImplementedError('Signal model ' + self.sequence + 'is not (yet) supported.') 
            self.ca = cb/(1-self.Hct) 

        # Tissue concentrations
        if self.kinetics == '2CXM': # (Fp, PS, vp, ve)
            self.Ktrans = self.Fp*self.PS/(self.Fp+self.PS)
            self.v = self.vp + self.ve
            C = dc.conc_tissue(self.ca, self.Fp, self.vp, self.PS, self.ve, t=self.t, dt=self.dt, sum=sum, kinetics='2CX')
            #C = dc._conc_2cx(self.ca, self.Fp, self.vp, self.PS, self.ve, self.t, dt=self.dt, sum=sum)
        elif self.kinetics == '2CUM': # (Fp, PS, vp)
            self.Ktrans = self.Fp*self.PS/(self.Fp+self.PS)
            C = dc.conc_tissue(self.ca, self.Fp, self.vp, self.PS, t=self.t, dt=self.dt, sum=sum, kinetics='2CU')
            #C = dc._conc_2cu(self.ca, self.Fp, self.vp, self.PS, self.t, dt=self.dt, sum=sum)
        elif self.kinetics == 'HF': # (PS, vp, ve)
            self.Fp = np.inf
            self.Ktrans = self.PS
            self.v = self.ve + self.vp
            C = dc.conc_tissue(self.ca, self.vp, self.PS, self.ve, t=self.t, dt=self.dt, sum=sum, kinetics='HF')
            #C = dc.conc_hf(self.ca, self.vp, self.PS, self.ve, self.t, dt=self.dt, sum=sum)
        elif self.kinetics == 'HFU': # (Fp, PS, vp)
            self.Fp = np.inf
            self.Ktrans = self.PS
            C = dc.conc_tissue(self.ca, self.vp, self.PS, t=self.t, dt=self.dt, sum=sum, kinetics='HFU')
            # C = dc._conc_hfu(self.ca, self.vp, self.PS, self.t, dt=self.dt, sum=sum)  
        elif self.kinetics == 'WV': # (Ktrans, ve)
            self.v = self.ve
            self.vp = 0    
            C = dc.conc_tissue(self.ca, self.Ktrans, self.ve, t=self.t, dt=self.dt, sum=sum, kinetics='FX')        
            #C = dc._conc_1c(self.ca, self.Ktrans, self.ve, self.t, dt=self.dt)
        elif self.kinetics == 'FX': # (Fp, v)
            self.PS = np.inf
            self.Ktrans = self.Fp
            C = dc.conc_tissue(self.ca, self.Fp, self.v, t=self.t, dt=self.dt, kinetics='FX')
            #C = dc._conc_1c(self.ca, self.Fp, self.v, self.t, dt=self.dt)
        elif self.kinetics == 'NX': # (Fp, vp)
            C = dc.conc_tissue(self.ca, self.Fp, self.vp, t=self.t, dt=self.dt, kinetics='NX')
            #C = dc._conc_1c(self.ca, self.Fp, self.vp, self.t, dt=self.dt)
        elif self.kinetics == 'U': # (Fp)
            C = dc.conc_tissue(self.ca, self.Fp, t=self.t, dt=self.dt, kinetics='U')
            #C = dc._conc_u(self.ca, self.Fp, self.t, dt=self.dt)
        return C


    def relax(self):
        """Longitudinal relaxation rate R1(t).

        Returns:
            np.ndarray: Relaxation rate. Dimensions are (nt,) for a tissue in fast water exchange, or (nc,nt) for a multicompartment tissue outside the fast water exchange limit.
        """
        C = self.conc(sum=False)
        r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        if self.water_exchange == 'fast':
            R1 = self.R10 + r1*(C[0,:]+C[1,:])
        elif self.kinetics in ['2CXM', 'HF']:
            vb = self.vp/(1-self.Hct)
            R1b = self.R10b + r1*C[0,:]/vb
            R1e = self.R10 + r1*C[1,:]/self.ve
            R1c = self.R10 + np.zeros(C.shape[1])
            R1 = np.stack((R1b, R1e, R1c))
        elif self.kinetics in ['2CUM','HFU']:
            vb = self.vp/(1-self.Hct)
            R1b = self.R10b + r1*C[0,:]/vb
            R1e = self.R10 + r1*C[1,:]/(1-vb)
            R1 = np.stack((R1b, R1e))    
        elif self.kinetics in ['WV','FX']:
            R1e = self.R10 + r1*C/self.v
            R1c = self.R10 + np.zeros(C.size)
            R1 = np.stack((R1e, R1c))
        elif self.kinetics == 'NX':
            vb = self.vp/(1-self.Hct)
            R1b = self.R10b + r1*C/vb
            R1e = self.R10 + np.zeros(C.size)
            R1 = np.stack((R1b, R1e))  
        elif self.kinetics == 'U':
            R1 = self.R10 + r1*C
        return R1


    def signal(self)->np.ndarray:
        """Pseudocontinuous signal S(t) as a function of time.

        Returns:
            np.ndarray: Signal as a 1D array.
        """
        R1 = self.relax()

        if self.water_exchange == 'fast':
            if self.sequence == 'SR':
                return dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC, self.TP)
            elif self.sequence == 'SS':
                return dc.signal_ss(R1, self.S0, self.TR, self.FA)

        elif self.water_exchange == 'none':

            if self.kinetics in ['2CXM','HF']:
                vb = self.vp/(1-self.Hct)
                v = [vb, self.ve, 1-vb-self.ve]
                if self.sequence == 'SR':
                    return dc.signal_sr_nex(v, R1, self.S0, self.TR, self.FA, self.TC, self.TP)
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=0)

            elif self.kinetics in ['2CUM','HFU','NX']:
                vb = self.vp/(1-self.Hct)
                v = [vb, 1-vb]
                if self.sequence == 'SR':
                    return dc.signal_sr_nex(v, R1, self.S0, self.TR, self.FA, self.TC, self.TP)
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=0)

            elif self.kinetics in ['WV','FX']:
                v = [self.ve, 1-self.ve]
                if self.sequence == 'SR':
                    return dc.signal_sr_nex(v, R1, self.S0, self.TR, self.FA, self.TC, self.TP)
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=0)

            elif self.kinetics == 'U':
                if self.sequence == 'SR':
                    return dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC, self.TP)
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA)

        elif self.water_exchange == 'any':

            if self.kinetics in ['2CXM','HF']:
                vb = self.vp/(1-self.Hct)
                v = [vb, self.ve, 1-vb-self.ve]
                PSw = np.array([[0,self.PSe,0],[self.PSe,0,self.PSc],[0,self.PSc,0]])
                # TODO: ADD diagonal element (flow term)!!
                # Also needs adding inflow 
                # Fb = self.Fp/(1-self.Hct)
                # PSw = np.array([[Fb,self.PSe,0],[self.PSe,0,self.PSc],[0,self.PSc,0]])
                if self.sequence == 'SR':
                    return NotImplementedError('Saturation-recovery not yet implemented for intermediate water exchange')
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=PSw)

            elif self.kinetics in ['2CUM','HFU','NX']:
                vb = self.vp/(1-self.Hct)
                v = [vb, 1-vb]
                # TODO: ADD diagonal element (flow term)!!
                # Fb = self.Fp/(1-self.Hct)
                # PSw = np.array([[Fb,self.PSe],[self.PSe,0]])
                PSw = np.array([[0,self.PSe],[self.PSe,0]])
                if self.sequence == 'SR':
                    return NotImplementedError('Saturation-recovery not yet implemented for intermediate water exchange')
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=PSw)

            elif self.kinetics in ['WV','FX']:
                v = [self.ve, 1-self.ve]
                PSw = np.array([[0,self.PSc],[self.PSc,0]])
                # TODO: ADD diagonal element (flow term)!!
                # FX:
                # Fb = self.Fp/(1-self.Hct)
                # PSw = np.array([[Fb,self.PSc],[self.PSc,0]])
                # WV:
                # PSw = np.array([[self.PSe,self.PSc],[self.PSc,0]])
                if self.sequence == 'SR':
                    return NotImplementedError('Saturation-recovery not yet implemented for intermediate water exchange')
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=PSw)

            elif self.kinetics == 'U':
                if self.sequence == 'SR':
                    return dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC, self.TP)
                elif self.sequence == 'SS':
                    return dc.signal_ss(R1, self.S0, self.TR, self.FA)

    def time(self):
        """Array of time points.

        Returns:
            np.ndarray: time points in seconds.
        """
        if self.t is None:
            if self.aif is None:
                return self.dt*np.arange(np.size(self.ca))
            else:
                return self.dt*np.arange(np.size(self.aif))
        else:
            return self.t

    def predict(self, xdata:np.ndarray)->np.ndarray:
        t = self.time()
        if np.amax(xdata) > np.amax(t):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'The largest time point that can be predicted is ' + str(np.amax(t)/60) + 'min.'
            raise ValueError(msg)
        sig = self.signal()
        return dc.sample(xdata, t, sig, self.dt)
   
    def train(self, xdata, ydata, **kwargs):
        if self.sequence == 'SR':
            Sref = dc.signal_sr(self.R10, 1, self.TR, self.FA, self.TC, self.TP)
        elif self.sequence == 'SS':
            Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        else:
            raise NotImplementedError('Signal model ' + self.sequence + 'is not (yet) supported.') 
        self.S0 = np.mean(ydata[:self.n0]) / Sref
        return dc.train(self, xdata, ydata, **kwargs)
    
    def export_params(self):
        vb = self.vp/(1-self.Hct)
        pars = {
            'Fp': ['Plasma flow',self.Fp,'mL/sec/mL'],
            'PS': ['Permeability-surface area product',self.PS,'mL/sec/mL'],
            'Ktrans': ['Volume transfer constant',self.Ktrans,'mL/sec/mL'],
            've': ['Extravascular extracellular volume',self.ve,'mL/mL'],
            'vb': ['Blood volume',vb,'mL/mL'],
            'vp': ['Plasma volume',self.vp,'mL/mL'],
            'v': ['Extracellular volume',self.v,'mL/mL'],
            'Te': ['Extracellular mean transit time',self.ve/self.PS,'sec'],
            'kep': ['Extravascular transfer constant',self.Ktrans/self.ve,'1/sec'],
            'E': ['Extraction fraction',self.PS/(self.PS+self.Fp),''],
            'Tp': ['Plasma mean transit time',self.vp/(self.Fp+self.PS),'sec'],
            'Tb': ['Blood mean transit time',self.vp/self.Fp,'sec'],
            'T': ['Mean transit time',self.v/self.Fp,'sec'],
            'PSe': ['Transendothelial water PS',self.PSe,'mL/sec/mL'],
            'PSc': ['Transcytolemmal water PS',self.PSc,'mL/sec/mL'],
            'Twc': ['Intracellular water mean transit time', (1-vb-self.ve)/self.PSc, 'sec'],
            'Twi': ['Interstitial water mean transit time', self.ve/(self.PSc+self.PSe), 'sec'],
            'Twb': [ 'Intravascular water mean transit time', vb/self.PSe, 'sec'],
        }
        return self._add_sdev(pars)
    
    def plot(self, 
                xdata:tuple[np.ndarray, np.ndarray], 
                ydata:tuple[np.ndarray, np.ndarray],  
                testdata=None, xlim=None, fname=None, show=True):
        time = self.time()
        C = self.conc(sum=True)
        if xlim is None:
            xlim = [np.amin(time), np.amax(time)]
        fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(xdata/60, ydata, marker='o', linestyle='None', color='cornflowerblue', label='Data')
        ax0.plot(time/60, self.predict(time), linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()
        ax1.set_title('Reconstruction of concentrations.')
        if testdata is not None:
            ax1.plot(testdata['t']/60, 1000*testdata['C'], marker='o', linestyle='None', color='cornflowerblue', label='Tissue ground truth')
            ax1.plot(testdata['t']/60, 1000*testdata['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        ax1.plot(time/60, 1000*C, linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        ax1.plot(time/60, 1000*self.ca, linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

