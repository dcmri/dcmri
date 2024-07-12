import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc


class Liver(dc.Model):

    """Single-inlet model for intracellular indicator measured with a steady-state sequence.

        **Input function**

        - **aif** (array-like, default=None): Signal-time curve in a feeding artery. If AIF is set to None, then the parameter ca must be provided (arterial concentrations).
        - **ca** (array-like, default=None): Concentration (M) in the arterial input. Must be provided when aif = None, ignored otherwise.

        **Acquisition parameters**

        - **sequence** (str, default='SS'): imaging sequence.
        - **t** (array-like, default=None): array of time points.
        - **dt** (float, optional): Sampling interval of the AIF in sec. 
        - **agent** (str, optional): Contrast agent generic name.
        - **field_strength** (float, optional): Magnetic field strength in T. 
        - **TR** (float, optional): Repetition time, or time between excitation pulses, in sec. 
        - **FA** (float, optional): Nominal flip angle in degrees.
        - **n0** (float, optional): Baseline length in number of acquisitions.

        **Kinetic parameters**

        - **kinetics** (str, default='stationary'): Tracer-kinetic model.
        - **Hct** (float, optional): Hematocrit. 
        - **Te**: mean transit time of the extracellular space.
        - **De**: Transit time dispersion of the extracellular space, in the range [0,1].
        - **ve**: volume faction of the extracellular space.
        - **khe**: rate constant for indicator transport from extracellular space to hepatocytes, in mL/sec/mL. 
        - **Th**: mean transit time of the hepatocytes (sec).

        **Prediction and training parameters**

        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.
        - **bounds** (array-like): 2-element list with lower and upper bounds of the free parameters. The default depends on the kinetics parameter.

        **Signal parameters**

        - **R10** (float, optional): Precontrast tissue relaxation rate in 1/sec. 
        - **S0** (float, optional): Signal scaling factor.

        **Other parameters**

        - **vol** (float, optional): Liver volume in mL.

    Args:
        params (dict, optional): override defaults for any of the parameters.

    See Also:
        `Tissue`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue(CNR=100, agent='gadoxetate', R10=1/dc.T1(3.0,'liver'))
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.Liver(
        ...     aif = aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadoxetate',
        ...     TR = 0.005,
        ...     FA = 20,
        ...     n0 = 10,
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
        self.dt = 0.5                    
        self.agent = 'gadoxetate'  
        self.n0 = 1  
        self.TR = 0.005  
        self.FA = 15.0  
        self.TC = 0.180

        # Tracer-kinetic parameters
        self.kinetics = 'non-stationary'
        self.Hct = 0.45    
        self.Te = 30.0
        self.De = 0.85
        self.ve = 0.3 
        self.khe = 20/6000
        self.Th = 30*60
        self.khe_f = 20/6000
        self.Th_f = 30*60

        # Signal parameters
        self.R10 = 1/dc.T1(3.0, 'liver')  
        self.R10b = 1/dc.T1(3.0, 'blood')  
        self.S0 = 1         
        
        # training parameters
        self.free = ['Te','De','ve','khe','Th']
        self.bounds = [
            [0.1, 0, 0.01, 0, 10*60, ],
            [60, 1, 0.6, 0.1, 10*60*60],
        ]

        # Other parameters
        self.vol = None
        
        # Preset parameters
        if 'kinetics' in params:
            if params['kinetics'] == 'non-stationary':
                self.free += ['khe_f','Th_f']
                self.bounds[0] += [0, 10*60]
                self.bounds[1] += [ 0.1, 10*60*60]
        
        # Override defaults
        for k, v in params.items():
            setattr(self, k, v)

        # Check inputs
        if (self.aif is None) and (self.ca is None):
            raise ValueError('Please provide either an arterial sigal (aif) or an arterial concentration (ca).')  
        
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


    def conc(self, sum=True):
        """Tissue concentrations

        Args:
            sum (bool, optional): If True, returns the total concentrations. If False, returns concentration in both compartments separately. Defaults to True.

        Returns:
            numpy.ndarray: Concentration in M
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

        t = self.time()

        if self.kinetics == 'non-stationary':
            khe = dc.interp([self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], t)
            Kbh = dc.interp([1/self.Th, 1/self.Th_f], t)
            return t, dc.conc_liver(self.ca, 
                    self.ve, self.Te, self.De, khe, 1/Kbh,
                    t=self.t, dt=self.dt, kinetics='ICNS', sum=sum)
        
        elif self.kinetics == 'non-stationary uptake':
            self.Th_f = self.Th
            khe = dc.interp([self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], t)
            return t, dc.conc_liver(self.ca, 
                    self.ve, self.Te, self.De, khe, self.Th,
                    t=self.t, dt=self.dt, kinetics='ICNSU', sum=sum)
        
        elif self.kinetics == 'stationary':
            self.khe_f = self.khe
            self.Th_f = self.Th
            khe = self.khe*(1-self.Hct)
            return t, dc.conc_liver(self.ca, 
                    self.ve, self.Te, self.De, khe, self.Th,
                    t=self.t, dt=self.dt, kinetics='IC', sum=sum)
        
        elif self.kinetics == 'IC-HF':
            self.khe_f = self.khe
            self.Th_f = self.Th
            khe = self.khe*(1-self.Hct)
            return t, dc.conc_liver(self.ca, 
                self.ve, khe, self.Th,
                t=self.t, dt=self.dt, kinetics='IC-HF', sum=sum)

    def relax(self):
        """Tissue relaxation rate

        Returns:
            numpy.ndarray: Relaxation rate in 1/sec
        """
        t, C = self.conc(sum=False)
        rp = dc.relaxivity(self.field_strength, 'blood', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        return t, self.R10 + rp*C[0,:] + rh*C[1,:]
    
    def predict(self, xdata): 
        t, R1 = self.relax()
        # if self.signal == 'SR':
        #     signal = dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC, R10=self.R10)
        # else:
        #     signal = dc.signal_ss(R1, self.S0, self.TR, self.FA, R10=self.R10)
        if self.sequence == 'SR':
            signal = dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC)
        else:
            signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.dt)
    
    def train(self, xdata, ydata, **kwargs):
        if self.sequence == 'SR':
            Sref = dc.signal_sr(self.R10, 1, self.TR, self.FA, self.TC)
        else:
            Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:self.n0]) / Sref
        return dc.train(self, xdata, ydata, **kwargs)
    
    def export_params(self):
        pars = {}
        pars['ve']=["Liver extracellular volume fraction", self.ve, 'mL/mL']
        if self.kinetics=='IC-HF':
            pars['khe']=["Hepatocellular uptake rate", self.khe, 'mL/sec/mL']
            pars['Th']=["Hepatocellular transit time", self.Th, 'sec']
            pars['kbh']=["Biliary excretion rate", (1-self.ve)/self.Th, 'mL/sec/mL']
            pars['Khe']=["Hepatocellular tissue uptake rate", self.khe/self.ve, 'mL/sec/mL']
            pars['Kbh']=["Biliary tissue excretion rate", 1/self.Th, 'mL/sec/mL']
            if self.vol is not None:
                pars['CL']=['Liver blood clearance', self.khe*self.vol, 'mL/sec']
        elif self.kinetics=='stationary':
            pars['Te']=["Extracellular transit time", self.Te, 'sec']
            pars['De']=["Extracellular dispersion", self.De, '']
            pars['khe']=["Hepatocellular uptake rate", self.khe, 'mL/sec/mL']
            pars['Th']=["Hepatocellular transit time", self.Th, 'sec']
            pars['kbh']=["Biliary excretion rate", (1-self.ve)/self.Th, 'mL/sec/mL']
            pars['Khe']=["Hepatocellular tissue uptake rate", self.khe/self.ve, 'mL/sec/mL']
            pars['Kbh']=["Biliary tissue excretion rate", 1/self.Th, 'mL/sec/mL']
            if self.vol is not None:
                pars['CL']=['Liver blood clearance', self.khe*self.vol, 'mL/sec']
        else:
            pars['Te']=["Extracellular transit time", self.Te, 'sec']
            pars['De']=["Extracellular dispersion", self.De, '']
            khe = [self.khe, self.khe_f]
            Kbh = [1/self.Th, 1/self.Th_f]
            khe_avr = np.mean(khe)
            Kbh_avr = np.mean(Kbh)
            khe_var = (np.amax(khe)-np.amin(khe))/khe_avr
            Kbh_var = (np.amax(Kbh)-np.amin(Kbh))/Kbh_avr 
            kbh = np.mean((1-self.ve)*Kbh_avr)
            Th = np.mean(1/Kbh_avr)
            pars['khe']=["Hepatocellular uptake rate", khe_avr, 'mL/sec/mL']
            pars['Th']=["Hepatocellular transit time", Th, 'sec']
            pars['kbh']=["Biliary excretion rate", kbh, 'mL/sec/mL']
            pars['Khe']=["Hepatocellular tissue uptake rate", khe_avr/self.ve, 'mL/sec/mL']
            pars['Kbh']=["Biliary tissue excretion rate", Kbh_avr, 'mL/sec/mL']
            pars['khe_i']=["Hepatocellular uptake rate (initial)", self.khe, 'mL/sec/mL']
            pars['khe_f']=["Hepatocellular uptake rate (final)", self.khe_f, 'mL/sec/mL']
            pars['Th_i']=["Hepatocellular transit time (initial)", self.Th, 'sec']
            pars['Th_f']=["Hepatocellular transit time (final)", self.Th_f, 'sec']
            pars['khe_var']=["Hepatocellular uptake rate variance", khe_var, '']
            pars['Kbh_var']=["Biliary tissue excretion rate variance", Kbh_var, '']
            pars['kbh_i']=["Biliary excretion rate (initial)", (1-self.ve)/self.Th, 'mL/sec/mL']
            pars['kbh_f']=["Biliary excretion rate (final)", (1-self.ve)/self.Th_f, 'mL/sec/mL']
            if self.vol is not None:
                pars['CL']=['Liver blood clearance', khe_avr*self.vol, 'mL/sec']
        return self._add_sdev(pars)

    def plot(self, 
                xdata:tuple[np.ndarray, np.ndarray], 
                ydata:tuple[np.ndarray, np.ndarray],  
                testdata=None, xlim=None, fname=None, show=True):

        time, C = self.conc(sum=True)
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

    


