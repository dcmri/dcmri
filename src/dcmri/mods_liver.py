import numpy as np
import dcmri as dc


class Liver(dc.Model):

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

    def __init__(self, **attr):
        # Set defaults
        self.cb = np.ones(100)
        self.dt = 0.5               
        self.Hct = 0.45             
        self.agent = 'gadoxetate'    
        self.field_strength = 3.0    
        self.TR = 0.005  
        self.FA = 15.0  
        self.TC = 0.180
        self.R10 = 1                 
        self.S0 = 1               
        self.t0 = 1 
        self.vol = None

        self.Te = 30.0
        self.De = 0.85
        self.ve = 0.3 
        self.khe = 20/6000
        self.Th = 30*60
        self.khe_f = 20/6000
        self.Th_f = 30*60
        self.signal = 'SS'
        self.kinetics = 'non-stationary'
        self.free = ['Te','De','ve','khe','Th','khe_f','Th_f']
        self.bounds = [[0.1, 0, 0.01, 0, 10*60, 0, 10*60],
                       [60, 1, 0.6, 0.1, 10*60*60, 0.1, 10*60*60]]
        dc.init(self, **attr) 
        # Precompute    
        self.ca = self.cb/(1-self.Hct)                #: Arterial plasma concentration (M)
        self.t = self.dt*np.arange(np.size(self.cb))

    def conc(self, sum=True):
        """Tissue concentrations

        Args:
            sum (bool, optional): If True, returns the total concentrations. If False, returns concentration in both compartments separately. Defaults to True.

        Returns:
            numpy.ndarray: Concentration in M
        """
        if self.kinetics == 'non-stationary':
            khe = dc.interp([self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], self.t)
            Kbh = dc.interp([1/self.Th, 1/self.Th_f], self.t)
            hepatocytes = ['nscomp', 1/Kbh]
        elif self.kinetics == 'non-stationary uptake':
            self.Th_f = self.Th
            khe = dc.interp([self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], self.t)
            hepatocytes = ['comp', self.Th]
        else:
            self.khe_f = self.khe
            self.Th_f = self.Th
            khe = self.khe*(1-self.Hct)
            hepatocytes = ['comp', self.Th]
        return self.t, dc.conc_liver_hep(
                self.ca, self.ve, khe, dt=self.dt, sum=sum,
                extracellular = ['pfcomp', (self.Te, self.De)],
                #extracellular = ['chain', (self.Te, self.De)],
                hepatocytes = hepatocytes)

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
        if self.signal == 'SR':
            signal = dc.signal_sr(R1, self.S0, self.TR, self.FA, self.TC)
        else:
            signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, self.t, signal, xdata[1]-xdata[0])
    
    def train(self, xdata, ydata, **kwargs):
        n0 = max([np.sum(xdata<self.t0), 1])
        if self.signal == 'SR':
            Sref = dc.signal_sr(self.R10, 1, self.TR, self.FA, self.TC)
        else:
            Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:n0]) / Sref
        return dc.train(self, xdata, ydata, **kwargs)
    
    def pars(self):
        pars = {}
        pars['Te']=["Extracellular transit time", self.Te, 'sec']
        pars['De']=["Extracellular dispersion", self.De, '']
        pars['ve']=["Liver extracellular volume fraction", self.ve, 'mL/mL']
        if self.kinetics=='stationary':
            pars['khe']=["Hepatocellular uptake rate", self.khe, 'mL/sec/mL']
            pars['Th']=["Hepatocellular transit time", self.Th, 'sec']
            pars['kbh']=["Biliary excretion rate", (1-self.ve)/self.Th, 'mL/sec/mL']
            pars['Khe']=["Hepatocellular tissue uptake rate", self.khe/self.ve, 'mL/sec/mL']
            pars['Kbh']=["Biliary tissue excretion rate", 1/self.Th, 'mL/sec/mL']
            if self.vol is not None:
                pars['CL']=['Liver blood clearance', self.khe*self.vol, 'mL/sec']
        else:
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
        return self.add_sdev(pars)


    
class Liver2scan(Liver):

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
    
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_2ss` to generate synthetic test data over 2 scans:

        >>> time, _, roi, gt = dc.make_tissue_2cm_2ss(CNR=100, R10=1/dc.T1(3.0,'liver'))
        
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
        ... )

        Train the model on the ROI data, fixing the baseline scaling factor, and predict the data:

        >>> model.train(time, roi)
        >>> sig = model.predict(time)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> #
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(np.concatenate(time)/60, np.concatenate(roi), marker='o', linestyle='None', color='cornflowerblue', label='Data')
        >>> ax0.plot(np.concatenate(time)/60, np.concatenate(sig), marker='x', linestyle='None', color='darkblue', label='Prediction')
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

    def __init__(self, **attr):
        # Set defaults
        super().__init__()
        self.S02 = 1
        self.R102 = 1
        self.free += ['S02']
        self.bounds[0] += [0]
        self.bounds[1] += [np.inf]
        dc.init(self, **attr)     
        # Precompute
        self.ca = self.cb/(1-self.Hct)                #: Arterial plasma concentration (M)
        self.t = self.dt*np.arange(np.size(self.cb))

    def predict(self, xdata:tuple[np.ndarray, np.ndarray]):
        t, R1 = self.relax()
        t1 = t<=xdata[0][-1]
        t2 = t>=xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        if self.signal == 'SR':
            signal1 = dc.signal_sr(R11, self.S0, self.TR, self.FA, self.TC)
            signal2 = dc.signal_sr(R12, self.S02, self.TR, self.FA, self.TC)
        else:
            signal1 = dc.signal_ss(R11, self.S0, self.TR, self.FA)
            signal2 = dc.signal_ss(R12, self.S02, self.TR, self.FA)
        return (
            dc.sample(xdata[0], t[t1], signal1, xdata[0][1]-xdata[0][0]),
            dc.sample(xdata[1], t[t2], signal2, xdata[1][1]-xdata[1][0]),
        )
    
    def train(self, 
              xdata:tuple[np.ndarray, np.ndarray], 
              ydata:tuple[np.ndarray, np.ndarray], **kwargs):
    
        if self.signal == 'SR':
            Sref1 = dc.signal_sr(self.R10, 1, self.TR, self.FA, self.TC)
            Sref2 = dc.signal_sr(self.R102, 1, self.TR, self.FA, self.TC)
        else:
            Sref1 = dc.signal_ss(self.R10, 1, self.TR, self.FA)
            Sref2 = dc.signal_ss(self.R102, 1, self.TR, self.FA)

        n0 = max([np.sum(xdata[0]<self.t0), 2])
        self.S0 = np.mean(ydata[0][1:n0]) / Sref1 
        self.S02 = np.mean(ydata[1][1:n0]) / Sref2
        return dc.train(self, xdata, ydata, **kwargs)

