import numpy as np
import dcmri as dc


class Aorta(dc.Model):
    """Whole-body aorta model acquired with a spoiled gradient echo sequence in steady-state - suitable for slowly sampled data with longer acquisition times.

    The model represents the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain compartment (`flux_chain`) and the organs are modelled as a two-compartment exchange model (`flux_2comp`). Bolus injection into the system is modelled as a step function.
    
        The free model parameters are:

        - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
        - **BAT** (Bolus arrival time, sec): time point where the indicator first arrives in the body. 
        - **CO** (Cardiac output, mL/sec): Blood flow through the body.
        - **Thl** (Heart-lung mean transit time, sec): average time to travel through heart and lungs.
        - **Dhl** (Heart-lung transit time dispersion): Dispersion through the heart-lung system, with a value in the range [0,1].
        - **To** (Organs mean blood transit time, sec): average time to travel through the organ's vasculature.
        - **Eb** (Extraction fraction): fraction of indicator extracted from the vasculature in a single pass. 
        - **Eo** (Organs extraction fraction): Fraction of indicator entering the organs which is extracted from the blood pool.
        - **Te** (Extravascular mean transit time, sec): average time to travel through the organs extravascular space.

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Internal time resolution of the AIF in sec. 
        dose_tolerance (float, optional): Stopping criterion in the forward simulation of the arterial fluxes.
        weight (float, optional): Subject weight in kg.
        agent (str, optional): Contrast agent generic name.
        dose (float, optional): Injected contrast agent dose in mL per kg bodyweight.
        rate (float, optional): Contrast agent injection rate in mL per sec.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec. 

    See Also:
        `AortaChCSS`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, _, gt = dc.make_tissue_2cm_ss()
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):

        >>> aorta = dc.AortaCh2CSS(
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'blood'),
        ... )

        Predict concentrations with default parameters, train the model on the data, and predict concentrations again:

        >>> aif0 = aorta.predict(time)
        >>> t0, cb0 = aorta.conc()
        >>> aorta.train(time, aif)
        >>> aif1 = aorta.predict(time)
        >>> t1, cb1 = aorta.conc()

        Plot the reconstructed signals and concentrations and compare against the experimentally derived data:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> # 
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(time/60, aif, 'ro', label='Measurement')
        >>> ax0.plot(time/60, aif0, 'b--', label='Prediction (before training)')
        >>> ax0.plot(time/60, aif1, 'b-', label='Prediction (after training)')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()
        >>> # 
        >>> ax1.set_title('Prediction of the concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['cb'], 'ro', label='Measurement')
        >>> ax1.plot(t0/60, 1000*cb0, 'b--', label='Prediction (before training)')
        >>> ax1.plot(t1/60, 1000*cb1, 'b-', label='Prediction (after training)')
        >>> ax1.set_ylim(0,5)
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Blood concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> aorta.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scale factor (S0): 112.42 (6.155) au
        Bolus arrival time (BAT): 21.835 (0.0) sec
        Cardiac output (CO): 19.516 (32.508) L/min
        Heart-lung mean transit time (Thl): 6.602 (0.131) sec
        Heart-lung transit time dispersion (Dhl): 29.116 (0.015) %
        Organs mean transit time (To): 29.151 (6.883) sec
        Body extraction fraction (Eb): 0.0 (0.414) %
        Organs extraction fraction (Eo): 23.141 (0.225) %
        Extracellular mean transit time (Te): 87.394 (271.401) sec
        ------------------
        Derived parameters
        ------------------
        Mean circulation time (Tc): 35.753 sec

        *Note*: The extracellular mean transit time has a high error, indicating that the acquisition time here is insufficient to resolve the transit through the leakage space.
    """         

    def __init__(self, **attr):
        self.dt = 0.5   
        self.tmax = 120             
        self.dose_tolerance = 0.1   
        self.weight = 70.0          
        self.agent = 'gadoterate'   
        self.dose = dc.ca_std_dose('gadoterate') 
        self.rate = 1  
        self.field_strength = 3.0
        self.R10 = 1/dc.T1(3.0, 'blood') 
        self.t0 = 0

        self.TR = 0.005 
        self.FA = 15.0 
        self.TC = 0.180

        self.S0 = 1
        self.BAT = 60.0
        self.CO = 100
        self.Thl = 10
        self.Dhl = 0.2
        self.To = 20
        self.Eb = 0.05
        self.Eo = 0.15
        self.Te = 120

        self.signal = 'SS'
        self.organs = '2cxm'
        self.free = ['BAT','CO','Thl','Dhl','To','Eb','Eo','Te']
        self.bounds = [
                [0, 0, 0, 0.05, 0, 0.01, 0, 0],
                [np.inf, np.inf, 30, 0.95, 60, 0.15, 0.5, 800],
            ]
        dc.init(self, **attr)

    def conc(self):
        """Aorta blood concentration

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        if self.organs == 'comp':
            organs = ['comp', self.To]
        else:
            organs = ['2cxm', (self.To, self.Te, self.Eo)]
        t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, 
                conc, self.dose, self.rate, self.BAT) 
        Jb = dc.flux_aorta(Ji, E=self.Eb,
            heartlung = ['pfcomp', (self.Thl, self.Dhl)],
            organs=organs, dt=self.dt, tol=self.dose_tolerance)
        return t, Jb/self.CO
    
    def relax(self):
        """Aorta longitudinal relation rate

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        t, cb = self.conc()
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10 + rp*cb

    def predict(self, xdata) -> np.ndarray:
        self.tmax = max(xdata)+xdata[1]+self.dt
        t, R1 = self.relax()
        if self.signal == 'SR':
            #signal = dc.signal_src(R1, self.S0, self.TC, R10=self.R10)
            signal = dc.signal_src(R1, self.S0, self.TC)
        else:
            #signal = dc.signal_ss(R1, self.S0, self.TR, self.FA, R10=self.R10)
            signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def train(self, xdata, ydata, **kwargs):
        n0 = max([np.sum(xdata<self.t0), 1])
        self.BAT = xdata[np.argmax(ydata)] - (1-self.Dhl)*self.Thl
        if self.signal == 'SR':
            Sref = dc.signal_src(self.R10, 1, self.TC)
        else:
            Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:n0]) / Sref
        return dc.train(self, xdata, ydata, **kwargs)
    
    def pars(self):
        pars = {}
        pars['BAT']= ['Bolus arrival time', self.BAT, "sec"] 
        pars['CO']= ['Cardiac output', self.CO, "mL/sec"]
        pars['Thl']= ['Heart-lung mean transit time', self.Thl, "sec"]
        pars['Dhl']= ['Heart-lung transit time dispersion', self.Dhl, ""]
        pars['To']= ["Organs mean transit time", self.To, "sec"]
        pars['Eb']= ["Extraction fraction", self.Eb, ""]
        pars['Tc']= ["Mean circulation time", self.Thl+self.To, 'sec'] 
        pars['Eo'] = ["Organs extraction fraction", self.Eo, ""]
        pars['Te'] = ["Extracellular mean transit time", self.Te, "sec"]
        #pars['S0'] = ["Baseline", self.S0, "au"]
        return self.add_sdev(pars)
    


class Aorta2scan(Aorta):
    """Whole-body aorta model acquired over 2 separate scans with a spoiled gradient echo sequence in steady-state - suitable when the tracer kinetics is so slow that data are acquired over two separate scan sessions. 

    The kinetic model is the same as for `AortaCh2CSS`. It represents the body as a leaky loop with a heart-lung system and an organ system. Bolus injection into the system is modelled as a step function.
    
        The free model parameters are:

        - **S01** (Signal scaling factor 1, a.u.): scale factor for the first MR signal.
        - **S02** (Signal scaling factor 2, a.u.): scale factor for the second MR signal.
        - **BAT1** (Bolus arrival time 1, sec): time point where the indicator first arrives in the body.
        - **BAT2** (Bolus arrival time 1, sec): time point when the second bolus injection first arrives in the body. 
        - **CO** (Cardiac output, mL/sec): Blood flow through the body.
        - **Thl** (Heart-lung mean transit time, sec): average time to travel through heart and lungs.
        - **Dhl** (Heart-lung transit time dispersion): Dispersion through the heart-lung system, with a value in the range [0,1].
        - **To** (Organs mean blood transit time, sec): average time to travel through the organ's vasculature.
        - **Eb** (Extraction fraction): fraction of indicator extracted from the vasculature in a single pass. 
        - **Eo** (Organs extraction fraction): Fraction of indicator entering the organs which is extracted from the blood pool.
        - **Te** (Extravascular mean transit time, sec): average time to travel through the organs extravascular space.

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        dt (float, optional): Internal time resolution of the AIF in sec. 
        dose_tolerance (float, optional): Stopping criterion in the forward simulation of the arterial fluxes.
        weight (float, optional): Subject weight in kg.
        agent (str, optional): Contrast agent generic name.
        dose (array-like, optional): 2-element array with injected contrast agent doses for first and second scan in mL per kg bodyweight.
        rate (float, optional): Contrast agent injection rate in mL per sec.
        field_strength (float, optional): Magnetic field strength in T. 
        TR (float, optional): Repetition time, or time between excitation pulses, in sec. 
        FA (float, optional): Nominal flip angle in degrees.
        R10 (float, optional): Precontrast tissue relaxation rate in 1/sec.
        R11 (float, optional): Estimate of the relaxation time before the second injection (1/sec)
        to (float, optional): Start of second acquisition (sec).

    See Also:
        `AortaChCSS`, `AortaCh2CSRC`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_2ss` to generate synthetic test data from experimentally-derived concentrations. In this example we consider two separate bolus injections with a 2min break in between:

        >>> tacq, tbreak = 300, 120
        >>> time, aif, _, gt = dc.make_tissue_2cm_2ss(tacq, tbreak)
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_). Note in this case the dose is a list of two values:

        >>> aorta = dc.AortaCh2C2SS(
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = [0.2,0.2],
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'blood'),
        ...     R11 = 1/dc.T1(3.0,'blood'),
        ...     t1 = tacq + tbreak,
        ... )

        Predict concentrations with default parameters, train the model on the data, and predict concentrations again:

        >>> aif0 = aorta.predict(time)
        >>> t0, cb0 = aorta.conc()
        >>> aorta.train(time, aif)
        >>> aif1 = aorta.predict(time)
        >>> t1, cb1 = aorta.conc()

        Plot the reconstructed signals and concentrations and compare against the experimentally derived data:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))
        >>> # 
        >>> ax0.set_title('Prediction of the MRI signals.')
        >>> ax0.plot(np.concatenate(time)/60, np.concatenate(aif), 'ro', label='Measurement')
        >>> ax0.plot(np.concatenate(time)/60, np.concatenate(aif0), 'b--', label='Prediction (before training)')
        >>> ax0.plot(np.concatenate(time)/60, np.concatenate(aif1), aif1, 'b-', label='Prediction (after training)')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('MRI signal (a.u.)')
        >>> ax0.legend()
        >>> # 
        >>> ax1.set_title('Prediction of the concentrations.')
        >>> ax1.plot(gt['t']/60, 1000*gt['cb'], 'ro', label='Measurement')
        >>> ax1.plot(t0/60, 1000*cb0, 'b--', label='Prediction (before training)')
        >>> ax1.plot(t0/60, 1000*cb1, 'b-', label='Prediction (after training)')
        >>> ax1.set_ylim(0,5)
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Blood concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()
    """  
 
    def __init__(self, **attr):
        super().__init__()
        self.dose = [dc.ca_std_dose('gadoterate')/2, dc.ca_std_dose('gadoterate')/2] 
        self.S02 = 1
        self.BAT2 = 1200 
        self.R102 = 1  
        self.free += ['BAT2','S02']
        self.bounds[0] += [0, 0]
        self.bounds[1] += [np.inf, np.inf]
        dc.init(self, **attr)

    def conc(self):
        """Aorta blood concentration

        Args:
            t (array-like): Time points of the concentration (sec)

        Returns:
            numpy.ndarray: Concentration in M
        """
        if self.organs == 'comp':
            organs = ['comp', self.To]
        else:
            organs = ['2cxm', (self.To, self.Te, self.Eo)]
        t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        J1 = dc.influx_step(t, self.weight, conc, self.dose[0], self.rate, self.BAT)
        J2 = dc.influx_step(t, self.weight, conc, self.dose[1], self.rate, self.BAT2)
        Jb = dc.flux_aorta(J1 + J2, E=self.Eb,
            heartlung = ['pfcomp', (self.Thl, self.Dhl)],
            organs=organs, dt=self.dt, tol=self.dose_tolerance)
        return t, Jb/self.CO
    
    def predict(self, xdata:tuple[np.ndarray, np.ndarray]):
        self.tmax = max(xdata[1])+xdata[0][1]+self.dt
        t, R1 = self.relax()

        # predict first scan
        t1 = t<=xdata[0][-1]
        t2 = t>=xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        if self.signal == 'SR':
            signal0 = dc.signal_src(R11, self.S0, self.TC)
            signal1 = dc.signal_src(R12, self.S02, self.TC)
        else:
            signal0 = dc.signal_ss(R11, self.S0, self.TR, self.FA)
            signal1 = dc.signal_ss(R12, self.S02, self.TR, self.FA)  
        return (
            dc.sample(xdata[0], t[t1], signal0, xdata[0][2]-xdata[0][1]),
            dc.sample(xdata[1], t[t2], signal1, xdata[1][2]-xdata[1][1]),
        )

    def train(self, 
              xdata:tuple[np.ndarray, np.ndarray], 
              ydata:tuple[np.ndarray, np.ndarray], **kwargs):

        # Estimate BAT and S0 from data
        T, D = self.Thl, self.Dhl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self.BAT2 = xdata[1][np.argmax(ydata[1])] - (1-D)*T
        if self.signal == 'SR':
            Sref0 = dc.signal_src(self.R10, 1, self.TC)
            Sref1 = dc.signal_src(self.R102, 1, self.TC)
        else:
            Sref0 = dc.signal_ss(self.R10, 1, self.TR, self.FA)
            Sref1 = dc.signal_ss(self.R102, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata[0]<self.t0), 1])
        self.S0 = np.mean(ydata[0][:n0]) / Sref0
        self.S02 = np.mean(ydata[1][:n0]) / Sref1
        return dc.train(self, xdata, ydata, **kwargs)

    def pars(self):
        pars = {}
        pars['BAT2'] = ["Second bolus arrival time", self.BAT2, "sec"]
        return self.add_sdev(pars) | super().pars()
    