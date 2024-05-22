import numpy as np
import dcmri as dc

class AortaChCSS(dc.Model):
    """Whole-body aorta model acquired with a spoiled gradient echo sequence in steady-state - suitable for rapidly sampled data with shorter acquisition times.

    The model is intended for use in data where the acquisition time is sufficiently short so that backflux of indicator that has leaked out of the vasculature is negligible. It represents the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain (`flux_chain`) and the organs are modelled as a leaky compartment (`flux_comp`) without backflux of filtered indicator. Bolus injection into the system is modelled as a step function.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **BAT** (Bolus arrival time, sec): time point where the indicator first arrives in the body. 
    - **CO** (Cardiac output, mL/sec): Blood flow through the loop.
    - **Thl** (Heart-lung mean transit time, sec): average time to travel through heart and lungs.
    - **Dhl** (Heart-lung transit time dispersion): the transit time through the heart-lung compartment as a fraction of the total transit time through the heart-lung system.
    - **To** (Organs mean blood transit time, sec): average time to travel through the organ's vasculature.
    - **Eb** (Extraction fraction): fraction of indicator extracted from the vasculature in a single pass. 

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `AortaCh2CSS`

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

        >>> aorta = dc.AortaChCSS(
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'blood'),
        ... )

        Predict data with default parameters, train the model on the data, and predict data again:

        >>> aif0 = aorta.predict(time)
        >>> cb0 = aorta.predict(time, return_conc=True)
        >>> aorta.train(time, aif)
        >>> aif1 = aorta.predict(time)
        >>> cb1 = aorta.predict(time, return_conc=True)

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
        >>> ax1.plot(time/60, 1000*cb0, 'b--', label='Prediction (before training)')
        >>> ax1.plot(time/60, 1000*cb1, 'b-', label='Prediction (after training)')
        >>> ax1.set_ylim(0,5)
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Blood concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> aorta.print(round_to=3, units='custom')
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scale factor (S0): 114.382 (7.039) au
        Bolus arrival time (BAT): 22.319 (0.0) sec
        Cardiac output (CO): 17.929 (33.159) L/min
        Heart-lung mean transit time (Thl): 6.508 (0.138) sec
        Heart-lung transit time dispersion (Dhl): 33.773 (0.018) %
        Organs mean transit time (To): 37.56 (1.729) sec
        Extraction fraction (E): 12.017 (0.01) %

        *Note*: while the model fits the synthetic MRI signals well, the concentrations show some mismatch. The parameter values all have relatively small error and realistic values, except that the cardiac output is high for a typical adult in rest. This is a known property of the experimentally derived AIF used in this example (`Yang et al. 2009 <https://doi.org/10.1002/mrm.21912>`_).
    """         
    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec.
    dose_tolerance = 0.1    #: Stopping criterion in the forward simulation of the arterial fluxes.
    weight = 70.0           #: Subject weight in kg.
    agent = 'gadoterate'    #: Contrast agent generic name.
    dose = 0.025            #: Injected contrast agent dose in mL per kg bodyweight.
    rate = 1                #: Contrast agent injection rate in mL per sec.
    field_strength = 3.0    #: Magnetic field strength in T.
    TR = 0.005              #: Repetition time, or time between excitation pulses, in sec.
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1.0               #: Precontrast relaxation rate (1/sec).

    def predict(self, xdata, return_conc=False, return_rel=False) ->np.ndarray:
        """Use the model to predict ydata for given xdata.

        Args:
            xdata (array-like): time points where the ydata are to be calculated.
            return_conc (bool, optional): If True, return the concentrations instead of the signal. Defaults to False.
            return_rel (bool, optional): If True, return the relaxation rate instead of the signal. Defaults to False.

        Returns:
            np.ndarray: array with predicted values, same length as xdata.
        """
        
        S0, BAT, CO, Thl, Dhl, To, Eb = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) 
        _, Jb = dc.body_flux_chc(Ji, Thl, Dhl, To, Eb, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO 
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([1, 60, 100, 10, 0.2, 20, 0.05])
        else:
            return np.array([1, 30, 100, 10, 0.2, 20, 0.05])

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, np.inf, 30, 0.95, 60, 0.15]
            lb = [0, 0, 0, 0, 0.05, 0, 0.01]
        else:
            ub = [np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0]
            lb = 0
        return (lb, ub)

    def train(self, xdata, ydata, **kwargs):

        # Estimate BAT from data
        T, D = self.pars[3], self.pars[4]
        BAT = xdata[np.argmax(ydata)] - (1-D)*T
        self.pars[1] = BAT

        # Estimate S0 from data
        baseline = xdata[xdata <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.pars[0] = np.mean(ydata[:baseline]) / Sref

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['S0', 'Signal scale factor', self.pars[0], "au"],
            ['BAT', 'Bolus arrival time', self.pars[1], "sec"], 
            ['CO', 'Cardiac output', self.pars[2], "mL/sec"], 
            ['Thl', 'Heart-lung mean transit time', self.pars[3], "sec"],
            ['Dhl', 'Heart-lung transit time dispersion', self.pars[4], ""],
            ['To', "Organs mean transit time", self.pars[5], "sec"],
            ['Eb', "Extraction fraction", self.pars[6], ""],
        ]
        if units == 'custom':
            pars[2][2:] = [self.pars[2]*60/1000, 'L/min']
            pars[4][2:] = [self.pars[4]*100, '%']
            pars[6][2:] = [self.pars[6]*100, '%']
        return pars

    
class AortaCh2CSS(dc.Model):
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
        attr: provide values for any attributes as keyword arguments. 

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
        >>> cb0 = aorta.predict(time, return_conc=True)
        >>> aorta.train(time, aif)
        >>> aif1 = aorta.predict(time)
        >>> cb1 = aorta.predict(time, return_conc=True)

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
        >>> ax1.plot(time/60, 1000*cb0, 'b--', label='Prediction (before training)')
        >>> ax1.plot(time/60, 1000*cb1, 'b-', label='Prediction (after training)')
        >>> ax1.set_ylim(0,5)
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Blood concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> aorta.print(round_to=3, units='custom')
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
    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec.
    dose_tolerance = 0.1    #: Stopping criterion in the forward simulation of the arterial fluxes.
    weight = 70.0           #: Subject weight in kg.
    agent = 'gadoterate'    #: Contrast agent generic name.
    dose = 0.025            #: Injected contrast agent dose in mL per kg bodyweight.
    rate = 1                #: Contrast agent injection rate in mL per sec.
    field_strength = 3.0    #: Magnetic field strength in T.
    TR = 0.005              #: Repetition time, or time between excitation pulses, in sec.
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1.0               #: Precontrast relaxation rate (1/sec).

    def predict(self, xdata, return_conc=False, return_rel=False)->np.ndarray:
        """Use the model to predict ydata for given xdata.

        Args:
            xdata (array-like): time points where the ydata are to be calculated.
            return_conc (bool, optional): If True, return the concentrations instead of the signal. Defaults to False.
            return_rel (bool, optional): If True, return the relaxation rate instead of the signal. Defaults to False.

        Returns:
            np.ndarray: array with predicted values, same length as xdata.
        """
        S0, BAT, CO, Thl, Dhl, To, Eb, Eo, Te = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) 
        _, Jb = dc.body_flux_ch2c(Ji, 
                Thl, Dhl, Eo, To, Te, Eb, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO 
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])
    
    def pars0(self, settings=None):
        # S0, BAT, CO, Thl, Dhl, To, Eb, Eo, Te
        if settings == 'TRISTAN':
            return np.array([1, 60, 100, 10, 0.2, 20, 0.05, 0.15, 120])
        else:
            return np.array([1, 30, 100, 10, 0.2, 20, 0.05, 0.15, 120])
        
    def bounds(self, settings=None):
        # S0, BAT, CO, Thl, Dhl, To, Eb, Eo, Te
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, np.inf, 30, 0.95, 60, 0.15, 0.5, 800]
            lb = [0, 0, 0, 0, 0.05, 0, 0.01, 0, 0]
        else:
            ub = [np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, 1.0, np.inf]
            lb = 0
        return (lb, ub)

    def train(self, xdata, ydata, **kwargs):

        # Estimate BAT from data
        T, D = self.pars[3], self.pars[4]
        BAT = xdata[np.argmax(ydata)] - (1-D)*T
        self.pars[1] = BAT

        # Estimate S0 from data
        baseline = xdata[xdata <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.pars[0] = np.mean(ydata[:baseline]) / Sref

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['S0', 'Signal scale factor', self.pars[0], "au"],
            ['BAT', 'Bolus arrival time', self.pars[1], "sec"], 
            ['CO', 'Cardiac output', self.pars[2], "mL/sec"], 
            ['Thl', 'Heart-lung mean transit time', self.pars[3], "sec"],
            ['Dhl', 'Heart-lung transit time dispersion', self.pars[4], ""],
            ['To', "Organs mean transit time", self.pars[5], "sec"],
            ['Eb', "Body extraction fraction", self.pars[6], ""],
            ['Eo', "Organs extraction fraction", self.pars[7], ""],
            ['Te', "Extracellular mean transit time", self.pars[8], "sec"],
        ]
        if units == 'custom':
            pars[2][2:] = [self.pars[2]*60/1000, 'L/min']
            pars[4][2:] = [self.pars[4]*100, '%']
            pars[6][2:] = [self.pars[6]*100, '%']
            pars[7][2:] = [self.pars[7]*100, '%']
        return pars
    
    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[3]+self.pars[5], 'sec'],
        ]
    

class AortaCh2CSRC(dc.Model):
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
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `AortaChCSS`, `AortaCh2CSS`, `AortaCh2C2SS`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_sr` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, _, gt = dc.make_tissue_2cm_sr()
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):

        >>> aorta = dc.AortaCh2CSRC(
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
        >>> cb0 = aorta.predict(time, return_conc=True)
        >>> aorta.train(time, aif)
        >>> aif1 = aorta.predict(time)
        >>> cb1 = aorta.predict(time, return_conc=True)

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
        >>> ax1.plot(time/60, 1000*cb0, 'b--', label='Prediction (before training)')
        >>> ax1.plot(time/60, 1000*cb1, 'b-', label='Prediction (after training)')
        >>> ax1.set_ylim(0,5)
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Blood concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> aorta.print(round_to=3, units='custom')
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scale factor (S0): 263.691 (13.288) au
        Bolus arrival time (BAT): 22.0 (0.0) sec
        Cardiac output (CO): 27.653 (40.496) L/min
        Heart-lung mean transit time (Thl): 7.164 (0.289) sec
        Heart-lung transit time dispersion (Dhl): 44.786 (0.027) %
        Organs mean transit time (To): 22.226 (5.579) sec
        Body extraction fraction (Eb): 0.0 (0.291) %
        Organs extraction fraction (Eo): 17.15 (0.161) %
        Extracellular mean transit time (Te): 84.886 (282.258) sec
        ------------------
        Derived parameters
        ------------------
        Mean circulation time (Tc): 29.39 sec

    """         
    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec.
    dose_tolerance = 0.1    #: Stopping criterion in the forward simulation of the arterial fluxes.
    weight = 70.0           #: Subject weight in kg.
    agent = 'gadoterate'    #: Contrast agent generic name.
    dose = 0.025            #: Injected contrast agent dose in mL per kg bodyweight.
    rate = 1                #: Contrast agent injection rate in mL per sec.
    field_strength = 3.0    #: Magnetic field strength in T.
    TC = 0.180              #: Time to the center of k-space, in sec.
    R10 = 1.0               #: Precontrast relaxation rate (1/sec).

    def predict(self, xdata, return_conc=False, return_rel=False) ->np.ndarray:
        """Use the model to predict ydata for given xdata.

        Args:
            xdata (array-like): time points where the ydata are to be calculated.
            return_conc (bool, optional): If True, return the concentrations instead of the signal. Defaults to False.
            return_rel (bool, optional): If True, return the relaxation rate instead of the signal. Defaults to False.

        Returns:
            np.ndarray: array with predicted values, same length as xdata.
        """
        S0, BAT, CO, Thl, Dhl, To, Eb, Eo, Te = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) 
        _, Jb = dc.body_flux_ch2c(Ji, Thl, Dhl, Eo, To, Te, Eb, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO 
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_src(R1, S0, self.TC)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        # S0, BAT, CO, Thl, Dhl, To, Eb, Eo, Te
        if settings == 'TRISTAN':
            return np.array([1, 60, 100, 10, 0.2, 20, 0.05, 0.15, 120])
        else:
            return np.array([1, 30, 100, 10, 0.2, 20, 0.05, 0.15, 120])
        
    def bounds(self, settings=None):
        # S0, BAT, CO, Thl, Dhl, To, Eb, Eo, Te
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, np.inf, 30, 0.95, 60, 0.15, 0.5, 800]
            lb = [0, 0, 0, 0, 0.05, 0, 0.01, 0, 0]
        else:
            ub = [np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, 1.0, np.inf]
            lb = 0
        return (lb, ub)

    def train(self, xdata, ydata, **kwargs):

        # Estimate BAT from data
        T, D = self.pars[3], self.pars[4]
        BAT = xdata[np.argmax(ydata)] - (1-D)*T
        self.pars[1] = BAT

        # Estimate S0 from data
        baseline = xdata[xdata <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.pars[0] = np.mean(ydata[:baseline]) / Sref

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['S0', 'Signal scale factor', self.pars[0], "au"],
            ['BAT', 'Bolus arrival time', self.pars[1], "sec"], 
            ['CO', 'Cardiac output', self.pars[2], "mL/sec"], 
            ['Thl', 'Heart-lung mean transit time', self.pars[3], "sec"],
            ['Dhl', 'Heart-lung transit time dispersion', self.pars[4], ""],
            ['To', "Organs mean transit time", self.pars[5], "sec"],
            ['Eb', "Body extraction fraction", self.pars[6], ""],
            ['Eo', "Organs extraction fraction", self.pars[7], ""],
            ['Te', "Extracellular mean transit time", self.pars[8], "sec"],
        ]
        if units == 'custom':
            pars[2][2:] = [self.pars[2]*60/1000, 'L/min']
            pars[4][2:] = [self.pars[4]*100, '%']
            pars[6][2:] = [self.pars[6]*100, '%']
            pars[7][2:] = [self.pars[7]*100, '%']
        return pars
    
    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[3]+self.pars[5], 'sec'],
        ]
    

class AortaCh2C2SS(dc.Model):
    """Whole-body aorta model acquired over 2 separate scans with a spoiled gradient echo sequence in steady-state - suitable when the tracer kinetics is so slow that data are acquired over two separate scan sessions. 

    The kinetic model is the same as for `AortaCh2CSS`. It represents the body as a leaky loop with a heart-lung system and an organ system. Bolus injection into the system is modelled as a step function.
    
    The 10 free model parameters are:

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
        attr: provide values for any attributes as keyword arguments. 

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
        >>> cb0 = aorta.predict(time, return_conc=True)
        >>> aorta.train(time, aif)
        >>> aif1 = aorta.predict(time)
        >>> cb1 = aorta.predict(time, return_conc=True)

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
        >>> ax1.plot(time/60, 1000*cb0, 'b--', label='Prediction (before training)')
        >>> ax1.plot(time/60, 1000*cb1, 'b-', label='Prediction (after training)')
        >>> ax1.set_ylim(0,5)
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Blood concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> aorta.print(round_to=2, units='custom')
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal amplitude S01 (S01): 167.815 (28.58) a.u.
        Signal amplitude S02 (S02): 621.444 (177.108) a.u.
        Bolus arrival time 1 (BAT1): 77.132 (0.0) sec
        Bolus arrival time 2 (BAT2): 793.088 (0.0) sec
        Cardiac output (CO): 115.01 (991.953) L/min
        Heart-lung mean transit time (Thl): 9.125 (10.358) sec
        Heart-lung transit time dispersion (Dhl): 98.3 (1.634) %
        Organs mean transit time (To): 3.366 (22.975) sec
        Body extraction fraction (Eb): 2.224 (0.128) %
        Organs extraction fraction (Eo): 5.868 (1.486) %
        Extracellular mean transit time (Te): 16.842 (321.081) sec
        ------------------
        Derived parameters
        ------------------
        Mean circulation time (Tc): 12.491 sec
    """  

    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec.
    dose_tolerance = 0.1    #: Stopping criterion in the forward simulation of the arterial fluxes.
    weight = 70.0           #: Subject weight in kg.
    agent = 'gadoterate'    #: Contrast agent generic name.
    dose = [0.025, 0.025]   #: Injected contrast agent doses for first and second scan in mL per kg bodyweight.
    rate = 1                #: Contrast agent injection rate in mL per sec.
    field_strength = 3.0    #: Magnetic field strength in T.
    TR = 0.005              #: Repetition time, or time between excitation pulses, in sec.
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1.0               #: Precontrast relaxation rate (1/sec).
    R11 = 1.0               #: Estimate of the relaxation time before the second injection (1/sec)
    t1 = 1                  #: Start of second acquisition (sec).

    def predict(self, xdata, return_conc=False, return_rel=False):

        # S01, S02, BAT1, BAT2, CO, Thl, Dhl, E_o, To, Te_o, Eb = self.pars
        S01, S02, BAT1, BAT2, CO, Thl, Dhl, To, Eb, Eo, Te = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        J1 = dc.influx_step(t, self.weight, conc, self.dose[0], self.rate, BAT1)
        J2 = dc.influx_step(t, self.weight, conc, self.dose[1], self.rate, BAT2)
        _, Jb = dc.body_flux_ch2c(J1 + J2, 
                Thl, Dhl, Eo, To, Te, Eb, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, S01, self.TR, self.FA)
        k = (t >= self.t1)
        signal[k] = dc.signal_ss(R1[k], S02, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([1, 1, 60, 1200, 100, 10, 0.2, 20, 0.05, 0.15, 120])
        else:
            return np.array([1, 1, 20, 120, 100, 10, 0.2, 20, 0.05, 0.15, 120])

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, np.inf, np.inf, np.inf, 30, 0.95, 60, 0.15, 0.5, 800]
            lb = [0,0,0,0,0, 0, 0.05, 0, 0.01, 0, 0]
        else:
            ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, 1.0, np.inf]
            lb = 0
        return (lb, ub)
        
    def train(self, xdata, ydata, pfix=None, **kwargs):

        T, D = self.pars[5], self.pars[6]

        # Estimate BAT1 and S01 from data
        k = xdata < self.t1
        x, y = xdata[k], ydata[k]
        BAT1 = x[np.argmax(y)] - (1-D)*T
        baseline = x[x <= BAT1-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.pars[0] = np.mean(y[:baseline]) / Sref
        self.pars[2] = BAT1

        # Estimate BAT2 and S02 from data
        k = xdata >= self.t1
        x, y = xdata[k], ydata[k]
        BAT2 = x[np.argmax(y)] - (1-D)*T
        baseline = x[x <= BAT2-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_ss(self.R11, 1, self.TR, self.FA)
        self.pars[1] = np.mean(y[:baseline]) / Sref
        self.pars[3] = BAT2

        # The default operation is to keep S01 fixed.
        if pfix is None:
            pfix = pfix=[1]+10*[0]

        super().train(xdata, ydata, pfix=pfix, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['S01', "Signal amplitude S01", self.pars[0], "a.u."],
            ['S02', "Signal amplitude S02", self.pars[1], "a.u."],
            ['BAT1', "Bolus arrival time 1", self.pars[2], "sec"],
            ['BAT2', "Bolus arrival time 2", self.pars[3], "sec"],
            ['CO', 'Cardiac output', self.pars[4], "mL/sec"], 
            ['Thl', 'Heart-lung mean transit time', self.pars[5], "sec"],
            ['Dhl', 'Heart-lung transit time dispersion', self.pars[6], ""],
            ['To', "Organs mean transit time", self.pars[7], "sec"],
            ['Eb', "Body extraction fraction", self.pars[8], ""],
            ['Eo', "Organs extraction fraction", self.pars[9], ""],
            ['Te', "Extracellular mean transit time", self.pars[10], "sec"],
        ]
        if units=='custom':
            pars[4][2:] = [pars[4][2]*60/1000, 'L/min']
            pars[6][2:] = [pars[6][2]*100, '%']
            pars[8][2:] = [pars[8][2]*100, '%']
            pars[9][2:] = [pars[9][2]*100, '%']
        return pars

    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[5]+self.pars[7], 'sec'],
        ]


