import numpy as np
import dcmri as dc

class AortaSignal6(dc.Model):
    """Whole-body aorta model acquired with a spoiled gradient echo sequence in steady-state - suitable for rapidly sampled data with shorter acquisition times.

    The model is intended for use in data where the acquisition time is sufficiently short so that backflux of indicator that has leaked out of the vasculature is negligible. It represents the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a chain (`flux_chain`) and the organs are modelled as a leaky compartment (`flux_comp`) without backflux of filtered indicator. Bolus injection into the system is modelled as a step function.

    The 6 free model parameters are:

    - *Bolus arrival time* (sec): time point where the indicator first arrives in the body. 
    - *Cardiac output* (mL/sec): Blood flow through the loop.
    - *Heart-lung mean transit time* (sec): average time to travel through heart and lungs.
    - *Heart-lung transit time dispersion*: the transit time through the heart-lung compartment as a fraction of the total transit time through the heart-lung system.
    - *Organs mean blood transit time* (sec): average time to travel through the organ's vasculature.
    - *Extraction fraction*: fraction of indicator extracted from the vasculature in a single pass. 

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `AortaSignal8b`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `dro_aif_1` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, cb = dc.dro_aif_1()
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):

        >>> aorta = dc.AortaSignal6(
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
        >>> ax1.plot(time/60, 1000*cb, 'ro', label='Measurement')
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
        Bolus arrival time: 22.0 (0.0) sec
        Cardiac output: 15.92 (5.28) L/min
        Heart-lung mean transit time: 6.87 (0.11) sec
        Heart-lung transit time dispersion: 26.99 (0.01) %
        Organs mean transit time: 36.77 (1.59) sec
        Extraction fraction: 14.16 (0.01) %

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
    S0 = 1                  #: Signal scaling factor in arbitrary units.

    def predict(self, xdata, return_conc=False, return_rel=False) ->np.ndarray:
        """Use the model to predict ydata for given xdata.

        Args:
            xdata (array-like): time points where the ydata are to be calculated.
            return_conc (bool, optional): If True, return the concentrations instead of the signal. Defaults to False.
            return_rel (bool, optional): If True, return the relaxation rate instead of the signal. Defaults to False.

        Returns:
            np.ndarray: array with predicted values, same length as xdata.
        """
        
        BAT, CO, T_hl, D_hl, Tp_o, E_b = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) 
        _, Jb = dc.body_flux_chc(Ji, 
                T_hl, D_hl,
                Tp_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO 
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 100, 10, 0.2, 20, 0.05])
        else:
            return np.array([30, 100, 10, 0.2, 20, 0.05])

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, 30, 0.95, 60, 0.15]
            lb = [0, 0, 0, 0.05, 0, 0.01]
        else:
            ub = [np.inf, np.inf, np.inf, 1.0, np.inf, 1.0]
            lb = [0, 0, 0, 0, 0, 0.0]
        return (lb, ub)

    def train(self, xdata, ydata, **kwargs):

        # Estimate BAT from data
        T, D = self.pars[2], self.pars[3]
        BAT = xdata[np.argmax(ydata)] - (1-D)*T
        self.pars[0] = BAT

        # Estimate S0 from data
        baseline = xdata[xdata <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:baseline]) / Sref

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['BAT', 'Bolus arrival time', self.pars[0], "sec"], 
            ['CO', 'Cardiac output', self.pars[1], "mL/sec"], 
            ['T_hl', 'Heart-lung mean transit time', self.pars[2], "sec"],
            ['D_hl', 'Heart-lung transit time dispersion', self.pars[3], ""],
            ['Tp_o', "Organs mean transit time", self.pars[4], "sec"],
            ['E', "Extraction fraction", self.pars[5], ""],
        ]
        if units == 'custom':
            pars[1][2:] = [self.pars[1]*60/1000, 'L/min']
            pars[3][2:] = [100*self.pars[3], '%']
            pars[5][2:] = [100*self.pars[5], '%']
        return pars

    
class AortaSignal8(dc.Model):
    """Whole-body aorta model acquired with a spoiled gradient echo sequence in steady-state - suitable for slowly sampled data with longer acquisition times.

    The model represents the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a plug-flow compartment (`flux_pfcomp`) and the organs are modelled as a two-compartment exchange model (`flux_2comp`). Bolus injection into the system is modelled as a step function.
    
    The 8 free model parameters are:

    - *Bolus arrival time* (sec): time point where the indicator first arrives in the body. 
    - *Cardiac output* (mL/sec): Blood flow through the loop.
    - *Heart-lung mean transit time* (sec): average time to travel through heart and lungs.
    - *Heart-lung transit time dispersion*: the transit time through the heart-lung compartment as a fraction of the total transit time through the heart-lung system.
    - *Extracellular extraction fraction*: Fraction of indicator entoring the organs which is extracted from the blood pool.
    - *Organs mean blood transit time* (sec): average time to travel through the organ's vasculature.
    - *Extravascular mean transit time* (sec): average time to travel through the organs extravascular space. 
    - *Body extraction fraction*: fraction of indicator extracted from the body in a single pass. 

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `AortaSignal6`, `AortaSignal8b`, `AortaSignal8c`, `AortaSignal10`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `dro_aif_1` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, cb = dc.dro_aif_1()
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):

        >>> aorta = dc.AortaSignal8(
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
        >>> ax1.plot(time/60, 1000*cb, 'ro', label='Measurement')
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
        Bolus arrival time: 22.0 (0.0) sec
        Cardiac output: 43.09 (346.04) L/min
        Heart-lung mean transit time: 3.58 (15.43) sec
        Heart-lung transit time dispersion: 100.0 (0.24) %
        Extracellular extraction fraction: 29.49 (37031.3) %
        Organs mean transit time: 0.0 (20.97) sec
        Extracellular mean transit time: 43.74 (5491451.89) sec
        Body extraction fraction: 0.93 (52034.08) %

        *Note*: the model provides a poor fit in the first pass, indicating that a compartment model does not adequately model the bolus dispersion through heart and lungs. A simple model of this type may be more useful in data that are sampled at lower temporal resolution, in which case the detailed structure is not visible and the model is more robust than more sophisticated heart-lung models.
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
    S0 = 1                  #: Signal scaling factor in arbitrary units.

    def predict(self, xdata, return_conc=False, return_rel=False) ->np.ndarray:
        """Use the model to predict ydata for given xdata.

        Args:
            xdata (array-like): time points where the ydata are to be calculated.
            return_conc (bool, optional): If True, return the concentrations instead of the signal. Defaults to False.
            return_rel (bool, optional): If True, return the relaxation rate instead of the signal. Defaults to False.

        Returns:
            np.ndarray: array with predicted values, same length as xdata.
        """
        
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) 
        _, Jb = dc.body_flux_ch2c(Ji, 
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO 
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 100, 10, 0.2, 0.15, 20, 120, 0.05])
        else:
            return np.array([30, 100, 10, 0.2, 0.15, 20, 120, 0.05])

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, 30, 0.95, 0.5, 60, 800, 0.15]
            lb = [0, 0, 0, 0.05, 0, 0, 0, 0.01]
        else:
            ub = [np.inf, np.inf, np.inf, 1.0, 1.0, np.inf, np.inf, 1.0]
            lb = [0, 0, 0, 0, 0, 0, 0, 0.0]
        return (lb, ub)

    def train(self, xdata, ydata, **kwargs):

        # Estimate BAT from data
        T, D = self.pars[2], self.pars[3]
        BAT = xdata[np.argmax(ydata)] - (1-D)*T
        self.pars[0] = BAT

        # Estimate S0 from data
        baseline = xdata[xdata <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(ydata[:baseline]) / Sref

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['BAT', 'Bolus arrival time', self.pars[0], "sec"], 
            ['CO', 'Cardiac output', self.pars[1], "mL/sec"], 
            ['T_hl', 'Heart-lung mean transit time', self.pars[2], "sec"],
            ['D_hl', 'Heart-lung transit time dispersion', self.pars[3], ""],
            ['E_o', "Extracellular extraction fraction", self.pars[4], ""],
            ['Tp_o', "Organs mean transit time", self.pars[5], "sec"],
            ['Te_o', "Extracellular mean transit time", self.pars[6], "sec"],
            ['E_b', "Body extraction fraction", self.pars[7], ""],
        ]
        if units == 'custom':
            pars[1][2:] = [self.pars[1]*60/1000, 'L/min']
            pars[3][2:] = [100*self.pars[3], '%']
            pars[4][2:] = [100*self.pars[4], '%']
            pars[7][2:] = [100*self.pars[7], '%']
        return pars
    
    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[5]+self.pars[6], 'sec'],
        ]
    
class AortaSignal8b(dc.Model):
    """Whole-body aorta model acquired with a spoiled gradient echo sequence in steady-state - suitable for slowly sampled data with longer acquisition times.

    The model represents the body as a leaky loop with a heart-lung system and an organ system. The heart-lung system is modelled as a plug-flow compartment (`flux_pfcomp`) and the organs are modelled as a two-compartment exchange model (`flux_2comp`). Bolus injection into the system is modelled as a step function.
    
    The 8 free model parameters are:

    - *Bolus arrival time* (sec): time point where the indicator first arrives in the body. 
    - *Cardiac output* (mL/sec): Blood flow through the loop.
    - *Heart-lung mean transit time* (sec): average time to travel through heart and lungs.
    - *Heart-lung transit time dispersion*: the transit time through the heart-lung compartment as a fraction of the total transit time through the heart-lung system.
    - *Extracellular extraction fraction*: Fraction of indicator entoring the organs which is extracted from the blood pool.
    - *Organs mean blood transit time* (sec): average time to travel through the organ's vasculature.
    - *Extravascular mean transit time* (sec): average time to travel through the organs extravascular space. 
    - *Body extraction fraction*: fraction of indicator extracted from the body in a single pass. 

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `AortaSignal6`, `AortaSignal8b`, `AortaSignal8c`, `AortaSignal10`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `dro_aif_1` to generate synthetic test data from experimentally-derived concentrations:

        >>> time, aif, cb = dc.dro_aif_1()
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_):

        >>> aorta = dc.AortaSignal8b(
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
        >>> ax1.plot(time/60, 1000*cb, 'ro', label='Measurement')
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
        Bolus arrival time: 22.0 (0.0) sec
        Cardiac output: 43.09 (346.04) L/min
        Heart-lung mean transit time: 3.58 (15.43) sec
        Heart-lung transit time dispersion: 100.0 (0.24) %
        Extracellular extraction fraction: 29.49 (37031.3) %
        Organs mean transit time: 0.0 (20.97) sec
        Extracellular mean transit time: 43.74 (5491451.89) sec
        Body extraction fraction: 0.93 (52034.08) %

        *Note*: the model provides a poor fit in the first pass, indicating that a compartment model does not adequately model the bolus dispersion through heart and lungs. A simple model of this type may be more useful in data that are sampled at lower temporal resolution, in which case the detailed structure is not visible and the model is more robust than more sophisticated heart-lung models.
    """         
    dt = 0.5                #: Pseudocontinuous time resolution of the simulation in sec.
    dose_tolerance = 0.1    #: Stopping criterion in the forward simulation of the arterial fluxes.
    weight = 70.0           #: Subject weight in kg.
    agent = 'gadoterate'    #: Contrast agent generic name.
    dose = 0.025            #: Injected contrast agent dose in mL per kg bodyweight.
    rate = 1                #: Contrast agent injection rate in mL per sec.
    field_strength = 3.0    #: Magnetic field strength in T.
    TC = 0.180              #: To to the center of k-space, in sec.
    R10 = 1.0               #: Precontrast relaxation rate (1/sec).
    S0 = 1                  #: Signal scaling factor in arbitrary units.

    def predict(self, xdata, return_conc=False, return_rel=False) ->np.ndarray:
        """Use the model to predict ydata for given xdata.

        Args:
            xdata (array-like): time points where the ydata are to be calculated.
            return_conc (bool, optional): If True, return the concentrations instead of the signal. Defaults to False.
            return_rel (bool, optional): If True, return the relaxation rate instead of the signal. Defaults to False.

        Returns:
            np.ndarray: array with predicted values, same length as xdata.
        """
        
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) 
        _, Jb = dc.body_flux_ch2c(Ji, 
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO 
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_src(R1, self.S0, self.TC)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 100, 10, 0.2, 0.15, 20, 120, 0.05])
        else:
            return np.array([30, 100, 10, 0.2, 0.15, 20, 120, 0.05])

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, 30, 0.95, 0.5, 60, 800, 0.15]
            lb = [0, 0, 0, 0.05, 0, 0, 0, 0.01]
        else:
            ub = [np.inf, np.inf, np.inf, 1.0, 1.0, np.inf, np.inf, 1.0]
            lb = [0, 0, 0, 0, 0, 0, 0, 0.0]
        return (lb, ub)

    def train(self, xdata, ydata, **kwargs):

        # Estimate BAT from data
        T, D = self.pars[2], self.pars[3]
        BAT = xdata[np.argmax(ydata)] - (1-D)*T
        self.pars[0] = BAT

        # Estimate S0 from data
        baseline = xdata[xdata <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_src(self.R10, 1, self.TC)
        self.S0 = np.mean(ydata[:baseline]) / Sref

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['BAT', 'Bolus arrival time', self.pars[0], "sec"], 
            ['CO', 'Cardiac output', self.pars[1], "mL/sec"], 
            ['T_hl', 'Heart-lung mean transit time', self.pars[2], "sec"],
            ['D_hl', 'Heart-lung transit time dispersion', self.pars[3], ""],
            ['E_o', "Extracellular extraction fraction", self.pars[4], ""],
            ['Tp_o', "Organs mean transit time", self.pars[5], "sec"],
            ['Te_o', "Extracellular mean transit time", self.pars[6], "sec"],
            ['E_b', "Body extraction fraction", self.pars[7], ""],
        ]
        if units == 'custom':
            pars[1][2:] = [self.pars[1]*60/1000, 'L/min']
            pars[3][2:] = [100*self.pars[3], '%']
            pars[4][2:] = [100*self.pars[4], '%']
            pars[7][2:] = [100*self.pars[7], '%']
        return pars
    
    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[5]+self.pars[6], 'sec'],
        ]
    

class AortaSignal10(dc.Model):
    """Whole-body aorta model acquired over 2 separate scans with a spoiled gradient echo sequence in steady-state - suitable when the tracer kinetics is so slow that data are acquired over two separate scan sessions. 

    The kinetic model is the same as for `AortaSignal8b`. It represents the body as a leaky loop with a heart-lung system and an organ system. Bolus injection into the system is modelled as a step function.
    
    The 10 free model parameters are:

    - *Bolus arrival time 1* (sec): time point where the indicator first arrives in the body. 
    - *Bolus arrival time 2* (sec): time point during the second scan when the indicator first arrives in the body. 
    - *Cardiac output* (mL/sec): Blood flow through the loop.
    - *Heart-lung mean transit time* (sec): average time to travel through heart and lungs.
    - *Heart-lung transit time dispersion*: the transit time through the heart-lung compartment as a fraction of the total transit time through the heart-lung system.
    - *Extracellular extraction fraction*: Fraction of indicator entoring the organs which is extracted from the blood pool.
    - *Organs mean blood transit time* (sec): average time to travel through the organ's vasculature.
    - *Extravascular mean transit time* (sec): average time to travel through the organs extravascular space. 
    - *Body extraction fraction*: fraction of indicator extracted from the body in a single pass. 
    - *Signal scale factor 2*: signal scale factor of the second scan.

    Args:
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `AortaSignal6`, `AortaSignal8b`

    Example:

        Use the model to reconstruct concentrations from experimentally derived signals.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `dro_aif_3` to generate synthetic test data from experimentally-derived concentrations. In this example we consider two separate bolus injections with a 2min break in between:

        >>> tacq, tbreak = 300, 120
        >>> time, aif, cb, = dc.dro_aif_3(tacq, tbreak)
        
        Build an aorta model and set weight, contrast agent, dose and rate to match the conditions of the original experiment (`Parker et al 2006 <https://doi.org/10.1002/mrm.21066>`_). Note in this case the dose is a list of two values:

        >>> aorta = dc.AortaSignal10(
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
        >>> ax1.plot(time/60, 1000*cb, 'ro', label='Measurement')
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
        Bolus arrival time 1: 22.0 (0.0) sec
        Bolus arrival time 2: 442.0 (0.0) sec
        Cardiac output: 27.99 (21.42) L/min
        Heart-lung mean transit time: 6.91 (0.16) sec
        Heart-lung transit time dispersion: 30.13 (0.02) %
        Extracellular extraction fraction: 7.33 (0.06) %
        Organs mean transit time: 23.32 (3.34) sec
        Extracellular mean transit time: 59.26 (69.7) sec
        Body extraction fraction: 11.34 (0.01) %
        Signal amplitude S02: 2.73 (0.08) a.u.

        *Note*: While the signal is reasonably well predicted, the reconstruction of the concentration is off. Since the same kinetic model reconstructs the single-scan concentrations well (`AortaSignal8b`) this appears to be due to the longer acquisition time for which the experimentally derived AIF models have not been validated.
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
    S0 = 1                  #: Signal scaling factor for the first acquisition (a.u.)
    R11 = 1.0               #: Estimate of the relaxation time before the second injection (1/sec)
    t1 = 1                  #: Start of second acquisition (sec).

    def predict(self, xdata, return_conc=False, return_rel=False):

        BAT1, BAT2, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b, S02 = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        J1 = dc.influx_step(t, self.weight, conc, self.dose[0], self.rate, BAT1)
        J2 = dc.influx_step(t, self.weight, conc, self.dose[1], self.rate, BAT2)
        _, Jb = dc.body_flux_ch2c(J1 + J2,
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO
        if return_conc:
            return dc.sample(xdata, t, cb, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        k = (t >= self.t1)
        signal[k] = dc.signal_ss(R1[k], S02, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 1200, 100, 10, 0.2, 0.15, 20, 120, 0.05, 1])
        else:
            return np.array([20, 120, 100, 10, 0.2, 0.15, 20, 120, 0.05, 1])

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, np.inf, 30, 0.95, 0.50, 60, 800, 0.15, np.inf]
            lb = [0,0,0,0,0.05,0,0,0,0.01,0]
        else:
            ub = [np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf]
            lb = np.zeros(10)
        return (lb, ub)
        
    def train(self, xdata, ydata, **kwargs):

        T, D = self.pars[3], self.pars[4]

        # Estimate BAT1 and S01 from data
        k = xdata < self.t1
        x, y = xdata[k], ydata[k]
        BAT1 = x[np.argmax(y)] - (1-D)*T
        baseline = x[x <= BAT1-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(y[:baseline]) / Sref
        self.pars[0] = BAT1

        # Estimate BAT2 and S02 from data
        k = xdata >= self.t1
        x, y = xdata[k], ydata[k]
        BAT2 = x[np.argmax(y)] - (1-D)*T
        baseline = x[x <= BAT2-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_ss(self.R11, 1, self.TR, self.FA)
        self.pars[-1] = np.mean(y[:baseline]) / Sref
        self.pars[1] = BAT2

        super().train(xdata, ydata, **kwargs)

    def pfree(self, units='standard'):
        pars = [
            ['BAT1', "Bolus arrival time 1", self.pars[0], "sec"],
            ['BAT2', "Bolus arrival time 2", self.pars[1], "sec"],
            ['CO', "Cardiac output", self.pars[2], "mL/sec"], # 6 L/min = 100 mL/sec
            ['T_hl', "Heart-lung mean transit time", self.pars[3], "sec"],
            ['D_hl', "Heart-lung transit time dispersion", self.pars[4], ""],
            ['E_o', "Extracellular extraction fraction", self.pars[5], ""],
            ['Tp_o', "Organs mean transit time", self.pars[6], "sec"],
            ['Te_o', "Extracellular mean transit time", self.pars[7], "sec"],
            ['E_b',"Body extraction fraction", self.pars[8], ""],
            ['S02', "Signal amplitude S02", self.pars[9], "a.u."],
            #['S01', "Signal amplitude S01", self.pars[9], "a.u."],
            #['S02', "Signal amplitude S02", self.pars[10], "a.u."],
        ]
        if units=='custom':
            pars[2][2:] = [pars[2][2]*60/1000, 'L/min']
            pars[4][2:] = [pars[4][2]*100, '%']
            pars[5][2:] = [pars[5][2]*100, '%']
            pars[8][2:] = [pars[8][2]*100, '%']
        return pars

    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[3]+self.pars[6], 'sec'],
        ]


