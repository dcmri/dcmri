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

        >>> model.train(time, roi, pfix=[1]+5*[0])

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
    def __init__(self, aif, 
            pars = None,
            dt = 0.5, 
            Hct = 0.45, 
            agent = 'gadoterate',
            field_strength = 3.0,
            TR = 5.0/1000.0,  
            FA = 15.0,
            Tsat = 0,
            TC = 0.085,
            R10 = 1,  
            R10b = 1,
            t0 = 0,
            vol = None, 
        ):
        n0 = max([round(t0/dt),1])
        r1 = dc.relaxivity(field_strength, 'blood', agent)
        cb = dc.conc_src(aif, TC, 1/R10b, r1, n0)
        self._t = dt*np.arange(np.size(aif))
        self._dt = dt
        self._Hct = Hct
        self._TR = TR
        self._FA = FA
        self._Tsat = Tsat
        self._TC = TC
        self._R10 = R10
        self._R10b = R10b
        self._r1 = r1
        self._t0 = t0
        self._vol = vol
        self.pars = self.pars0(pars)
        self.ca = cb/(1-Hct)        #: Arterial plasma concentration (M)

    def _forward_model(self, xdata):
        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg) 
        S0, Fp, Tp, Ft, Tt, Ta = self.pars
        C = self.conc()
        R1 = self._R10 + self._r1*C
        signal = dc.signal_sr(R1, S0, self._TR, self._FA, self._TC, self._Tsat)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])
    
    def conc(self):
        """Tissue concentration

        Returns:
            numpy.ndarray: Concentration in M
        """
        S0, Fp, Tp, Ft, Tt, Ta = self.pars
        ca = dc.flux_plug(self.ca, Ta, self._t)
        return dc.kidney_conc_2cm(ca, Fp, Tp, Ft, Tt, self._t)
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_sr(self._R10, 1, self._TR, self._FA, self._TC, self._Tsat)
        n0 = max([np.sum(xdata<self._t0), 1])
        self.pars[0] = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)
    
    def pars0(self, settings=None):
        if settings == None:
            return np.array([1, 200/6000, 5, 30/6000, 120, 0])
        else:
            return np.zeros(6)

    def bounds(self, settings=None):
        if settings == None:
            ub = [np.inf, np.inf, 8, np.inf, np.inf, 3]
            lb = 0
        else:
            ub = np.inf
            lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        # Fp, Tp, Ft, Tt, Ta, S0
        pars = [
            ['S0', 'Signal scaling factor', self.pars[0], 'a.u.'],
            ['Fp', 'Plasma flow', self.pars[1], 'mL/sec/mL'],
            ['Tp', 'Plasma mean transit time', self.pars[2], 'sec'],
            ['Ft', 'Tubular flow', self.pars[3], 'mL/sec/mL'],
            ['Tt', 'Tubular mean transit time', self.pars[4], 'sec'],
            ['Ta', 'Arterial mean transit time', self.pars[5], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Fb','Blood flow',self.pars[1]/(1-self._Hct),'mL/sec/mL'],
            ['ve', 'Extracellular volume', self.pars[1]*self.pars[2], ''],
            ['FF', 'Filtration fraction', self.pars[3]/self.pars[1], ''],
            ['E', 'Extraction fraction', self.pars[3]/(self.pars[1]+self.pars[3]), ''],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*100, '%']
            pars[3][2:] = [pars[3][2]*100, '%']

        if self._vol is None:
            return pars
        
        pars += [
            ['SK-GFR', 'Single-kidney glomerular filtration rate', self.pars[3]*self._vol, 'mL/sec'],
            ['SK-RBF', 'Single-kidney renal blood flow', self.pars[1]*self._vol/(1-self._Hct), 'mL/sec'],
        ]
        if units == 'custom':
            pars[4][2:] = [pars[4][2]*60, 'mL/min']
            pars[5][2:] = [pars[5][2]*60, 'mL/min']

        return pars



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

    def __init__(self, aif, 
            pars = None,
            dt = 0.5, 
            Hct = 0.45, 
            agent = 'gadoterate',
            field_strength = 3.0,
            TR = 5.0/1000.0,  
            FA = 15.0,
            R10 = 1,  
            R10b = 1,
            t0 = 0,
            vol = None, 
        ):
        n0 = max([round(t0/dt),1])
        r1 = dc.relaxivity(field_strength, 'blood', agent)
        cb = dc.conc_ss(aif, TR, FA, 1/R10b, r1, n0)
        self._t = dt*np.arange(np.size(aif))
        self._dt = dt
        self._Hct = Hct
        self._TR = TR
        self._FA = FA
        self._R10 = R10
        self._R10b = R10b
        self._r1 = r1
        self._t0 = t0
        self._vol = vol
        self.pars = self.pars0(pars)
        self.ca = cb/(1-Hct)        #: Arterial plasma concentration (M)

    def _forward_model(self, xdata):
        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg) 
        S0, Fp, Tp, Ft, Tt, Ta = self.pars
        C = self.conc()
        R1 = self._R10 + self._r1*C
        signal = dc.signal_ss(R1, S0, self._TR, self._FA)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])

    def conc(self):
        """Tissue concentration

        Returns:
            numpy.ndarray: Concentration in M
        """
        S0, Fp, Tp, Ft, Tt, Ta = self.pars
        ca = dc.flux_plug(self.ca, Ta, t=self._t, dt=self._dt)
        return dc.kidney_conc_2cm(ca, Fp, Tp, Ft, Tt, t=self._t, dt=self._dt)
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_ss(self._R10, 1, self._TR, self._FA)
        n0 = max([np.sum(xdata<self._t0), 1])
        self.pars[0] = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)
    
    def pars0(self, settings=None):
        if settings == None:
            return np.array([1, 200/6000, 5, 30/6000, 120, 0])
        else:
            return np.zeros(6)

    def bounds(self, settings=None):
        if settings == None:
            ub = [np.inf, np.inf, 8, np.inf, np.inf, 3]
            lb = 0
        else:
            ub = np.inf
            lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        # Fp, Tp, Ft, Tt, Ta, S0
        pars = [
            ['S0', 'Signal scaling factor', self.pars[0], 'a.u.'],
            ['Fp', 'Plasma flow', self.pars[1], 'mL/sec/mL'],
            ['Tp', 'Plasma mean transit time', self.pars[2], 'sec'],
            ['Ft', 'Tubular flow', self.pars[3], 'mL/sec/mL'],
            ['Tt', 'Tubular mean transit time', self.pars[4], 'sec'],
            ['Ta', 'Arterial mean transit time', self.pars[5], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Fb','Blood flow',self.pars[1]/(1-self._Hct),'mL/sec/mL'],
            ['ve', 'Extracellular volume', self.pars[1]*self.pars[2], ''],
            ['FF', 'Filtration fraction', self.pars[3]/self.pars[1], ''],
            ['E', 'Extraction fraction', self.pars[3]/(self.pars[1]+self.pars[3]), ''],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*100, '%']
            pars[3][2:] = [pars[3][2]*100, '%']

        if self._vol is None:
            return pars
        
        pars += [
            ['SK-GFR', 'Single-kidney glomerular filtration rate', self.pars[3]*self._vol, 'mL/sec'],
            ['SK-RBF', 'Single-kidney renal blood flow', self.pars[1]*self._vol/(1-self._Hct), 'mL/sec'],
        ]
        if units == 'custom':
            pars[4][2:] = [pars[4][2]*60, 'mL/min']
            pars[5][2:] = [pars[5][2]*60, 'mL/min']

        return pars


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
    
    def __init__(self, aif, 
            pars = None,
            dt = 0.5, 
            Hct = 0.45, 
            agent = 'gadoterate',
            field_strength = 3.0,
            TR = 5.0/1000.0,  
            FA = 15.0,
            R10 = 1,  
            R10b = 1,
            t0 = 0,
            TT = [15,30,60,90,150,300,600],
            vol = None, 
        ):
        n0 = max([round(t0/dt),1])
        r1 = dc.relaxivity(field_strength, 'blood', agent)
        cb = dc.conc_ss(aif, TR, FA, 1/R10b, r1, n0)
        self._t = dt*np.arange(np.size(aif))
        self._dt = dt
        self._Hct = Hct
        self._TR = TR
        self._FA = FA
        self._R10 = R10
        self._R10b = R10b
        self._r1 = r1
        self._t0 = t0
        self._vol = vol
        self._TT = TT
        self.pars = self.pars0(pars)
        self.ca = cb/(1-Hct)        #: Arterial plasma concentration (M)
        
    def _forward_model(self, xdata):
        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg) 
        S0, Fp, Tp, Ft, h0,h1,h2,h3,h4,h5 = self.pars
        C = self.conc()
        R1 = self._R10 + self._r1*C
        signal = dc.signal_ss(R1, S0, self._TR, self._FA)
        return dc.sample(xdata, self._t, signal, xdata[1]-xdata[0])
    
    def conc(self):
        """Tissue concentration

        Returns:
            numpy.ndarray: Concentration in M
        """
        S0, Fp, Tp, Ft, h0,h1,h2,h3,h4,h5 = self.pars
        H = [h0,h1,h2,h3,h4,h5]
        return dc.kidney_conc_pf(self.ca, Fp, Tp, Ft, H, TT=self._TT, t=self._t, dt=self._dt)

    def train(self, xdata, ydata, pfix=[1]+9*[0], p0=None,
            bounds=None, xrange=None, xvalid=None, 
            **kwargs):
        """Train the free parameters of the model using the data provided.

        After training, the attribute ``pars`` will contain the updated parameter values, and ``popt`` contains the covariance matrix. The function uses the scipy function `scipy.optimize.curve_fit`, but has some additional keywords for convenience during model prototyping. 

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (array-like): Array with y-data (signal data)
            p0 (array-like, optional): Initial values for the free parameters. Defaults to None.
            bounds (str or tuple, optional): String or tuple defining the parameter bounds to be used whil training. The tuple must have 2 elements where each element is either an array with values (one for each parameter) or a single value (when each parameter has the same bound). Defaults to None.
            pfix (array-like, optional): Binary array defining which parameters should be held fixed during training (value=1) or which to fit (value=0). If not provided, all free parameters are fitted. By default the baseline signal value S0 is not fixed.
            xrange (array-like, optional): tuple of two values [xmin, xmax] showing lower and upper x-values to use for fitting. This parameters is useful to exclude particular ranges to be used in the training. If not provided, all x-values are used for training. Defaults to None.
            xvalid (array-like, optional): Binary array defining which xdata to use for training, with values of either 1 (use) or 0 (don't use). This parameter is useful to exclude individual x-values from the fit, e.g. because the data are corrupted. If not provided, all x-values are used for training. Defaults to None.
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        Sref = dc.signal_ss(self._R10, 1, self._TR, self._FA)
        n0 = max([np.sum(xdata<self._t0), 1])
        self.pars[0] = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, pfix=pfix, p0=p0,
            bounds=bounds, xrange=xrange, xvalid=xvalid, 
            **kwargs)
    
    def pars0(self, settings=None):
        if settings == None:
            return np.array([1, 0.2, 0.1, 3.0, 1, 1, 1, 1, 1, 1])
        else:
            return np.zeros(10)

    def bounds(self, settings=None):
        ub = +np.inf
        lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0', 'Signal scaling factor', self.pars[0], 'a.u.'],
            ['Fp', 'Plasma flow', self.pars[1], 'mL/sec/mL'],
            ['Tp', 'Plasma mean transit time', self.pars[2], 'sec'],
            ['Ft', 'Tubular flow', self.pars[3], 'mL/sec/mL'],
            ['h0', "Transit time weight (15-30s)", self.pars[4], '1/sec'],
            ['h1', "Transit time weight (30-60s)", self.pars[5], '1/sec'],
            ['h2', "Transit time weight (60-90s)", self.pars[6], '1/sec'],
            ['h3', "Transit time weight (90-150s)", self.pars[7], '1/sec'],
            ['h4', "Transit time weight (150-300s)", self.pars[8], '1/sec'],
            ['h5', "Transit time weight (300-600s)", self.pars[9], '1/sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
        return pars

    def pdep(self, units='standard'):
        TT = rv_histogram((self.pars[4:],self._TT), density=True)
        pars = [
            ['Fb','Blood flow',self.pars[1]/(1-self._Hct),'mL/sec/mL'],
            ['ve', 'Extracellular volume', self.pars[1]*self.pars[2], ''],
            ['FF', 'Filtration fraction', self.pars[3]/self.pars[1], ''],
            ['E', 'Extraction fraction', self.pars[3]/(self.pars[1]+self.pars[3]), ''],
            ['Tt', 'Tubular mean transit time', TT.mean(), 'sec'],
            ['Dt', 'Tubular transit time dispersion', TT.std()/TT.mean(), ''],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*100, '%']
            pars[3][2:] = [pars[3][2]*100, '%']
            pars[4][2:] = [pars[4][2]/60, 'min']
            pars[5][2:] = [pars[5][2]*100, '%']

        if self._vol is None:
            return pars
        
        pars += [
            ['SK-GFR', 'Single-kidney glomerular filtration rate', self.pars[3]*self._vol, 'mL/sec'],
            ['SK-RBF', 'Single-kidney renal blood flow', self.pars[1]*self._vol/(1-self._Hct), 'mL/sec'],
        ]
        if units == 'custom':
            pars[6][2:] = [pars[6][2]*60, 'mL/min']
            pars[7][2:] = [pars[7][2]*60, 'mL/min']
        return pars

    

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

    def __init__(self, aif, 
            pars = None,
            dt = 0.5, 
            Hct = 0.45, 
            agent = 'gadoterate',
            field_strength = 3.0,
            TR = 5.0/1000.0,  
            FA = 15.0,
            Tsat = 0,
            TC = 0.085,
            R10c = 1,
            R10m = 1,  
            R10b = 1,
            S0c = 1,
            S0m = 1,
            t0 = 0,
            vol = None, 
        ):
        n0 = max([round(t0/dt),1])
        r1 = dc.relaxivity(field_strength, 'blood', agent)
        cb = dc.conc_src(aif, TC, 1/R10b, r1, n0)
        self.pars = self.pars0(pars)
        self.ca = cb/(1-Hct)        #: Arterial plasma concentration (M)
        self._t = dt*np.arange(np.size(aif))
        self._r1 = r1
        self._dt = dt
        self._Hct = Hct
        self._TR = TR
        self._FA = FA
        self._Tsat = Tsat
        self._TC = TC
        self._R10c = R10c
        self._R10m = R10m
        self._R10b = R10b
        self._S0c = S0c
        self._S0m = S0m
        self._t0 = t0
        self._vol = vol

    def conc(self):
        """Cortical and medullary concentration

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Concentration in cortex, concentration in medulla, in M
        """
        return dc.kidney_conc_cm9(self.ca, *self.pars, dt=self._dt)
        
    def _forward_model(self, xdata):
        Cc, Cm = self.conc()
        R1c = self._R10c + self._r1*Cc
        R1m = self._R10m + self._r1*Cm
        Sc = dc.signal_sr(R1c, self.S0c, self._TR, self._FA, self._TC, self._Tsat)
        Sm = dc.signal_sr(R1m, self.S0m, self._TR, self._FA, self._TC, self._Tsat)
        nt = int(len(xdata)/2)
        Sc = dc.sample(xdata[:nt], self._t, Sc, xdata[2]-xdata[1])
        Sm = dc.sample(xdata[nt:], self._t, Sm, xdata[2]-xdata[1])
        return np.concatenate((Sc, Sm)) 

    def predict(self, time):
        """Predict the data

        Args:
            time (array-like): Time points (sec) where signal is predicted

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: signal in cortex and medulla.
        """
        xdata = np.concatenate((time, time))
        S = self._forward_model(xdata)
        nt = int(len(xdata)/2)
        return S[:nt], S[nt:]
    
    def train(self, time, Sc, Sm, **kwargs):
        """Train the model.

        Args:
            time (array-like): Time points 
            Sc (array-like): signal in the cortex (same size as time)
            Sm (array-like): signal in the medulla (same size as time)
            kwargs: any other arguments accepted by `dc.Model.train`
        """
        n0 = max([np.sum(time<self._t0), 1])
        Scref = dc.signal_sr(self._R10c, 1, self._TR, self._FA, self._TC, self._Tsat)
        Smref = dc.signal_sr(self._R10m, 1, self._TR, self._FA, self._TC, self._Tsat)
        self.S0c = np.mean(Sc[:n0]) / Scref
        self.S0m = np.mean(Sm[:n0]) / Smref
        xdata = np.concatenate((time, time))
        ydata = np.concatenate((Sc, Sm))
        return super().train(xdata, ydata, **kwargs)
    
    def pars0(self, settings=None):
        if settings == None:
            return np.array([0.03, 0.15, 0.8, 4, 10, 60, 60, 30, 30])
        else:
            return np.zeros(9)

    def bounds(self, settings=None):
        if settings == None:
            ub = [1, 1, 1, 10, 30, np.inf, np.inf, np.inf, np.inf]
            lb = [0.01, 0, 0, 0, 0, 0, 0, 0, 0] 
        else:
            ub = [np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            lb = 0
        return (lb, ub)
    
    def pfree(self, units='standard'):
        pars = [
            ['Fp','Plasma flow',self.pars[0],'mL/sec/mL'], 
            ['Eg','Glomerular extraction fraction',self.pars[1],''], 
            ['fc','Cortical flow fraction',self.pars[2],''], 
            ['Tg','Glomerular mean transit time',self.pars[3],'sec'], 
            ['Tv','Peritubular & venous mean transit time',self.pars[4],'sec'], 
            ['Tpt','Proximal tubuli mean transit time',self.pars[5],'sec'], 
            ['Tlh','Lis of Henle mean transit time',self.pars[6],'sec'], 
            ['Tdt','Distal tubuli mean transit time',self.pars[7],'sec'], 
            ['Tcd','Collecting duct mean transit time',self.pars[8],'sec'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*100, '%']
            pars[5][2:] = [pars[5][2]/60, 'min']
            pars[6][2:] = [pars[6][2]/60, 'min']
            pars[7][2:] = [pars[7][2]/60, 'min']
            pars[8][2:] = [pars[8][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        pars = [
            ['FF', 'Filtration fraction', p[1]/(1-p[1]), ''],
            ['Ft', 'Tubular flow', p[1]/(1-p[1])*p[0], 'mL/sec/mL'],
            ['CBF', 'Cortical blood flow', p[0]/(1-self._Hct), 'mL/sec/mL'],
            ['MBF','Medullary blood flow', p[2]*(1-p[1])*p[0]/(1-self._Hct), 'mL/sec/mL'],
        ]
        if units=='custom':
            pars[0][2:] = [pars[0][2]*100, '%']
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']

        if self._vol is None:   
            return pars
        
        pars += [
            ['SKGFR','Single-kidney glomerular filtration rate', p[1]/(1-p[1])*p[0]*self._vol,'mL/sec'],
            ['SKBF', 'Single-kidney blood flow', self._vol*p[0]/(1-self._Hct), 'mL/sec'],
            ['SKMBF', 'Single-kidney medullary blood flow', self._vol*p[2]*(1-p[1])*p[0]/(1-self._Hct), 'mL/sec'],
        ] 
        if units=='custom':
            pars[4][2:] = [pars[4][2]*60, 'mL/min']
            pars[5][2:] = [pars[5][2]*60, 'mL/min']
            pars[6][2:] = [pars[6][2]*60, 'mL/min']

        return pars
