import numpy as np
import dcmri as dc



class UptSS(dc.Model):
    """One-compartment uptake tissue, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Plasma flow into the compartment per unit tissue.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `OneCompSS`, `PatlakSS`, `ToftsSS`, `EToftsSS`, `TwoCompUptWXSS`, `TwoCompExchSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.UptSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 229.188 (8.902) a.u.
        Plasma flow (Fp): 0.001 (0.0) 1/sec
    """         
    dt = 0.5                #: Sampling interval of the AIF in sec. 
    Hct = 0.45              #: Hematocrit. 
    agent = 'gadoterate'    #: Contrast agent generic name.
    field_strength = 3.0    #: Magnetic field strength in T. 
    TR = 5.0/1000.0         #: Repetition time, or time between excitation pulses, in sec. 
    FA = 15.0               #: Nominal flip angle in degrees.
    R10 = 1                 #: Precontrast tissue relaxation rate in 1/sec. 
    R10b = 1                #: Precontrast arterial relaxation rate in 1/sec. 
    t0 = 1                  #: Baseline length (sec).

    def __init__(self, aif, pars='default', **attr):
        super().__init__(pars, **attr)

        # Calculate constants
        n0 = max([round(self.t0/self.dt),1])
        self._r1 = dc.relaxivity(self.field_strength, 'blood', self.agent)
        cb = dc.conc_ss(aif, self.TR, self.FA, 1/self.R10b, self._r1, n0)
        self._ca = cb/(1-self.Hct)
        self._t = self.dt*np.arange(np.size(aif))

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp = self.pars
        C = dc.conc_1cum(self._ca, Fp, dt=self.dt)
        if return_conc:
            return dc.sample(xdata, self._t, C, xdata[2]-xdata[1])
        R1 = self.R10 + self._r1*C
        ydata = dc.signal_ss(R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def train(self, xdata, ydata, **kwargs):
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata<self.t0), 1])
        self.pars[0] = np.mean(ydata[:n0]) / Sref
        return super().train(xdata, ydata, **kwargs)
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.01])
        else:
            return np.zeros(2)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf]
        lb = [0, 0]
        return (lb, ub)
    
    def pfree(self, units='standard'):
        # vp, Ktrans, ve
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'1/sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
        return pars
    
    def aif_conc(self):
        """Reconstructed plasma concentrations in the arterial input.

        Returns:
            np.ndarray: Concentrations in M.
        """
        return self._ca
    

# Fast/no water exchange
    

class OneCompSS(UptSS):
    """One-compartment tissue, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Plasma flow into the compartment per unit tissue.
    - **v** (Volume of distribution, mL/mL): Volume fraction of the compartment. 
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `UptSS`, `PatlakSS`, `ToftsSS`, `EToftsSS`, `TwoCompUptWXSS`, `TwoCompExchSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.OneCompSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)       
        -----------------------------------------       
        Signal scaling factor (S0): 175.126 (6.654) a.u.
        Plasma flow (Fp): 0.004 (0.0) 1/sec
        Volume of distribution (v): 0.004 (0.015) mL/mL 
    """ 

    water_exchange = True       #: Assume fast water exchange (True) or no water exchange (False). Default is True.

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp, v = self.pars
        C = dc.conc_1cm(self._ca, Fp, v, dt=self.dt)
        if return_conc:
            return dc.sample(xdata, self._t, C, xdata[2]-xdata[1])
        if self.water_exchange:
            R1 = self.R10 + self._r1*C
            ydata = dc.signal_ss(R1, S0, self.TR, self.FA)
        else:
            R1e = self.R10 + self._r1*C/v
            R1c = self.R10 + np.zeros(C.size)
            v = [v, 1-v]
            R1 = np.stack((R1e, R1c))
            ydata = dc.signal_ss_nex(v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.01, 0.05])
        else:
            return np.zeros(3)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf, 1]
        lb = [0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        # vp, Ktrans, ve
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'1/sec'],
            ['v','Volume of distribution',self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            
        return pars


class ToftsSS(OneCompSS):
    """Tofts tissue, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Ktrans** (Vascular transfer constant, mL/sec/mL): clearance of the plasma compartment per unit tissue.
    - **ve** (Extravascular, extracellular volume, mL/mL): Volume fraction of the interstitial compartment.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `UptSS`, `OneCompSS`, `PatlakSS`, `EToftsSS`, `TwoCompUptWXSS`, `TwoCompExchSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.ToftsSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 175.126 (6.654) a.u.
        Volume transfer constant (Ktrans): 0.004 (0.0) 1/sec
        Extravascular, extracellular volume (ve): 0.156 (0.015) mL/mL
        ------------------
        Derived parameters
        ------------------
        Extracellular mean transit time (Te): 39.619 sec
        Extravascular transfer constant (kep): 0.025 1/sec
    """         

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Ktrans','Volume transfer constant',self.pars[1],'1/sec'],
            ['ve','Extravascular, extracellular volume',self.pars[2],'mL/mL'],
            
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Te','Extracellular mean transit time',self.pars[2]/self.pars[1],'sec'],
            ['kep','Extravascular transfer constant',self.pars[1]/self.pars[2],'1/sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]/60, 'min']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
        return pars

    
class PatlakSS(UptSS):
    """Patlak tissue in fast water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **vp** (Plasma volume, mL/mL): Volume fraction of the plasma compartment. 
    - **Ktrans** (Vascular transfer constant, mL/sec/mL): clearance of the plasma compartment per unit tissue.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `UptSS`, `OneCompSS`, `ToftsSS`, `EToftsSS`, `TwoCompUptWXSS`, `TwoCompExchSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.PatlakSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 174.415 (5.95) a.u.
        Plasma volume (vp): 0.049 (0.004) mL/mL
        Volume transfer constant (Ktrans): 0.001 (0.0) 1/sec
    """ 

    water_exchange = True       #: Assume fast water exchange (True) or no water exchange (False). Default is True.        

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, vp, Ktrans = self.pars
        C = dc.conc_patlak(self._ca, vp, Ktrans, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        if self.water_exchange:
            R1 = self.R10 + self._r1*(C[0,:]+C[1,:])
            ydata = dc.signal_ss(R1, S0, self.TR, self.FA)
        else:
            vb = vp/(1-self.Hct)
            R1b = self.R10b + self._r1*C[0,:]/vb
            R1e = self.R10 + self._r1*C[1,:]/(1-vb)
            v = [vb, 1-vb]
            R1 = np.stack((R1b, R1e))
            ydata = dc.signal_ss_nex(v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.05, 0.003])
        else:
            return np.zeros(3)

    def bounds(self, settings='default'):
        ub = [np.inf, 1, np.inf]
        lb = [0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['vp','Plasma volume',self.pars[1],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[2],'1/sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
        return pars


class EToftsSS(UptSS):
    """Extended Tofts tissue in fast water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    Probably the most common modelling approach for generic tissues. The arterial concentrations are calculated by direct analytical inversion of the arterial signal 

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **vp** (Plasma volume, mL/mL): Volume fraction of the plasma compartment. 
    - **Ktrans** (Vascular transfer constant, mL/sec/mL): clearance of the plasma compartment per unit tissue.
    - **ve** (Extravascular, extracellular volume, mL/mL): Volume fraction of the interstitial compartment.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `UptSS`, `OneCompSS`, `PatlakSS`, `ToftsSS`, `TwoCompUptWXSS`, `TwoCompExchSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.EToftsSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1.0,
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 148.41 (0.79) a.u.
        Plasma volume (vp): 0.047 (0.001) mL/mL
        Volume transfer constant (Ktrans): 0.003 (0.0) 1/sec
        Extravascular extracellular volume (ve): 0.207 (0.003) mL/mL
        ------------------
        Derived parameters
        ------------------
        Extracellular mean transit time (Te): 65.935 sec
        Extravascular transfer constant (kep): 0.015 1/sec
        Extracellular volume (v): 0.254 mL/mL
    """  

    water_exchange = True       #: Assume fast water exchange (True) or no water exchange (False). Default is True.       

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, vp, Ktrans, ve = self.pars
        C = dc.conc_etofts(self._ca, vp, Ktrans, ve, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        if self.water_exchange:
            R1 = self.R10 + self._r1*(C[0,:]+C[1,:])
            ydata = dc.signal_ss(R1, S0, self.TR, self.FA)
        else:
            vb = vp/(1-self.Hct)
            R1b = self.R10b + self._r1*C[0,:]/vb
            R1e = self.R10 + self._r1*C[1,:]/ve
            R1c = self.R10 + np.zeros(C.shape[1])
            v = [vb, ve, 1-vb-ve]
            R1 = np.stack((R1b, R1e, R1c))
            ydata = dc.signal_ss_nex(v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.05, 0.003, 0.3])
        else:
            return np.zeros(4)

    def bounds(self, settings='default'):
        ub = [np.inf, 1, np.inf, 1]
        lb = [0, 0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['vp','Plasma volume',self.pars[1],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[2],'1/sec'],
            ['ve','Extravascular extracellular volume',self.pars[3],'mL/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Te','Extracellular mean transit time',self.pars[3]/self.pars[2],'sec'],
            ['kep','Extravascular transfer constant',self.pars[2]/self.pars[3],'1/sec'],
            ['v','Extracellular volume',self.pars[1]+self.pars[3],'mL/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]/60, 'min']
            pars[2][2:] = [pars[2][2]*60, '1/min']
            pars[3][2:] = [pars[3][2]*100, 'mL/100mL']
        return pars
    

class TwoCompUptSS(UptSS):
    """Two-compartment uptake model (2CUM) in fast water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
    - **vp** (Plasma volume, mL/mL): Volume fraction of the plasma compartment. 
    - **PS** (Permeability-surface area product, mL/sec/mL): volume of plasma cleared of indicator per unit time and per unit tissue.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `UptSS`, `OneCompSS`, `PatlakSS`, `ToftsSS`, `EToftsSS`, `TwoCompExchSS`

    Example:

        Derive 2CUM model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.TwoCompUptWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 171.71 (5.044) a.u.
        Plasma flow (Fp): 0.021 (0.003) mL/sec/mL
        Plasma volume (vp): 0.064 (0.005) mL/mL
        Permeability-surface area product (PS): 0.001 (0.0) mL/sec/mL
        ------------------
        Derived parameters
        ------------------
        Extraction fraction (E): 0.037 sec
        Volume transfer constant (Ktrans): 0.001 mL/sec/mL
        Plasma mean transit time (Tp): 2.88 sec
    """         

    water_exchange = True       #: Assume fast water exchange (True) or no water exchange (False). Default is True.

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp, vp, PS = self.pars
        C = dc.conc_2cum(self._ca, Fp, vp, PS, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        if self.water_exchange:
            R1 = self.R10 + self._r1*(C[0,:]+C[1,:])
            ydata = dc.signal_ss(R1, S0, self.TR, self.FA)
        else:
            vb = vp/(1-self.Hct)
            R1b = self.R10b + self._r1*C[0,:]/vb
            R1e = self.R10 + self._r1*C[1,:]/(1-vb)
            v = [vb, 1-vb]
            R1 = np.stack((R1b, R1e))
            ydata = dc.signal_ss_nex(v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.1, 0.1, 0.003])
        else:
            return np.zeros(4)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf, 1, np.inf]
        lb = [0, 0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'mL/sec/mL'],
            ['vp','Plasma volume',self.pars[2],'mL/mL'],
            ['PS','Permeability-surface area product',self.pars[3],'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        pars = [
            ['E','Extraction fraction',p[3]/(p[1]+p[3]),'sec'],
            ['Ktrans','Volume transfer constant',p[1]*p[3]/(p[1]+p[3]),'mL/sec/mL'],
            ['Tp','Plasma mean transit time',p[2]/(p[1]+p[3]),'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
        return pars


class TwoCompExchSS(UptSS):
    """Two-compartment exchange model (2CXM) in fast water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
    - **vp** (Plasma volume, mL/mL): Volume fraction of the plasma compartment. 
    - **PS** (Permeability-surface area product, mL/sec/mL): volume of plasma cleared of indicator per unit time and per unit tissue.
    - **ve** (Extravascular, extracellular volume, mL/mL): Volume fraction of the interstitial compartment.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `UptSS`, `OneCompSS`, `PatlakSS`, `ToftsSS`, `EToftsSS`, `TwoCompUptWXSS`

    Example:

        Derive 2CXM model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.TwoCompExchSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 149.772 (0.073) a.u.
        Plasma flow (Fp): 0.098 (0.001) mL/sec/mL
        Plasma volume (vp): 0.051 (0.0) mL/mL
        Permeability-surface area product (PS): 0.003 (0.0) mL/sec/mL
        Extravascular extracellular volume (ve): 0.2 (0.0) mL/mL
        ------------------
        Derived parameters
        ------------------
        Extraction fraction (E): 0.03 sec
        Volume transfer constant (Ktrans): 0.003 mL/sec/mL
        Plasma mean transit time (Tp): 0.5 sec
        Extracellular mean transit time (Te): 66.718 sec
        Extracellular volume (v): 0.251 mL/mL
        Mean transit time (T): 0.002 sec
    """         

    water_exchange = True       #: Assume fast water exchange (True) or no water exchange (False). Default is True.

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp, vp, PS, ve = self.pars
        C = dc.conc_2cxm(self._ca, Fp, vp, PS, ve, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        if self.water_exchange:
            R1 = self.R10 + self._r1*(C[0,:]+C[1,:])
            ydata = dc.signal_ss(R1, S0, self.TR, self.FA)
        else:
            vb = vp/(1-self.Hct)
            R1b = self.R10b + self._r1*C[0,:]/vb
            R1e = self.R10 + self._r1*C[1,:]/ve
            R1c = self.R10 + np.zeros(C.shape[1])
            v = [vb, ve, 1-vb-ve]
            R1 = np.stack((R1b, R1e, R1c))
            ydata = dc.signal_ss_nex(v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.1, 0.1, 0.003, 0.3])
        else:
            return np.zeros(5)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf, 1, np.inf, 1]
        lb = [0, 0, 0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'mL/sec/mL'],
            ['vp','Plasma volume',self.pars[2],'mL/mL'],
            ['PS','Permeability-surface area product',self.pars[3],'mL/sec/mL'],
            ['ve','Extravascular extracellular volume',self.pars[4],'mL/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        pars = [
            ['E','Extraction fraction',p[3]/(p[1]+p[3]),'sec'],
            ['Ktrans','Volume transfer constant',p[1]*p[3]/(p[1]+p[3]),'mL/sec/mL'],
            ['Tp','Plasma mean transit time',p[2]/(p[1]+p[3]),'sec'],
            ['Te','Extracellular mean transit time',p[4]/p[3],'sec'],
            ['v','Extracellular volume',p[2]+p[4],'mL/mL'],
            ['T','Mean transit time',(p[2]+p[4])/p[0],'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]/60, 'min']
            pars[5][2:] = [pars[5][2]*100, 'mL/100mL']
        return pars


# Intermediate water exchange


class OneCompWXSS(UptSS):
    """One-compartment tissue in intermediate water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Plasma flow into the compartment per unit tissue.
    - **v** (Volume of distribution, mL/mL): Volume fraction of the compartment. 
    - **PSw** (Water permeability-surface area product, mL/sec/mL): PS for water across the compartment wall.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `OneCompSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.OneCompWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 172.679 (6.628) a.u.
        Plasma flow (Fp): 0.006 (0.002) 1/sec
        Volume of distribution (v): 0.195 (0.049) mL/mL
        Water permeability-surface area product (PSw): 0.0 (2.346) mL/mL
        ------------------
        Derived parameters
        ------------------
        Extracompartmental water mean transit time (Twe): 9.412177217600262e+32 sec
        Intracompartmental water mean transit time (Twb): 2.2783889577230225e+32 sec
    """         
    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp, v, PSw = self.pars
        C = dc.conc_1cm(self._ca, Fp, v, dt=self.dt)
        if return_conc:
            return dc.sample(xdata, self._t, C, xdata[2]-xdata[1])
        R1e = self.R10 + self._r1*C/v
        R1c = self.R10 + np.zeros(C.size)
        PS = np.array([[0,PSw],[PSw,0]])
        v = [v, 1-v]
        R1 = np.stack((R1e, R1c))
        ydata = dc.signal_ss_iex(PS, v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.01, 0.05, 10])
        else:
            return np.zeros(4)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf, 1, np.inf]
        lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'1/sec'],
            ['v','Volume of distribution',self.pars[2],'mL/mL'],
            ['PSw','Water permeability-surface area product',self.pars[3],'mL/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        pars = [
            ['Twe', 'Extracompartmental water mean transit time', (1-p[2])/p[3], 'sec'],
            ['Twb', 'Intracompartmental water mean transit time', p[2]/p[3], 'sec'],
        ]
        return pars


class ToftsWXSS(OneCompWXSS):
    """Tofts tissue with intermediate water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **Ktrans** (Vascular transfer constant, mL/sec/mL): clearance of the plasma compartment per unit tissue.
    - **ve** (Extravascular, extracellular volume, mL/mL): Volume fraction of the interstitial compartment.
    - **PSe** (Transendothelial water permeability-surface area product, mL/sec/mL): Permeability of water across the endothelium.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `ToftsWXSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.ToftsWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 172.679 (6.628) a.u.
        Volume transfer constant (Ktrans): 0.006 (0.002) 1/sec
        Extravascular, extracellular volume (ve): 0.195 (0.049) mL/mL
        Transendothelial water PS (PSe): 0.0 (2.346) mL/sec/mL
        ------------------
        Derived parameters
        ------------------
        Extracellular mean transit time (Te): 33.025 sec
        Extravascular transfer constant (kep): 0.03 1/sec
        Extravascular water mean transit time (Twe): 2.2783889577230225e+32 sec
        Intravascular water mean transit time (Twb): 9.412177217600262e+32 sec
    """         

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Ktrans','Volume transfer constant',self.pars[1],'1/sec'],
            ['ve','Extravascular, extracellular volume',self.pars[2],'mL/mL'],
            ['PSe','Transendothelial water PS',self.pars[3],'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Te','Extracellular mean transit time',self.pars[2]/self.pars[1],'sec'],
            ['kep','Extravascular transfer constant',self.pars[1]/self.pars[2],'1/sec'],
            ['Twe', 'Extravascular water mean transit time', self.pars[2]/self.pars[3], 'sec'],
            ['Twb', 'Intravascular water mean transit time', (1-self.pars[2])/self.pars[3], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]/60, 'min']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
        return pars


class PatlakWXSS(UptSS):
    """Patlak tissue with intermediate exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **vp** (Plasma volume, mL/mL): Volume fraction of the plasma compartment. 
    - **Ktrans** (Vascular transfer constant, mL/sec/mL): clearance of the plasma compartment per unit tissue.
    - **PSe** (Transendothelial water permeability-surface area product, mL/sec/mL): PS for water across the endothelium.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `PatlakSS`.

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.PatlakWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 165.529 (5.377) a.u.
        Plasma volume (vp): 0.091 (0.014) mL/mL
        Volume transfer constant (Ktrans): 0.001 (0.0) 1/sec
        Transendothelial water PS (PSe): 0.0 (0.666) mL/sec/mL
        ------------------
        Derived parameters
        ------------------
        Extravascular water mean transit time (Twe): 4803607269928.271 sec
        Intravascular water mean transit time (Twb): 950667689299.263 sec
    """         
    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, vp, Ktrans, PSe = self.pars
        C = dc.conc_patlak(self._ca, vp, Ktrans, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        vb = vp/(1-self.Hct)
        R1b = self.R10b + self._r1*C[0,:]/vb
        R1e = self.R10 + self._r1*C[1,:]/(1-vb)
        PS = np.array([[0,PSe],[PSe,0]])
        v = [vb, 1-vb]
        R1 = np.stack((R1b, R1e))
        ydata = dc.signal_ss_iex(PS, v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.05, 0.003, 10])
        else:
            return np.zeros(4)

    def bounds(self, settings='default'):
        ub = [np.inf, 1, np.inf, np.inf]
        lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['vp','Plasma volume',self.pars[1],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[2],'1/sec'],
            ['PSe','Transendothelial water PS',self.pars[3],'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        vb = p[1]/(1-self.Hct)
        pars = [
            ['Twe', 'Extravascular water mean transit time', (1-vb)/p[3], 'sec'],
            ['Twb', 'Intravascular water mean transit time', vb/p[3], 'sec'],
        ]
        return pars


class EToftsWXSS(UptSS):
    """Extended Tofts tissue with intermediate exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    Probably the most common modelling approach for generic tissues. The arterial concentrations are calculated by direct analytical inversion of the arterial signal 

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): scale factor for the MR signal.
    - **vp** (Plasma volume, mL/mL): Volume fraction of the plasma compartment. 
    - **Ktrans** (Vascular transfer constant, mL/sec/mL): clearance of the plasma compartment per unit tissue.
    - **ve** (Extravascular, extracellular volume, mL/mL): Volume fraction of the interstitial compartment.
    - **PSe** (Transendothelial water permeability-surface area product, mL/sec/mL): PS for water across the endothelium.
    - **PSc** (Transcytolemmal water permeability-surface area product, mL/sec/mL): PS for water across the cell wall.

    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `EToftsSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.EToftsWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1.0,
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 149.985 (0.791) a.u.
        Plasma volume (vp): 0.046 (0.001) mL/mL
        Volume transfer constant (Ktrans): 0.003 (0.0) 1/sec
        Extravascular extracellular volume (ve): 0.205 (0.003) mL/mL
        Transendothelial water PS (PSe): 702.341 (3626.087) mL/sec/mL
        Transcytolemmal water PS (PSc): 1034.743 (5701.33) mL/sec/mL
        ------------------
        Derived parameters
        ------------------
        Extracellular mean transit time (Te): 65.948 sec
        Extravascular transfer constant (kep): 0.015 1/sec
        Extracellular volume (v): 0.251 mL/mL
        Intracellular water mean transit time (Twc): 0.001 sec
        Interstitial water mean transit time (Twi): 0.0 sec
        Intravascular water mean transit time (Twb): 0.0 sec
    """         

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:

        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, vp, Ktrans, ve, PSe, PSc = self.pars
        C = dc.conc_etofts(self._ca, vp, Ktrans, ve, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        vb = vp/(1-self.Hct)
        R1b = self.R10b + self._r1*C[0,:]/vb
        R1e = self.R10 + self._r1*C[1,:]/ve
        R1c = self.R10 + np.zeros(C.shape[1])
        PS = np.array([[0,PSe,0],[PSe,0,PSc],[0,PSc,0]])
        v = [vb, ve, 1-vb-ve]
        R1 = np.stack((R1b, R1e, R1c))
        ydata = dc.signal_ss_iex(PS, v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.05, 0.003, 0.3, 10, 10])
        else:
            return np.zeros(6)

    def bounds(self, settings='default'):
        ub = [np.inf, 1, np.inf, 1, np.inf, np.inf]
        lb = 0
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['vp','Plasma volume',self.pars[1],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[2],'1/sec'],
            ['ve','Extravascular extracellular volume',self.pars[3],'mL/mL'],
            ['PSe','Transendothelial water PS',self.pars[4],'mL/sec/mL'],
            ['PSc','Transcytolemmal water PS',self.pars[5],'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        vb = p[2]/(1-self.Hct)
        vc = 1-vb-p[3]
        pars = [
            ['Te','Extracellular mean transit time',self.pars[3]/self.pars[2],'sec'],
            ['kep','Extravascular transfer constant',self.pars[2]/self.pars[3],'1/sec'],
            ['v','Extracellular volume',self.pars[1]+self.pars[3],'mL/mL'],
            ['Twc', 'Intracellular water mean transit time', vc/p[5], 'sec'],
            ['Twi', 'Interstitial water mean transit time', p[3]/(p[4]+p[5]), 'sec'],
            ['Twb', 'Intravascular water mean transit time', vb/p[4], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]/60, 'min']
            pars[2][2:] = [pars[2][2]*60, '1/min']
            pars[3][2:] = [pars[3][2]*100, 'mL/100mL']
        return pars


class TwoCompUptWXSS(UptSS):
    """Two-compartment uptake model (2CUM) in intermediate water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): Scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
    - **vp** (Plasma volume): Volume fraction of the plasma compartment. 
    - **PS** (Permeability-surface area product, mL/sec/mL): Volume of plasma cleared of indicator per unit time and per unit tissue.
    - **PSe** (Transendothelial water permeability-surface area product, mL/sec/mL): PS for water across the endothelium.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `TwoCompUptWXSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.TwoCompUptWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 166.214 (4.318) a.u.
        Plasma flow (Fp): 0.044 (0.009) mL/sec/mL
        Plasma volume (vp): 0.1 (0.012) mL/mL
        Permeability-surface area product (PS): 0.001 (0.0) mL/sec/mL
        Transendothelial water PS (PSe): 0.0 (0.639) mL/sec/mL
        ------------------
        Derived parameters
        ------------------
        Extraction fraction (E): 0.016 sec
        Volume transfer constant (Ktrans): 0.001 mL/sec/mL
        Plasma mean transit time (Tp): 2.249 sec
        Extravascular water mean transit time (Twe): 3.934327679681704e+32 sec
        Intravascular water mean transit time (Twb): 8.696123529149469e+31 sec

        **Note**: The model does not fit the data well because the no-washout assumption is invalid in this example.
    """         

    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:
        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp, vp, PS, PSe = self.pars
        vb = vp/(1-self.Hct)
        C = dc.conc_2cum(self._ca, Fp, vp, PS, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        R1b = self.R10b + self._r1*C[0,:]/vb
        R1e = self.R10 + self._r1*C[1,:]/(1-vb)
        PS = np.array([[0,PSe],[PSe,0]])
        v = [vb, 1-vb]
        R1 = np.stack((R1b, R1e))        
        ydata = dc.signal_ss_iex(PS, v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.1, 0.1, 0.003, 10])
        else:
            return np.zeros(5)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf, 1, np.inf, np.inf]
        lb = [0, 0, 0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'mL/sec/mL'],
            ['vp','Plasma volume',self.pars[2],'mL/mL'],
            ['PS','Permeability-surface area product',self.pars[3],'mL/sec/mL'],
            ['PSe','Transendothelial water PS',self.pars[4],'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        vb = p[2]/(1-self.Hct)
        ve = 1-vb
        pars = [
            ['E','Extraction fraction',p[3]/(p[1]+p[3]),'sec'],
            ['Ktrans','Volume transfer constant',p[1]*p[3]/(p[1]+p[3]),'mL/sec/mL'],
            ['Tp','Plasma mean transit time',p[2]/(p[1]+p[3]),'sec'],
            ['Twe', 'Extravascular water mean transit time', ve/p[4], 'sec'],
            ['Twb', 'Intravascular water mean transit time', vb/p[4], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
        return pars


class TwoCompExchWXSS(UptSS):
    """Two-compartment exchange model (2CXM) in intermediate water exchange, acquired with a spoiled gradient echo sequence in steady state and using a direct inversion of the AIF.

    The free model parameters are:

    - **S0** (Signal scaling factor, a.u.): Scale factor for the MR signal.
    - **Fp** (Plasma flow, mL/sec/mL): Flow of plasma into the plasma compartment.
    - **vp** (Plasma volume): Volume fraction of the plasma compartment. 
    - **PS** (Permeability-surface area product, mL/sec/mL): Volume of plasma cleared of indicator per unit time and per unit tissue.
    - **ve** (Extravascular, extracellular volume): Volume fraction of the interstitial compartment.
    - **PSe** (Transendothelial water permeability-surface area product, mL/sec/mL): PS for water across the endothelium.
    - **PSc** (Transcytolemmal water permeability-surface area product, mL/sec/mL): PS for water across the cell wall.
    
    Args:
        aif (array-like): MRI signals measured in the arterial input.
        pars (str or array-like, optional): Either explicit array of values, or string specifying a predefined array (see the pars0 method for possible values). 
        attr: provide values for any attributes as keyword arguments. 

    See Also:
        `TwoCompExchSS`

    Example:

        Derive model parameters from data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `make_tissue_2cm_ss` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.make_tissue_2cm_ss(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.TwoCompExchWXSS(aif,
        ...     dt = time[1],
        ...     Hct = 0.45, 
        ...     agent = 'gadodiamide',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 20,
        ...     R10 = 1/dc.T1(3.0,'muscle'),
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
        >>> ax1.plot(time/60, 1000*model.predict(time, return_conc=True), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue prediction')
        >>> ax1.plot(gt['t']/60, 1000*gt['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        >>> ax1.plot(time/60, 1000*model.aif_conc(), linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Concentration (mM)')
        >>> ax1.legend()
        >>> #
        >>> plt.show()

        We can also have a look at the model parameters after training:

        >>> model.print(round_to=3)
        -----------------------------------------
        Free parameters with their errors (stdev)
        -----------------------------------------
        Signal scaling factor (S0): 151.51 (0.072) a.u.
        Plasma flow (Fp): 0.097 (0.001) mL/sec/mL
        Plasma volume (vp): 0.05 (0.0) mL/mL
        Permeability-surface area product (PS): 0.003 (0.0) mL/sec/mL
        Extravascular extracellular volume (ve): 0.198 (0.0) mL/mL
        Transendothelial water PS (PSe): 1488.691 (352.238) mL/sec/mL
        Transcytolemmal water PS (PSc): 1964.363 (537.865) mL/sec/mL
        ------------------
        Derived parameters
        ------------------
        Extraction fraction (E): 0.03 sec
        Volume transfer constant (Ktrans): 0.003 mL/sec/mL
        Plasma mean transit time (Tp): 0.499 sec
        Extracellular mean transit time (Te): 66.725 sec
        Extracellular volume (v): 0.248 mL/mL
        Mean transit time (T): 2.547 sec
        Intracellular water mean transit time (Twc): 0.0 sec
        Interstitial water mean transit time (Twi): 0.0 sec
        Intravascular water mean transit time (Twb): 0.0 sec

        **Note**: fitted water PS water values are high because the simulated data are in fast water exchange.

    """         
    def predict(self, xdata:np.ndarray, return_conc=False)->np.ndarray:
        if np.amax(self._t) < np.amax(xdata):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce xdata.'
            raise ValueError(msg)
        S0, Fp, vp, PS, ve, PSe, PSc = self.pars
        vb = vp/(1-self.Hct)
        C = dc.conc_2cxm(self._ca, Fp, vp, PS, ve, dt=self.dt, sum=False)
        if return_conc:
            return dc.sample(xdata, self._t, C[0,:]+C[1,:], xdata[2]-xdata[1])
        R1b = self.R10b + self._r1*C[0,:]/vb
        R1e = self.R10 + self._r1*C[1,:]/ve
        R1c = self.R10 + np.zeros(C.shape[1])
        PS = np.array([[0,PSe,0],[PSe,0,PSc],[0,PSc,0]])
        v = [vb, ve, 1-vb-ve]
        R1 = np.stack((R1b, R1e, R1c))        
        ydata = dc.signal_ss_iex(PS, v, R1, S0, self.TR, self.FA)
        return dc.sample(xdata, self._t, ydata, xdata[2]-xdata[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([1, 0.1, 0.1, 0.003, 0.3, 10, 10])
        else:
            return np.zeros(7)

    def bounds(self, settings='default'):
        ub = [np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf]
        lb = [0, 0, 0, 0, 0, 0, 0]
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['S0','Signal scaling factor',self.pars[0],'a.u.'],
            ['Fp','Plasma flow',self.pars[1],'mL/sec/mL'],
            ['vp','Plasma volume',self.pars[2],'mL/mL'],
            ['PS','Permeability-surface area product',self.pars[3],'mL/sec/mL'],
            ['ve','Extravascular extracellular volume',self.pars[4],'mL/mL'],
            ['PSe','Transendothelial water PS',self.pars[5],'mL/sec/mL'],
            ['PSc','Transcytolemmal water PS',self.pars[6],'mL/sec/mL'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        vb = p[2]/(1-self.Hct)
        vc = 1-vb-p[4]
        pars = [
            ['E','Extraction fraction',p[3]/(p[1]+p[3]),'sec'],
            ['Ktrans','Volume transfer constant',p[1]*p[3]/(p[1]+p[3]),'mL/sec/mL'],
            ['Tp','Plasma mean transit time',p[2]/(p[1]+p[3]),'sec'],
            ['Te','Extracellular mean transit time',p[4]/p[3],'sec'],
            ['v','Extracellular volume',p[2]+p[4],'mL/mL'],
            ['T','Mean transit time',(p[2]+p[4])/p[1],'sec'],
            ['Twc', 'Intracellular water mean transit time', vc/p[6], 'sec'],
            ['Twi', 'Interstitial water mean transit time', p[4]/(p[5]+p[6]), 'sec'],
            ['Twb', 'Intravascular water mean transit time', vb/p[5], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]/60, 'min']
            pars[5][2:] = [pars[5][2]*100, 'mL/100mL']
        return pars

