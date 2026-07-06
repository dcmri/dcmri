"""Signal after readout of given Mz.

Args:
    S0 (float): Signal scaling factor (arbitrary units).
    R2 (array-like): Transverse relaxation rate R1 or R2* in 1/sec. 
    FAR (float): Readout flip angle (deg)
    TE (float): Echo time (sec)
    noise_sdev (float, optional): standard deviation of the signal noise. 

Returns:
    np.ndarray: Signal in the same units as S0 and with the same 
    dimensions as Mz.
"""  

"""Longitudinalitudinal magnetization.

See section :ref:`basics-relaxation-T1` for more detail.

Args:
    R1 (array-like): Longitudinal relaxation rates in 1/sec. For a tissue 
        with n compartments, the first dimension of R1 must be n. For a
        single compartment, R1 can be scalar or a 1D time-array.
    T (float): duration of free recovery.
    v (array-like, optional): volume fractions of the compartments. For a 
        one-compartment tissue this is a scalar - otherwise it is an 
        array with one value for each compartment. Defaults to 1.
    Fw (array-like, optional): Water flow between the compartments and to 
        the environment, in units of mL/sec/cm3. Generally Fw must be a nxn 
        array, where n is the number of compartments, and the off-diagonal 
        elements Fw[j,i] are the permeability for water moving from 
        compartment i into j. The diagonal elements Fw[i,i] quantify the 
        flow of water from compartment i to outside. For a closed system 
        with equal permeabilities between all compartments, a scalar value 
        for Fw can be provided. Defaults to 0.
    j (array-like, optional): normalized tissue magnetization flux. j has 
        to have the same shape as R1. Defaults to None.
    n_init (array-like, optional): initial relative magnetization at T=0. 
        If this is a scalar, all compartments are assumed to have the same 
        initial magnetization. Defaults to 0.
    me (array-like, optional): equilibrium magnetization of the tissue 
        compartments. If a scalar value is provided, all compartments are 
        assumed to have the same equilibrium magnetization. Defaults to 1.

Returns:
    np.ndarray: Magnetization in the compartments after a time T.

Example:

    Magnetization recovery after inversion.

.. plot::
    :include-source:
    :context: close-figs

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dcmri as dc

    Plot magnetization recovery for the first 10 seconds after an 
    inversion pulse, for a closed tissue with R1 = 1 sec, and for an open 
    tissue with equilibrium inflow and inverted inflow:

    >>> TI = 0.1*np.arange(100)
    >>> R1 = 1
    >>> f = 0.5

    >>> Mz = dc._Mz_free(R1, TI, n_init=-1)
    >>> Mz_e = dc._Mz_free(R1, TI, n_init=-1, Fw=f, j=f)
    >>> Mz_i = dc._Mz_free(R1, TI, n_init=-1, Fw=f, j=-f)

    >>> plt.plot(TI, Mz, label='No flow', linewidth=3)
    >>> plt.plot(TI, Mz_e, label='Equilibrium inflow', linewidth=3)
    >>> plt.plot(TI, Mz_i, label='Inverted inflow', linewidth=3)
    >>> plt.xlabel('Inversion time (sec)')
    >>> plt.ylabel('Magnetization (A/cm)')
    >>> plt.legend()
    >>> plt.show()

    Now consider a two-compartment model, with a central compartment 
    that has in- and outflow, and a peripheral compartment that only 
    exchanges with the central compartment:

    >>> R1 = [1,2]
    >>> v = [0.3, 0.7]
    >>> PS = 0.1
    >>> Fw = [[f, PS], [PS, 0]]
    >>> Mz = dc._Mz_free(R1, TI, v, Fw, n_init=-1, j=[f, 0])

    >>> plt.plot(TI, Mz[0,:], label='Central compartment', linewidth=3)
    >>> plt.plot(TI, Mz[1,:], label='Peripheral compartment', linewidth=3)
    >>> plt.xlabel('Inversion time (sec)')
    >>> plt.ylabel('Magnetization (A/cm)')
    >>> plt.legend()
    >>> plt.show()

    In DC-MRI the more usual situation is one where TI is fixed and the 
    relaxation rates are variable due to the effect of a contrast agent. 
    As an illustration, consider the previous result again at TI=500 msec 
    and an R1 that is linearly declining in the central compartment and 
    constant in the peripheral compartment:

    >>> TI = 0.5
    >>> nt = 1000
    >>> t = 0.1*np.arange(nt)
    >>> R1 = np.stack((1-t/np.amax(t), np.ones(nt)))
    >>> j = np.stack((f*np.ones(nt), np.zeros(nt)))
    >>> Mz = dc._Mz_free(R1, TI, v, Fw, n_init=-1, j=j)

    >>> plt.plot(t, Mz[0,:], label='Central compartment', linewidth=3)
    >>> plt.plot(t, Mz[1,:], label='Peripheral compartment', linewidth=3)
    >>> plt.xlabel('Time (sec)')
    >>> plt.ylabel('Magnetization (A/cm)')
    >>> plt.legend()
    >>> plt.show()   

    The function allows for R1 and TI to be both variable. Computing the 
    result for 10 different TI values and extracting the result 
    corresponding to TI=0.5 gives again the same result:

    >>> TI = 0.1*np.arange(10)
    >>> Mz = dc._Mz_free(R1, TI, v, Fw, n_init=-1, j=j)

    >>> plt.plot(t, Mz[0,:,5], label='Central compartment', linewidth=3)
    >>> plt.plot(t, Mz[1,:,5], label='Peripheral compartment', linewidth=3)
    >>> plt.xlabel('Time (sec)')
    >>> plt.ylabel('Magnetization (A/cm)')
    >>> plt.legend()
    >>> plt.show()      

"""
import numpy as np
from scipy.special import i0, i1

from dcmri.core.module import Module
from dcmri.utils.misc import sample
from dcmri.bloch.modules_tissue import Magnetization


class Signal(Module):
    configs = {
        'magnitude': {False, True},
    }
    defaults = {
        'magnitude': True,
    }
    def inputs(self):
        inputs = {'M', 'S0', 'noise_sdev'}
        inputs |= {'tacq', 'TS', 'dt'}
        return inputs
    
    def outputs(self):
        return {'S'} # (channels, components, times)
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        if p['M'].ndim not in [3, 4]:
            raise ValueError("Magnetization M must be a 3D or 4D array with shape (components, compartments, times) or (channels, components, compartments, times)")

        shape = p['M'].shape

        # Sum over compartments
        if p['M'].ndim==3:
            # (components, compartments, times)
            Mxy = p['M'][:2, :, :].sum(axis=1) 
            # (components, times)

            # Sample
            t = p['dt'] * np.arange(p['M'].shape[-1])
            Mxy = sample(p['tacq'], t, Mxy, p['TS'])
            # (components, times)

            # Add channel dimension of 1
            Mxy = Mxy[None, ...]
            # (channels, components, times)

        else:
            # (channels, components, compartments, times)
            Mxy = p['M'][:, :2, :, :].sum(axis=2) 
            # (channels, components, times)

            # Sample
            Mxy = Mxy.reshape(-1, shape[-1])
            # (channels * components, times)
            t = p['dt'] * np.arange(p['M'].shape[-1])
            Mxy = sample(p['tacq'], t, Mxy, p['TS'])
            Mxy = Mxy.reshape((shape[0], 2, -1))
            # (channels, components, times)

        if self.config['magnitude']:
            Mxy = np.linalg.norm(Mxy, axis=1, keepdims=True)
            S = signal_rice(p['S0'] * Mxy, p['noise_sdev'])
            # (channels, 1, times)
        else:
            S = p['S0'] * Mxy

        return {'S': S} # (channels, components, times)
    

def signal_rice(nu, sigma)-> np.ndarray:
    if sigma==0:
        return nu
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        K = nu**2 / (2*sigma**2)
        arg = K/2
        pref = sigma * np.sqrt(np.pi/2)
        rice_mean = pref * np.exp(-K/2) * ((1+K)*i0(arg) + K*i1(arg))
    # Nan values are points where the distribution is indistinguisable from Gaussian
    return np.where(np.isnan(rice_mean) | np.isinf(rice_mean), nu, rice_mean)

   


class CalibrateSignal(Module):
    configs = Magnetization.configs | Signal.configs
    defaults = Magnetization.defaults | Signal.defaults

    def __init__(self, imap:dict=None, **config):
        self.set_config(config)
        self._magn = Magnetization(
            imap = {'R1':'R1b', 'R2':'R2b', 'R2s':'R2sb', 'R1i':'R1ib'}, 
            **config,
        )
        self._signal = Signal(**config)
        self.map_inputs(imap)

    def inputs(self) -> set:
        inputs = {'Sb'}
        inputs |= self._magn.mapped_inputs()
        inputs |= self._signal.mapped_inputs()
        inputs -= {'S0', 'M'}
        return inputs

    def outputs(self) -> set:
        return 'S0'
    
    def map_lexicon(self, qvalues, data={}):
        p = self.map_data(qvalues, data)
        if 'Sb' in data:
            return p
        # Set baseline if not provided
        channels = 2 if self.config['sequence'] in ['Eq-DE-EPI', 'DE-EPI'] else 1
        components = 1 if self.config['magnitude'] else 2
        p['Sb'] = np.full((channels, components, 1), qvalues['Sb']) 
        p['tacq'] = np.atleast_1d(p['tacq'])       
        return p

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)    

        # If needed: extend scalar baseline values in time to match duration of Sb
        # Extend into (nt,) array
        relax_b = {}
        for Rb in ['R1b', 'R2b', 'R2sb', 'R1ib']:
            if Rb in p:
                relax_b[Rb] = np.full(p['Sb'].shape[2], p[Rb])

        # Compute signal scaling factor
        magn_b = self._magn(p, **relax_b)
        
        s_cal = self._signal(p, S0=1, tacq=p['tacq'][:p['Sb'].shape[2]], **magn_b)['S'] 
        with np.errstate(divide='ignore', invalid='ignore'):
            S0 = np.nanmean(np.where(s_cal != 0, p['Sb'] / s_cal, np.nan))

        return {'S0': S0}