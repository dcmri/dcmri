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

from dcmri.core.module import Module
from dcmri.utils.misc import sample


class Signal(Module):
    configs = {
        'magnitude': {False, True},
        'sample': {False, True},
    }
    defaults = {
        'magnitude': False,
        'sample': False,
    }
    def inputs(self):
        inputs = {'M', 'S0'}
        if self.config['sample']:
            inputs += {'time', 'TS', 'dt'}
        return inputs
    
    def outputs(self):
        return {'S'} # (times) or (channels, times)
    
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
            if self.config['sample']:
                t = p['dt'] * np.arange(p['M'].shape[-1])
                Mxy = sample(p['time'], t, Mxy, p['TS'])
            # (components, times)
            xy_axis = 0

        else:
            # (channels, components, compartments, times)
            Mxy = p['M'][:, :2, :, :].sum(axis=2) 
            # (channels, components, times)
            if self.config['sample']:
                Mxy = Mxy.reshape(-1, shape[-1])
                # (channels * components, times)
                t = p['dt'] * np.arange(p['M'].shape[-1])
                Mxy = sample(p['time'], t, Mxy, p['TS'])
                Mxy = Mxy.reshape((shape[0], 2, -1))
                # (channels, components, times)
            xy_axis = 1

        if self.config['magnitude']:
            Mxy = np.linalg.norm(Mxy, axis=xy_axis)
            # (times) or (channels, times)

        S = p['S0'] * Mxy

        return {'S': S}
   


# Original Signal class including the calibration option
# In the new setup, calibration needs to be done at end-to-end model level.

# class Signal(Module):
#     configs = {
#         'sequence': set(SEQUENCES.keys()),
#         'inflow': {False, True},
#         'calibrate': {False, True},
#     }
#     defaults = {
#         'sequence': '3D-SPGR-SS',
#         'inflow': False,
#         'calibrate': False,
#     }
#     def __init__(self, imap: dict=None, **config):
#         self.set_config(config)
#         self._long = Longitudinal(**config)
#         self._read = Readout(**config)
#         self.map_inputs(imap) 
    
#     def inputs(self):
#         inputs = self._long.mapped_inputs()
#         inputs |= self._read.mapped_inputs()
#         if self.config['calibrate']:
#             inputs |= {'Sb'}
#             inputs -= {'S0'}
#         return inputs
    
#     def outputs(self):
#         return {'S'}
    
#     def __call__(self, data: dict=None, **kwargs) -> dict:
#         p = self.map_data(data, kwargs)

#         if self.config['calibrate']:
#             multichannel = self.config['sequence'] in ['Eq-DE-EPI', 'DE-EPI'] # TODO: All signals multichannel? Or have property in SEQUENCE?

#             # Baseline values
#             baseline = {'S0': 1}
#             for R in {'R1', 'R2', 'R2s'}:
#                 if R in p:
#                     Rb = p[R] if np.isscalar(p[R]) else p[R][0]

#                     # The baseline signal is not always a scalar constant (non steady state sequences)
#                     # Extend into (nt,) array if p["Sb"] is (nc, nt) (multichannel) or (nt, ) (single channel)
#                     if multichannel:
#                         if p['Sb'].ndim == 2:
#                             Rb = np.full(p['Sb'].shape[1], Rb)
#                     else:
#                         if not np.isscalar(p['Sb']):
#                             Rb = np.full(p['Sb'].shape[0], Rb)
#                     baseline[R] = Rb

#             # Derive S0
#             s_cal = self._signal(p | baseline)
#             p['S0'] = np.mean(np.where(s_cal > 0, p["Sb"] / s_cal, 0.0))

#         return self._signal(p)
    
#     def _signal(self, p):
#         if 'R1' in p:
#             Mz_arr = self._long(p)['Mz']
#         elif 'R2' in p:
#             Mz_arr = np.full_like(p['R2'], p['me']).reshape(1, -1)
#         elif 'R2s' in p:
#             Mz_arr = np.full_like(p['R2s'], p['me']).reshape(1, -1)
#         return self._read(p | {'Mz': Mz_arr})['S']
    
#     def map_lexicon(self, qvalues):
#         p = self.map_data(qvalues)
#         if self.config['calibrate']:
#             if self.config['sequence'] in ['Eq-DE-EPI', 'DE-EPI']:
#                 p['Sb'] = np.full(2, p['Sb'])
#         return p     


