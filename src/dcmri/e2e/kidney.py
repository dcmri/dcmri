"""General model for whole-kidney signals.

See Also:
    `Liver`, `Tissue`

Args:
    kinetics (str, optional): Kinetic model for the kidneys. 
        Options are '2CF' (Two-compartment filtration) and 'HF' 
        (High-flow). Defaults to '2CF'. 
    sequence (str, optional): imaging sequence model. Possible 
        values are '3D-SPGR-SS' (steady-state), 'SR' (saturation-recovery), 
        and 'lin' (linear). Defaults to '3D-SPGR-SS'.
    params (dict, optional): values for the model parameters,
        specified as keyword parameters. Defaults are used for any 
        that are not provided. See table 
        :ref:`Kidney-defaults` for a list of parameters and 
        their default values.

Example:

    Use the model to fit minipig data. The AIF is corrupted by 
    inflow effects so for the purpose of this example we will 
    use a standard input function:

.. plot::
    :include-source:
    :context: close-figs

    >>> import numpy as np
    >>> import pydmr
    >>> import dcmri as dc

    Read the dataset:

    >>> datafile = dc.fetch('minipig_renal_fibrosis')
    >>> data = pydmr.read(datafile, 'nest')
    >>> rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']
    >>> time = pars['TS'] * np.arange(len(rois['LeftKidney']))

    Generate an AIF at high temporal resolution (250 msec):

    >>> dt = 0.25
    >>> t = np.arange(0, np.amax(time) + dt, dt) 
    >>> ca = dc.aif.tristan(
    ...    t, 
    ...    agent="gadoterate",
    ...    dose=pars['dose'],
    ...    rate=pars['rate'],
    ...    weight=pars['weight'],
    ...    CO=60,
    ...    BAT=time[np.argmax(rois['Aorta'])] - 20,
    >>> )        

    Initialize the tissue:

    >>> kidney = dc.Kidney(
    ...    ca=ca,
    ...    dt=dt,
    ...    kinetics='HF',
    ...    field_strength=pars['B0'],
    ...    agent="gadoterate",
    ...    t0=pars['TS'] * pars['n0'],
    ...    TS=pars['TS'], 
    ...    TR=pars['TR'],
    ...    FA=pars['FA'],
    ...    R1ba=1/dc.const.T1(pars['B0'], 'blood'),
    ...    R1b=1/dc.const.T1(pars['B0'], 'kidney'),
    >>> )

    Train the kidney on the data:

    >>> kidney.set_free(Ta=[0,30])
    >>> kidney.train(time, rois['LeftKidney'])
    
    Plot the reconstructed signals and concentrations:

    >>> kidney.plot(time, rois['LeftKidney'])

    Print the model parameters:

    >>> kidney.print_params(round_to=4)
    --------------------------------
    Free parameters with their stdev
    --------------------------------
    Arterial mean transit time (Ta): 13.8658 (0.1643) sec
    Plasma volume (vp): 0.0856 (0.003) mL/cm3
    Tubular flow (Ft): 0.0024 (0.0001) mL/sec/cm3
    Tubular mean transit time (Tt): 116.296 (7.6526) sec

"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_quantity, get_bounds
from dcmri.inverse.sig2conc import SignalToConc
from dcmri.core.types import Input
from dcmri.utils.fit import train_bat, loss
from dcmri.models.kidney import KidneyModel


class Kidney():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = KidneyModel(**config)

        # Initialise model parameters
        pars = self._model.dummy_data()
        if data is not None:
            pars |= data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if get_quantity(p)['group']=='phys'} 
            params_free |= {p for p in ['BAT', 'BAT_1', 'BAT_2'] if p in params}
            return params_free
        return params

    def _predict(self, time: tuple):
        pred = self._model(self._pars)
        return pred['S'][:, :, :len(time)].reshape(-1)

    # ==========================================
    # User Interface
    # ==========================================

    def params(self, group=None) -> list:
        """Return a list of model parameters"""
        return self._params(group)

    def predict(self) -> np.ndarray:
        """Predicts the data."""
        return self._model(self._pars)
    
    def train(
        self, data: dict, aif:dict=None, 
        free: dict=None, bounds: dict=None, n0=1, **kwargs):

        p = self._pars
        
        if aif is not None:
            input = Input(aif)
            ca = SignalToConc(**self._model.config)(
                p, S=input.signal, R1b=input.R1b, nb=n0, 
                B1corr=input.B1corr, 
            )
            t = np.arange(0, np.amax(data['tS']) + p['dt'], p['dt'])
            p['c_ar'] = np.interp(t, input.time, ca['C'])

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)
        time = data['tS']
        signal = data['S']
        return train_bat(self._predict, time, signal, p, free, **kwargs)


    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        prediction = self._model(self._pars)

        if xlim is None:
            xlim = [prediction['tR'][0], prediction['tR'][-1]]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Signals Plot
        ax0.set_title('Prediction of the MRI signals.')
        for i in range(data['S'].shape[0]):
            for j in range(data['S'].shape[1]):
                ax0.plot(data['tS'] / 60, data['S'][i, j, :], marker='o', linestyle='None', color='cornflowerblue', label='Data')
                ax0.plot(prediction['tS'] / 60, prediction['S'][i, j, :], linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations')

        ax1.plot(prediction['tC'] / 60, 1000 * self._pars['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][0,:], linestyle='-', linewidth=3.0, color='darkred', label='Blood')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Tubuli')
           
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


    def cost(self, data: dict, metric: str = 'NRMS', nfree=None) -> float:
        time = data['tS']
        signal = data['S'].reshape(-1)

        signal_pred = self._predict(time)
        return loss(signal_pred, signal, metric, nfree)

