"""General model for liver tissue.

This is the standard interface for liver tissues with known input 
function(s). For more detail see :ref:`liver-tissues`.

Args:
    kinetics (str, optional): Tracer-kinetic model. See table 
        :ref:`table-liver-models` for options. Defaults to '2I-EC'.
    non_stationary (str, optional): For intracellular tracers - stationarity 
        regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
        For more detail see :ref:`liver-tissues`. Defaults to None.
    sequence (str, optional): imaging sequence. Possible values are 'SS'
        and 'SR'. Defaults to 'SS'.
    free (dict, optional): Dictionary with free parameters and their
        bounds. If not provided, a default set of free parameters is used.
        Defaults to None.
    params (dict, optional): values for the parameters of the tissue,
        specified as keyword parameters. Defaults are used for any that are
        not provided. See tables :ref:`Liver-parameters` and
        :ref:`Liver-defaults` for a list of parameters and their
        default values.

See Also:
    `Tissue`

Example:

    Fit a dual-inlet liver model:

.. plot::
    :include-source:
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> import dcmri as dc

    Use `fake.liver` to generate synthetic test data:

    >>> time, aif, vif, roi, gt = dc.fake.liver()

    Build a tissue model and set the constants to match the experimental 
    conditions of the synthetic test data. Note the default model is the 
    dual-inlet model for extracellular agents (2I-EC). Since the 
    synthetic data are generated with an intracellular agent, the default 
    for the kinetic model needs to be overwritten:

    >>> model = dc.Liver(
    ...     kinetics = '2I-IC',
    ...     t = time,
    ...     agent = 'gadoxetate',
    ...     field_strength = 3.0,
    ...     TR = 0.005,
    ...     FA = 15,
    ...     R1b = 1/dc.const.T1(3.0,'liver'),
    ...     R1ba = 1/dc.const.T1(3.0, 'blood'), 
    ...     R1bv = 1/dc.const.T1(3.0, 'blood'), 
    ... )

    Train the model on the ROI data:

    >>> model.train(time, roi, aif, vif, n0=10)

    Plot the reconstructed signals (left) and concentrations (right) and 
    compare the concentrations against the noise-free ground truth. Since 
    the data are analysed with an exact model, and there are no other data 
    errors present, this should fior the data exactly.

    >>> model.plot(time, roi, ref=gt)

"""
import matplotlib.pyplot as plt
import numpy as np

from dcmri.inverse.sig2conc import SignalToConc
from dcmri.core.types import Input
from dcmri.core.tools import get_quantity, get_bounds
from dcmri.utils.fit import train_bat, loss
from dcmri.models.liver import LiverModel


class Liver():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = LiverModel(**config)

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
            self, data: dict, aif: dict=None, vif: dict=None, 
            free: dict=None, bounds: dict=None, n0=1, **kwargs):

        p = self._pars

        def conc(input: Input):
            ca = SignalToConc(**self._model.config)(
                p, S=input.signal, R1b=input.R1b, nb=n0, 
                B1corr=input.B1corr, 
            )
            t = np.arange(0, np.amax(data['tS']) + p['dt'], p['dt'])
            return np.interp(t, input.time, ca['C'])
      
        if aif is not None: 
            c_la = conc(Input(aif)) # Needs a better strategy for dual inlet possibly separate ca and cv inputs
            c_pv = conc(Input(vif))
            p['ci_li'] = (c_la, c_pv)

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)
        time = data['tS']
        signal = data['S']
        return train_bat(self._predict, time, signal, p, free, **kwargs)

    
    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        prediction = self._model(self._pars)

        xlim = xlim or [np.amin(prediction['tR']), np.amax(prediction['tR'])]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signals Plot
        ax0.set_title('MRI Signal Prediction')
        for i in range(data['S'].shape[0]):
            for j in range(data['S'].shape[1]):
                ax0.plot(data['tS'] / 60, data['S'][i, j, :], 'o', color='cornflowerblue', label='Data')
                ax0.plot(prediction['tS'] / 60, prediction['S'][i, j, :], '-', linewidth=3, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=np.array(xlim) / 60)
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        if '1I' in self._model.config['kinetics']:
            ax1.plot(prediction['tC'] / 60, 1000 * self._pars['ci_li'], '-', linewidth=3, color='darkred', label='Input')
        if '2I' in self._model.config['kinetics']:
            ax1.plot(prediction['tC'] / 60, 1000 * self._pars['ci_li'][0], '-', linewidth=3, color='darkred', label='Arterial')
            ax1.plot(prediction['tC'] / 60, 1000 * self._pars['ci_li'][1], '-', linewidth=3, color='purple', label='Portal')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][0,:], '-.', linewidth=3, color='darkblue', label='Extracellular')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][1,:], '-', linewidth=3, color='green', label='Hepatocytes]')

        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim) / 60)
        ax1.legend()

        if fname: 
            plt.savefig(fname)
        if show: 
            plt.show()
        else: 
            plt.close()


    def cost(self, data: dict, metric: str = 'NRMS', nfree=None) -> float:
        time = data['tS']
        signal = data['S'].reshape(-1)
        signal_pred = self._predict(time)
        return loss(signal_pred, signal, metric, nfree)
    


    # def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    # def print_params(self, *args, round_to=None, group=None, 
    #                  fixed_only=False, free_only=False, deriv=False):
    #     """Pretty print model parameters"""
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     if args != ():
    #         pars = {k: v for k, v in self._pars.items() if k in args}
    #     if fixed_only:
    #         pars = {k: v for k, v in pars.items() if k not in self._params('free')}
    #     if free_only:
    #         pars = {k: v for k, v in pars.items() if k in self._params('free')}
    #     print_params(pars, round_to=round_to, group=group)

