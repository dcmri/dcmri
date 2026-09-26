import matplotlib.pyplot as plt
import numpy as np


from dcmri.core.tools import get_quantity, get_bounds
from dcmri.utils.fit import train_bat, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.forward.aorta_liver import ForwardAortaLiver


class AortaLiver():
    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = ForwardAortaLiver(**config)

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
        return (
            pred['S_ao'][:, :, :len(time[0])].reshape(-1), 
            pred['S_li'][:, :, :len(time[1])].reshape(-1)
        )

    # ==========================================
    # User Interface
    # ==========================================

    def params(self, group=None) -> list:
        """Return a list of model parameters"""
        return self._params(group)

    def predict(self) -> dict:
        """Predicts the data."""
        return self._model(self._pars)
    
    def train(self, data: dict, free: dict = None, 
            bounds: dict = None, n0=1, **kwargs) -> tuple:

        p = self._pars
        
        # Estimate BAT 
        bat = estimate_bat(data['tS_ao'], data['S_ao'], n0)
        p['BAT'] = max(bat - p['T_hl'], 0)

        # Set calibration data
        if self._model.config['calibrate']:
            for roi in ['ao', 'li']:
                p[f"Scal_{roi}"] = data[f"S_{roi}"][..., :n0]
                p[f'iScal_{roi}'] = np.arange(n0)

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)

        time = (data['tS_ao'], data['tS_li'])
        signal = (data['S_ao'], data['S_li'])
        return train_bat(self._predict, time, signal, p, free, **kwargs)


    def plot(self, data: dict, xlim=None, fname=None, show=True):
        prediction = self._model(self._pars)

        if xlim is None: 
            xlim = [prediction['tR'][0], prediction['tR'][-1]]
        xlim = np.array(xlim) / 60
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot signals
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(prediction['tS_ao'], prediction['S_ao'], data['tS_ao'], data['S_ao'], ax1, ['lightcoral', 'darkred'])
        plot_data(prediction['tS_li'], prediction['S_li'], data['tS_li'], data['S_li'], ax3, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax2.plot(prediction['tC'] / 60, 1000 * prediction['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
        ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
        ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()


    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
        time = (data['tS_ao'], data['tS_li'])
        signal = (data['S_ao'], data['S_li'])

        pred = self._predict(time)
        signal = np.concatenate([s.reshape(-1) for s in signal])
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)
        
    # def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._model._config['liver'])
    #     return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    # def print_params(self, *args, round_to=None, group=None, 
    #                  fixed_only=False, free_only=False, deriv=False):
    #     """Pretty print model parameters"""
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._model._config['liver'])
    #     if args != ():
    #         pars = {k: v for k, v in self._pars.items() if k in args}
    #     if fixed_only:
    #         pars = {k: v for k, v in pars.items() if k not in self._params('free')}
    #     if free_only:
    #         pars = {k: v for k, v in pars.items() if k in self._params('free')}
    #     print_params(pars, round_to=round_to, group=group)