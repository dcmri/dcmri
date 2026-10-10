# +--------------------------------------------------------------------------------------------------+
# |                              InverseTissueX - all configs (n = 11)                               |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, lin, quad                                                    | lin        |
# | inflow         | none, pool                                                         | none       |
# | sequence       | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,  | 3D-SPGR-SS |
# |                | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS,         |            |
# |                | 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR,           |            |
# |                | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,                   |            |
# |                | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                                  |            |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | water_exchange | FF, FN, FR, NF, NN, NR, RF, RN, RR                                 | FF         |
# | kinetics       | 2CU, 2CX, FX, HF, HFU, NX, NXP, U, WV                              | 2CX        |
# | baseline       | literature, measured                                               | literature |
# +--------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            InverseTissueX - all inputs (n = 48)                                                            |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                     | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                              | Indicator       | gadoterate |                |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                              | Indicator       | 0.005      | (0, 1)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR            |            | noise-to-signal ratio                                    | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S              | a.u.       | signal                                                   | Signal          | 1.0        | (0, 5)         |       |           |
# | S0             | a.u.       | signal scaling factor                                    | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | iStrig         |            | indices of the signal trigger                            | Signal          | None       |                |       |           |
# | nb             | a.u.       | number of baseline time points                           | Signal          | 1          |                |       |           |
# | pfree          | a.u.       | set of free parameters                                   | Signal          | 1          |                |       |           |
# | tS             | sec        | signal time points                                       | Signal          | 0.0        |                |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FA             | deg        | flip angle                                               | Sequence        | 15         | (0, 180)       |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space  | Sequence        | 64         | (0, 1000)      |       |           |
# | Nph            |            | number of acquired phase lines in k-space                | Sequence        | 128        | (0, 1000)      |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition            | Sequence        | 64         | (0, 1000)      |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                             | Sequence        | 90         | (0, 180)       |       |           |
# | TA             | sec        | acquisition time                                         | Sequence        | 2.0        | (0, 30)        |       |           |
# | TD             | sec        | prepulse delay                                           | Sequence        | 0.05       | (0, 1)         |       |           |
# | TE             | sec        | echo time                                                | Sequence        | 0.001      | (0, 10)        |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                 | Sequence        | 0.001      | (0, 1)         |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                | Sequence        | 0.005      | (0, 1)         |       |           |
# | TP             | sec        | preparation delay                                        | Sequence        | 0.05       | (0, 1)         |       |           |
# | TR             | sec        | repetition time                                          | Sequence        | 0.005      | (0, 1)         |       |           |
# | field_strength | T          | magnetic field strength                                  | Sequence        | 3          | (0, 20)        |       |           |
# | iz             |            | slice number in a multi-slice acquisition                | Sequence        | 0          | (0, 1000)      |       |           |
# | tstart         | sec        | start of the acquisition                                 | Sequence        | 0          | (0, 10000.0)   |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | B1corr         |            | B1-correction factor                                     | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                   | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_bi          | Hz         | tissue R1 in blood and interstitium                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                       | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_i           | Hz         | tissue R1 in interstitium                                | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ic          | Hz         | tissue R1 in interstitium and cells                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ti          | Hz         | tissue R1 in the tissue                                  | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                | Electromagnetic | 1          | (0, 5)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | F_b            | mL/sec/cm3 | flow per unit tissue in the blood                        | Physiological   | 0.02       | (0, 1)         |       |           |
# | H              |            | hematocrit                                               | Physiological   | 0.45       | (0, 1)         |       |           |
# | Ktrans         | mL/sec/cm3 | plasma clearance                                         | Physiological   | 0.015      | (0.0, 0.1)     |       |           |
# | P              |            | porosity                                                 | Physiological   | 0.3        | (0, 1)         |       |           |
# | PS             | mL/sec/cm3 | permeability-surface area product                        | Physiological   | 0.003      | (0, 1)         |       |           |
# | PSc            | mL/sec/cm3 | transcytolemmal water permeability-surface area product  | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSe            | mL/sec/cm3 | transendothelial water permeability-surface area product | Physiological   | 0.03       | (0, 100)       |       |           |
# | T_ar           | sec        | mean transit time in the artery                          | Physiological   | 30         | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                             | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_bi           | mL/cm3     | volume fraction in blood and interstitium                | Physiological   | 1          | (0, 1)         |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                 | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_e            | mL/cm3     | volume fraction in extracellular space                   | Physiological   | 0.3        | (0.01, 0.6)    |       |           |
# | v_i            | mL/cm3     | volume fraction in interstitium                          | Physiological   | 0.3        | (0.001, 0.999) |       |           |
# | v_ic           | mL/cm3     | volume fraction in interstitium and cells                | Physiological   | 1          | (0, 1)         |       |           |
# | v_ti           | mL/cm3     | volume fraction in the tissue                            | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                              | Hyperparameters | 0.5        |                |       |           |
# +------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------+
# |                                       InverseTissueX - all outputs (n = 4)                                      |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | Key   | Unit | Name                                           | Group           | Init | Bounds | DICOM | OSIPI |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | loss  | a.u. | loss value of optimized model                  | Signal          | 1    |        |       |       |
# | pcov  | a.u. | dictionary with covariances of free parameters | Signal          | 1    |        |       |       |
# | popt  | a.u. | dictionary of optimized free parameter values  | Signal          | 1    |        |       |       |
# | psdev | a.u. | dictionary with parameter standard deviations  | Signal          | 1    |        |       |       |
# +-----------------------------------------------------------------------------------------------------------------+

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train
from dcmri.forward.tissue_x import ForwardTissueX as Forward

configs = deepcopy(Forward.configs) 
defaults = deepcopy(Forward.defaults)

class InverseTissueX(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'R1_b', 'T_ar', 'v_e', 'Nph', 'pfree', 'tS', 'PSe', 'TE', 'Nz', 'TP', 'agent', 'TE1', 'TD', 'Nk0', 'R1_bi', 'TR', 'B1corr', 'field_strength', 'H', 'v_bi', 'v_b', 'NSR', 'R1_ti', 'R1_ic', 'PA', 'S', 'tstart', 'R1_c', 'PS', 'Ktrans', 'P', 'v_ti', 'TA', 'v_c', 'FA', 'iStrig', 'iz', 'R1_i', 'v_i', 'S0', 'nb', 'v_ic', 'PSc', 'TE2', 'dt', 'F_b', 'c_ar', 'me'}
    _all_outputs = {'pcov', 'psdev', 'popt', 'loss'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        return pred['S'][:, :, :len(time)].reshape(-1)

    def _preproc(self, p):
        # Reshape signal if needed
        if p['S'].ndim == 1:
            p['S'] = p['S'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            p['Scal'] = p['S'][..., :p['nb']]
            p['iScal'] = np.arange(p['nb'])

    def _pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        if not self.config['calibrate']:
            pfree |= {'S0'}
        return {p: get_quantity(p)['bounds'] for p in pfree}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  
        self._preproc(p)

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self._pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        time = p['tS']
        signal = p['S']
        output = train(self._predict, time, signal, p, p['pfree'], **kwargs)

        return self.map_results(output)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        if self.config['calibrate']:
            inputs |= {'nb'}
            inputs -= {'Scal', 'iScal'}
        inputs |= {'tS', 'S', 'pfree'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'loss'}
        return outputs
    
    def test_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.test_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data({'v_b': (0, 1), 'v_i': (0, 1), 'v_e': (0, 1), 'F_b': (0, 1)}),
            'tS': pred['tS'],
            'S': pred['S'],
        }
        return self.input_data(p, data)

    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        p = self.map_data(data)
        self._preproc(p)
        prediction = self.forward(p)

        if xlim is None:
            xlim = [prediction['tR'][0], prediction['tR'][-1]]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Signals Plot
        ax0.set_title('Prediction of the MRI signals.')
        for i in range(p['S'].shape[0]):
            for j in range(p['S'].shape[1]):
                ax0.plot(p['tS'] / 60, p['S'][i, j, :], marker='o', linestyle='None', color='cornflowerblue', label='Data')
                ax0.plot(prediction['tS'] / 60, prediction['S'][i, j, :], linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations')

        ax1.plot(prediction['tC'] / 60, 1000 * p['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        if prediction['C'].shape[0] == 2:
            ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][0,:], linestyle='-', linewidth=3.0, color='darkred', label='Blood')
            ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Interstitium')
        else:
            ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Tissue')
           
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    # Original plot function showing magnetization etc
    # Needs some adaptation

    # def plot(self, data: dict, xlim:list=None, fname:str=None, show=True, 
    #          sdev=None, round_to=None):
    #     clr = {
    #         'Plasma': 'darkred',
    #         'Interstitium': 'steelblue',
    #         'Extracellular': 'dimgrey',
    #         'Tissue': 'darkgrey',
    #         'Blood': 'darkred',
    #         'Extravascular': 'blue',
    #         'Tissue cells': 'lightblue',
    #         'Blood + Interstitium': 'purple',
    #     }
    #     plot_labels_kin = {
    #         '2CX': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         '2CU': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         'HF': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         'HFU': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         'FX': (['ve'], ['Extracellular']),
    #         'NX': (['vb'], ['Blood']), 
    #         'NXP': (['vb'], ['Blood']),
    #         'U': (['vb'], ['Blood']),
    #         'WV': (['vi'], ['Interstitium']),
    #     }
    #     def plot_labels_relax(kin, wex) -> list:

    #         if wex == 'FF':
    #             return ['Tissue']

    #         if wex in ['RR', 'NN', 'NR', 'RN']:
    #             if kin == 'WV':
    #                 return ['Interstitium', 'Tissue cells']
    #             else:
    #                 return ['Blood', 'Interstitium', 'Tissue cells']

    #         if wex in ['RF', 'NF']:
    #             if kin == 'WV':
    #                 return ['Extravascular']
    #             else:
    #                 return ['Blood', 'Extravascular']

    #         if wex in ['FR', 'FN']:
    #             if kin == 'WV':
    #                 return ['Interstitium', 'Tissue cells']
    #             else:
    #                 return ['Blood + Interstitium', 'Tissue cells']
                
    #     prediction = self._model(self._pars)
    #     if xlim is None:
    #         xlim = [np.amin(t), np.amax(t)]
    #     xlim = np.array(xlim) / 60

    #     if self._model.config['water_exchange'] != 'FF':
    #         fig, ax = plt.subplots(2, 2, figsize=(10, 12))
    #         fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #         ax00 = ax[0, 0]
    #         ax01 = ax[0, 1]
    #         ax10 = ax[1, 0]
    #         ax11 = ax[1, 1]
    #         ax_text = None
    #     else:
    #         fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #         fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #         ax00 = ax[0]
    #         ax01 = ax[1]
    #         ax_text = ax[2]

    #     ax00.set_title('MRI signals')
    #     for ci in range(self._shape[1]):
    #         if time is not None:
    #             ax00.plot(time / 60, self._predict_all(time)[0, ci, :], marker='o', linestyle='None', color='cornflowerblue', label='Predicted data')
    #             ax00.plot(time / 60, signal[0, ci, :], marker='x', linestyle='None', color='darkblue', label='Data')
    #         ax00.plot(t / 60, S[0, ci, :], linestyle='-', linewidth=3.0, color='darkblue', label='Model')
    #     ax00.set(ylabel='MRI signal (a.u.)', xlabel='Time (min)', xlim=xlim)
    #     ax00.legend()

    #     conc_comp, conc_label = plot_labels_kin[self._cnfg['kinetics']]
    #     relax_comp = plot_labels_relax(self._cnfg['kinetics'], self._cnfg['water_exchange'])
    
    #     ax01.set_title('Tissue concentration in indicator compartments')
    #     ax01.plot(t / 60, 1000 * self._pars['c_a'], linestyle='-', linewidth=5.0, color='lightcoral', label='Arterial blood')
    #     for k, vk in enumerate(conc_comp):
    #         # ck = C[k, ...] / p[vk] if p[vk] > 0 else 0 * C[k, ...]
    #         ax01.plot(t / 60, 1000 * C[0, k, :], linestyle='-', linewidth=3.0, label=conc_label[k], color=clr[conc_label[k]])
    #     ax01.set(ylabel='Concentration (mM)', xlabel='Time (min)', xlim=xlim)
    #     ax01.legend()

    #     if self._cnfg['water_exchange'] != 'FF':
    #         ax11.set_title('Concentration in water compartments')
    #         for i in range(c.shape[1]):
    #             ax11.plot(t / 60, 1000 * c[0, i, :], linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
    #         ax11.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
    #         ax11.legend()

    #         ax10.set_title('Magnetization in water compartments')
    #         for i in range(Mz.shape[1]):
    #             mi = Mz[0, i, :] / v[i] if v[i] > 0 else 0 * Mz[0, i, :]
    #             ax10.plot(t / 60, mi, linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
    #         ax10.set(xlabel='Time (min)', ylabel='Magnetization (a.u.)', xlim=xlim)
    #         ax10.legend()

    #     if ax_text is not None:

    #         if sdev is None:
    #             pars = self._params('free')
    #         else:
    #             pars = list(sdev.keys())
    #             sdev = {k: float(v) for k, v in sdev.items()}

    #         vals = {k: p[k][0] for k in pars}
    #         msg = string_params(vals, sdev, round_to)
    #         msg = "\n".join(list(msg.values()))
    #         ax_text.set_title('Free parameters')
    #         ax_text.axis("off")  # hide axes
    #         ax_text.text(0, 0.9, f"Kinetics: {self._cnfg['kinetics']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.85, f"Water exchange: {self._cnfg['water_exchange']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.8, f"Sequence: {self._cnfg['sequence']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.75, f"R2* model: {self._cnfg['t2s_relaxation']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.6, msg, fontsize=10, transform=ax_text.transAxes, ha="left", va="top")

    #     if fname is not None: 
    #         plt.savefig(fname=fname)
    #     if show: 
    #         plt.show()
    #     else: 
    #         plt.close()


