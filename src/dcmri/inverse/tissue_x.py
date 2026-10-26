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
from dcmri.core.tools import get_quantity, update_bounds, unit
from dcmri.utils.fit import train_brute
from dcmri.forward.tissue_x import TissueX as ForwardModel
from dcmri.kinetics.functions_tissue import dpars_tissue
from dcmri.inverse.modules import SignalToConc
from dcmri.bloch.functions_sequences import repetition_time

configs = deepcopy(ForwardModel.configs) | {'aif': {'fixed', 'measured'}}
defaults = deepcopy(ForwardModel.defaults) | {'aif': 'fixed', 'calibrate': True}

class InverseTissueX(Module):

    configs = configs
    defaults = defaults

    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = ForwardModel(**self.config)
        if self.config['aif'] == 'measured':
            self._sig2conc = SignalToConc(**self.config)
        self.map_io(imap, omap)

    def _predict(self, p):
        nt = np.size(p['tS'])
        pred = self.forward(p)
        return pred['S'][:, :, :nt].reshape(-1)

    def _preproc(self, p):
        # Reshape signal if needed
        if p['S'].ndim == 1:
            p['S'] = p['S'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            p['Scal'] = p['S'][..., :p['nb']]
            p['iScal'] = np.arange(p['nb'])

        if self.config['aif'] == 'measured':
            tr = repetition_time(self.config['sequence'], p)
            tmax = max(p['tS_ar']) + 2 * (tr + p['dt'])
            t = np.arange(0, tmax, p['dt'])
            c_ar = self._sig2conc(p | {'S': p['S_ar'], 'R1b': p['R1_b']})['C']
            p['c_ar'] = np.interp(t, p['tS_ar'], c_ar) 

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        if not self.config['calibrate']:
            pfree |= {'S0'}
        pfree -= {'v_ti', 'H'}
        return {p: get_quantity(p)['bounds'] for p in pfree}

    def _pder(self, data:dict):
        p = {k: v for k, v in data.items() if get_quantity(k)['group'] in ['phys', 'body']}
        p = dpars_tissue(p)
        return p
    
    def _call(self, p: dict, **kwargs) -> dict: 
        self._preproc(p)

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self.pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        output = train_brute(self._predict, p['S'], p, p['pfree'], **kwargs) 
        output['pars'] = self._pder(p | output['popt'])

        return output

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        if self.config['calibrate']:
            inputs |= {'nb'}
            inputs -= {'Scal', 'iScal'}
        inputs |= {'tS', 'S', 'pfree'}
        if self.config['aif'] == 'measured':
            inputs |= {'tS_ar', 'S_ar', 'R1_b'}
            inputs -= {'c_ar'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'pars', 'loss'}
        if self.config['aif'] == 'measured':
            outputs |= {'c_ar'}
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
            'tS_ar': pred['tS'],
            'S_ar': np.ones(len(pred['tS'])) ** 2,
        }
        return self.input_data(p, data)


    def print(self, result: dict = None, data: dict = None):
        if result is None:
            result = self.__call__(data)

        header = ['Parameter', 'Unit', 'Reconstructed']
        if data is not None:
            header += ['Truth', 'Error']

        rows = []
        for par, rec in result['popt'].items():
            row = [par, unit(par), f'{rec:.4g}']
            if data is not None:
                if np.isinf(data[par]):
                    err = 'nan'
                elif data[par]==0:
                    err = 'inf'
                else:
                    err = 100 * (rec - data[par]) / data[par]
                    err = f'{err:.1f} %'
                row += [f'{data[par]:.4g}', err]
            rows.append(row)

        # Column widths adapt to the longest entry in each column
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

        def fmt(cells):
            return '| ' + ' | '.join(f'{c:<{w}}' for c, w in zip(cells, widths)) + ' |'

        divider = '+-' + '-+-'.join('-' * w for w in widths) + '-+'

        print(divider)
        print(fmt(header))
        print(divider)
        for r in rows:
            print(fmt(r))
        print(divider)


    def plot(self, data: dict, result:dict=None, xlim:list=None, fname:str=None, show=True):
        if result is None:
            result = self.__call__(data)
        p = self.map_data(data | result['popt'])
        self._preproc(p)
        pred = self.forward(p)
        
        if xlim is None:
            xlim = [pred['tR'][0], pred['tR'][-1]]
        xlim = np.array(xlim) / 60

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        ax[0].set_title('MRI signal')
        for i in range(pred['S'].shape[0]):
            for j in range(pred['S'].shape[1]):
                ax[0].plot(p['tS'] / 60, p['S'][i, j, :], marker='o', linestyle='None', color='cornflowerblue', label='Data')
                ax[0].plot(pred['tS'] / 60, pred['S'][i, j, :], linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax[0].set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=xlim)
        ax[0].legend()

        relax_comp = plot_labels(self.config['kinetics'], self.config['water_exchange'])
    
        ax[1].set_title('Tissue concentration')
        ax[1].plot(pred['tC'] / 60, 1000 * p['c_ar'], '-', linewidth=3, color='darkred', label='Arterial blood')
        if pred['C'].shape[0] == 2:
            ax[1].plot(pred['tC'] / 60, 1000 * pred['C'][0,:], linestyle='-', linewidth=3.0, color='darkred', label='Tissue blood')
            ax[1].plot(pred['tC'] / 60, 1000 * pred['C'][1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Interstitium')
        else:
            ax[1].plot(pred['tC'] / 60, 1000 * pred['C'][0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Tissue') 
        ax[1].set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax[1].legend()


        ax[2].set_title('Magnetization')
        M = np.linalg.norm(pred['M'], axis=1)
        for ch in range(M.shape[0]):
            for i in range(M.shape[1]):
                ax[2].plot(pred['tM'] / 60, M[ch, i, :], linestyle='-', linewidth=3.0, color=COLOR[relax_comp[i]], label=relax_comp[i])
        ax[2].set(xlabel='Time (min)', ylabel='Magnetization (A/cm)', xlim=xlim)
        ax[2].legend()

        if fname is not None: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()

COLOR = {
    'Plasma': 'darkred',
    'Interstitium': 'steelblue',
    'Extracellular': 'dimgrey',
    'Tissue': 'darkgrey',
    'Tissue blood': 'darkred',
    'Extravascular': 'blue',
    'Tissue cells': 'lightblue',
    'Tissue blood + Interstitium': 'purple',
}

def plot_labels(kin, wex) -> list:
    if wex == 'FF':
        return ['Tissue']

    if wex in ['RR', 'NN', 'NR', 'RN']:
        if kin == 'WV':
            return ['Interstitium', 'Tissue cells']
        else:
            return ['Tissue blood', 'Interstitium', 'Tissue cells']

    if wex in ['RF', 'NF']:
        if kin == 'WV':
            return ['Extravascular']
        else:
            return ['Tissue blood', 'Extravascular']

    if wex in ['FR', 'FN']:
        if kin == 'WV':
            return ['Interstitium', 'Tissue cells']
        else:
            return ['Tissue blood + Interstitium', 'Tissue cells']