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
from dcmri.relaxivity.modules_tissue import ConcToRelax
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.bloch.functions_sequences import channels


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

# +--------------------------------------------------------------------------------------------------+
# |                                   Signal - all configs (n = 3)                                   |
# +-----------+----------------------------------------------------------------------------+---------+
# | Key       | Values                                                                     | Default |
# +-----------+----------------------------------------------------------------------------+---------+
# | magnitude | False, True                                                                | True    |
# | trigger   | False, True                                                                | False   |
# | calibrate | False, True                                                                | False   |
# +--------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------+
# |                                        Signal - all inputs (n = 7)                                         |
# +--------+------+-------------------------------+-----------------+------+---------------+-------+-----------+
# | Key    | Unit | Name                          | Group           | Init | Bounds        | DICOM | OSIPI     |
# +--------+------+-------------------------------+-----------------+------+---------------+-------+-----------+
# | NSR    |      | noise-to-signal ratio         | Signal          | 0.0  | (0, 100000.0) |       |           |
# | S0     | a.u. | signal scaling factor         | Signal          | 1.0  | (0, 5)        |       | Q.MS1.010 |
# | Scal   | a.u. | calibration signal            | Signal          | 1.0  | (0, 5)        |       | Q.MS1.002 |
# | iScal  |      | indices of calibration signal | Signal          | 0    |               |       |           |
# | iStrig |      | indices of the signal trigger | Signal          | None |               |       |           |
# +--------+------+-------------------------------+-----------------+------+---------------+-------+-----------+
# | M      | A/cm | magnetization                 | Electromagnetic | 1    | (0, 5)        |       |           |
# | tM     | sec  | magnetization time points     | Electromagnetic | 0.0  |               |       |           |
# +------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------+
# |                               Signal - all outputs (n = 3)                               |
# +-----+------+-----------------------+-----------------+------+--------+-------+-----------+
# | Key | Unit | Name                  | Group           | Init | Bounds | DICOM | OSIPI     |
# +-----+------+-----------------------+-----------------+------+--------+-------+-----------+
# | S   | a.u. | signal                | Signal          | 1.0  | (0, 5) |       |           |
# | S0  | a.u. | signal scaling factor | Signal          | 1.0  | (0, 5) |       | Q.MS1.010 |
# +-----+------+-----------------------+-----------------+------+--------+-------+-----------+
# | tS  | sec  | signal time points    | Electromagnetic | 0.0  |        |       |           |
# +------------------------------------------------------------------------------------------+

class Signal(Module):
    configs = {
        'magnitude': {False, True},
        'trigger': {False, True},
        'calibrate': {False, True},
    }
    defaults = {
        'magnitude': True,
        'trigger': False,
        'calibrate': False,
    }
    def inputs(self):
        inputs = {'tM', 'M'} 
        if self.config['magnitude']:
            inputs |= {'NSR'}
        if self.config['calibrate']:
            inputs |= {'iScal', 'Scal'} 
        else:
            inputs |= {'S0'}
        if self.config['trigger']:
            inputs |= {'iStrig'}
        return inputs
    
    def outputs(self):
        outputs = {'tS', 'S'} # (channels, components, times)
        if self.config['calibrate']:
            outputs |= {'S0'}
        return outputs

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        if p['M'].ndim not in [4]:
            raise ValueError("Magnetization M must be a 4D array with shape (channels, components, compartments, times)")
        # (channels, components, compartments, times)

        # Sum xy components over compartments
        Mxy = p['M'][:, :2, :, :].sum(axis=2) 
        # (channels, components, times)

        results = {}

        # Build signal
        tS = p['tM']
        if self.config['calibrate']:
            S = Mxy # normalized signal (S0=1)
        else:
            S = p['S0'] * Mxy
        
        if self.config['magnitude']:
            S = np.linalg.norm(S, axis=1, keepdims=True)
            Sb = np.mean(S[:, 0, 0])
            noise_sdev = p['NSR'] / Sb if Sb != 0 else 0
            S = signal_rice(S, noise_sdev)
            # (channels, 1, times)

        if self.config['trigger']:
            if p['iStrig'] is not None:
                accept = p['iStrig']
                tS = tS[accept]
                S = S[:, :, accept]
                # (channels, components, times)

        if self.config['calibrate']:
            s_cal_norm = S[:, :, p['iScal']]
            s_cal = p['Scal']
            nozero = np.where(s_cal_norm != 0)
            results['S0'] = np.mean(s_cal[nozero] / s_cal_norm[nozero])
            S *= results['S0']

        results |= {'tS': tS, 'S': S}  # (channels, components, times)
        return self.map_results(results)
    
    def dummy_data(self, nt=5):
        data = self.init_data()
        n_channels = 1
        n0 = 1
        components = 1 if self.config['magnitude'] else 2
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            'tM': np.zeros(nt),
            'M': np.ones((n_channels, 3, 1, nt)), # (channels, components, compartments, times)
            'iScal': np.zeros(n0, dtype=int),
            'Scal': Scal, 
            'iStrig': np.zeros(n0, dtype=int),
        }
        return data

# +--------------------------------------------------------------------------------------------------+
# |                               RelaxToSignal - all configs (n = 5)                                |
# +-----------+-------------------------------------------------------------------------+------------+
# | Key       | Values                                                                  | Default    |
# +-----------+-------------------------------------------------------------------------+------------+
# | sequence  | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,       | 3D-SPGR-SS |
# |           | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS, 3D-PR-SPGR,  |            |
# |           | 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR, 3D-SPGR-SS, 3D-SPGR-SSI,   |            |
# |           | 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS, ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS  |            |
# | inflow    | False, True                                                             | False      |
# | magnitude | False, True                                                             | True       |
# | trigger   | False, True                                                             | False      |
# | calibrate | False, True                                                             | False      |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                     RelaxToSignal - all inputs (n = 32)                                                      |
# +---------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Key     | Unit       | Name                                                    | Group           | Init  | Bounds        | DICOM | OSIPI     |
# +---------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | NSR     |            | noise-to-signal ratio                                   | Signal          | 0.0   | (0, 100000.0) |       |           |
# | S0      | a.u.       | signal scaling factor                                   | Signal          | 1.0   | (0, 5)        |       | Q.MS1.010 |
# | Scal    | a.u.       | calibration signal                                      | Signal          | 1.0   | (0, 5)        |       | Q.MS1.002 |
# | iScal   |            | indices of calibration signal                           | Signal          | 0     |               |       |           |
# | iStrig  |            | indices of the signal trigger                           | Signal          | None  |               |       |           |
# +---------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | FA      | deg        | flip angle                                              | Sequence        | 15    | (0, 180)      |       |           |
# | Nk0     |            | number of acquired phase lines to the center of k-space | Sequence        | 64    | (0, 1000)     |       |           |
# | Nph     |            | number of acquired phase lines in k-space               | Sequence        | 128   | (0, 1000)     |       |           |
# | Nz      |            | number of slices in a multi-slice acquisition           | Sequence        | 64    | (0, 1000)     |       |           |
# | PA      | deg        | preparation Pulse Flip Angle                            | Sequence        | 90    | (0, 180)      |       |           |
# | SA      | deg        | saturation Slab Flip Angle                              | Sequence        | 0     | (0, 180)      |       |           |
# | TA      | sec        | acquisition time                                        | Sequence        | 2.0   | (0, 30)       |       |           |
# | TD      | sec        | prepulse delay                                          | Sequence        | 0.05  | (0, 1)        |       |           |
# | TE      | sec        | echo time                                               | Sequence        | 0.001 | (0, 10)       |       |           |
# | TE1     | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001 | (0, 1)        |       |           |
# | TE2     | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005 | (0, 1)        |       |           |
# | TP      | sec        | preparation delay                                       | Sequence        | 0.05  | (0, 1)        |       |           |
# | TR      | sec        | repetition time                                         | Sequence        | 0.005 | (0, 1)        |       |           |
# | iz      |            | slice number in a multi-slice acquisition               | Sequence        | 0     | (0, 1000)     |       |           |
# | tacq   | sec        | acquisition duration                                    | Sequence        | 240   | (0, 10000.0)  |       |           |
# | tstart | sec        | start of the acquisition                                | Sequence        | 0     | (0, 10000.0)  |       |           |
# +---------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | B1corr  |            | B1-correction factor                                    | Electromagnetic | 1     | (0, 5)        |       |           |
# | R1      | Hz         | tissue R1                                               | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | R1i     | Hz         | inlet R1                                                | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | R2      | Hz         | tissue R2                                               | Electromagnetic | 2.0   | (0, 5)        |       |           |
# | R2s     | Hz         | tissue R2*                                              | Electromagnetic | 20    | (0, 5)        |       |           |
# | me      | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1     | (0, 5)        |       |           |
# | tR      | sec        | relaxation rate time points                             | Electromagnetic | 0.0   |               |       |           |
# +---------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Fwi     | mL/sec/cm3 | inflow in all water compartments                        | Physiological   | 0.02  | (0, 1)        |       |           |
# | Kw      | mL/sec/cm3 | water exchange matrix                                   | Physiological   | 0     | (0, 1)        |       |           |
# | TF      | sec        | fnflow time                                             | Physiological   | 0.5   | (0, 10)       |       |           |
# | vw      | mL/cm3     | water volume fraction                                   | Physiological   | 1     | (0, 1)        |       |           |
# +----------------------------------------------------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------+
# |                             RelaxToSignal - all outputs (n = 5)                              |
# +-----+------+---------------------------+-----------------+------+--------+-------+-----------+
# | Key | Unit | Name                      | Group           | Init | Bounds | DICOM | OSIPI     |
# +-----+------+---------------------------+-----------------+------+--------+-------+-----------+
# | S   | a.u. | signal                    | Signal          | 1.0  | (0, 5) |       |           |
# | S0  | a.u. | signal scaling factor     | Signal          | 1.0  | (0, 5) |       | Q.MS1.010 |
# +-----+------+---------------------------+-----------------+------+--------+-------+-----------+
# | M   | A/cm | magnetization             | Electromagnetic | 1    | (0, 5) |       |           |
# | tM  | sec  | magnetization time points | Electromagnetic | 0.0  |        |       |           |
# | tS  | sec  | signal time points        | Electromagnetic | 0.0  |        |       |           |
# +----------------------------------------------------------------------------------------------+


class RelaxToSignal(Module): 
    configs = Magnetization.configs | Signal.configs
    defaults = Magnetization.defaults | Signal.defaults

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)
        self._magn = Magnetization(**self.config) 
        self._signal = Signal(**self.config)
        self.map_io(imap, omap, iomap)  
        
    def inputs(self):
        inputs = self._magn.mapped_inputs()
        inputs |= self._signal.mapped_inputs()
        inputs -= self._magn.new_mapped_outputs()
        return inputs 
   
    def outputs(self):
        outputs = self._magn.mapped_outputs()
        outputs |= self._signal.mapped_outputs()
        return outputs

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  
        p |= self._magn(p)
        p |= self._signal(p)
        return self.map_results(p)

    def dummy_data(self, nc=2, nt=5):
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            'tacq': nt-1,
            'tR': np.arange(nt),
            'R1': np.ones((nc, nt)),
            'R2': np.ones((nc, nt)),
            'R2s': np.ones(nt),
            'R1i': np.ones((nc, nt)),
            'Fwi': np.ones(nc),
            'inlets': np.arange(nc),
            'Kw': np.eye(nc),
            'vw': np.ones(nc) / nc,
            #'wx': [[0]],
            'iScal': np.zeros(n0, dtype=int),
            'Scal': Scal, 
            'iStrig': np.zeros(n0, dtype=int),
        }
        return data


# +--------------------------------------------------------------------------------------------------+
# |                                ConcToSignal - all configs (n = 9)                                |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, leakage, lin, quad                                           | lin        |
# | inflow         | False, True                                                        | False      |
# | sequence       | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,  | 3D-SPGR-SS |
# |                | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS,         |            |
# |                | 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR,           |            |
# |                | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,                   |            |
# |                | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                                  |            |
# | tof_corr       | False, True                                                        | False      |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# +--------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                           ConcToSignal - all inputs (n = 43)                                                          |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Key    | Unit       | Name                                                              | Group           | Init  | Bounds        | DICOM | OSIPI     |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | C      | mmol/cm3   | tissue concentration                                              | Indicator       | 0.005 | (0, 1)        |       |           |
# | ci     | mmol/mL    | inlet concentration                                               | Indicator       | 0.005 |               |       |           |
# | tC     | sec        | concentration time points                                         | Indicator       | 0.0   |               |       |           |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | NSR    |            | noise-to-signal ratio                                             | Signal          | 0.0   | (0, 100000.0) |       |           |
# | S0     | a.u.       | signal scaling factor                                             | Signal          | 1.0   | (0, 5)        |       | Q.MS1.010 |
# | Scal   | a.u.       | calibration signal                                                | Signal          | 1.0   | (0, 5)        |       | Q.MS1.002 |
# | iScal  |            | indices of calibration signal                                     | Signal          | 0     |               |       |           |
# | iStrig |            | indices of the signal trigger                                     | Signal          | None  |               |       |           |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | FA     | deg        | flip angle                                                        | Sequence        | 15    | (0, 180)      |       |           |
# | Nk0    |            | number of acquired phase lines to the center of k-space           | Sequence        | 64    | (0, 1000)     |       |           |
# | Nph    |            | number of acquired phase lines in k-space                         | Sequence        | 128   | (0, 1000)     |       |           |
# | Nz     |            | number of slices in a multi-slice acquisition                     | Sequence        | 64    | (0, 1000)     |       |           |
# | PA     | deg        | preparation Pulse Flip Angle                                      | Sequence        | 90    | (0, 180)      |       |           |
# | SA     | deg        | saturation Slab Flip Angle                                        | Sequence        | 0     | (0, 180)      |       |           |
# | TA     | sec        | acquisition time                                                  | Sequence        | 2.0   | (0, 30)       |       |           |
# | TD     | sec        | prepulse delay                                                    | Sequence        | 0.05  | (0, 1)        |       |           |
# | TE     | sec        | echo time                                                         | Sequence        | 0.001 | (0, 10)       |       |           |
# | TE1    | sec        | first echo time in a multi-echo sequence                          | Sequence        | 0.001 | (0, 1)        |       |           |
# | TE2    | sec        | second echo time in a multi-echo sequence                         | Sequence        | 0.005 | (0, 1)        |       |           |
# | TP     | sec        | preparation delay                                                 | Sequence        | 0.05  | (0, 1)        |       |           |
# | TR     | sec        | repetition time                                                   | Sequence        | 0.005 | (0, 1)        |       |           |
# | iz     |            | slice number in a multi-slice acquisition                         | Sequence        | 0     | (0, 1000)     |       |           |
# | tacq   | sec        | acquisition duration                                              | Sequence        | 240   | (0, 10000.0)  |       |           |
# | tstart | sec        | start of the acquisition                                          | Sequence        | 0     | (0, 10000.0)  |       |           |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | B1corr |            | B1-correction factor                                              | Electromagnetic | 1     | (0, 5)        |       |           |
# | R1b    | Hz         | precontrast tissue R1                                             | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | R1ib   | Hz         | precontrast inlet R1                                              | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | R2b    | Hz         | precontrast tissue R2                                             | Electromagnetic | 20    | (0, 100)      |       |           |
# | R2sb   | Hz         | precontrast tissue R2*                                            | Electromagnetic | 20    | (0, 100)      |       |           |
# | me     | A cm2/mL   | equilibrium magnetization                                         | Electromagnetic | 1     | (0, 5)        |       |           |
# | r1     | Hz/M       | longitudinal contrast agent relaxivity                            | Electromagnetic | 3500  | (0, 10000.0)  |       |           |
# | r1i    | Hz/M       | inlet longitudinal contrast agent relaxivity                      | Electromagnetic | 3500  | (0, 10000.0)  |       |           |
# | r2     | Hz/M       | transverse contrast agent relaxivity                              | Electromagnetic | 4000  | (0, 10000.0)  |       |           |
# | r2s    | Hz/M       | transverse contrast agent relaxivity                              | Electromagnetic | 20000 | (0, 100000.0) |       |           |
# | r2se   | Hz/M       | extravascular, extracellular transverse contrast agent relaxivity | Electromagnetic | 20000 | (0, 100000.0) |       |           |
# | r2sq   | Hz/M^2     | quadratic transverse contrast agent relaxivity                    | Electromagnetic | 1000  | (0, 10000.0)  |       |           |
# | r2sv   | Hz/M       | vascular transverse contrast agent relaxivity                     | Electromagnetic | 20000 | (0, 100000.0) |       |           |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Fwi    | mL/sec/cm3 | inflow in all water compartments                                  | Physiological   | 0.02  | (0, 1)        |       |           |
# | Kw     | mL/sec/cm3 | water exchange matrix                                             | Physiological   | 0     | (0, 1)        |       |           |
# | TF     | sec        | inflow time                                                       | Physiological   | 0.5   | (0, 10)       |       |           |
# | inlets |            | water inlet compartments                                          | Physiological   | (0,)  |               |       |           |
# | vw     | mL/cm3     | water volume fraction                                             | Physiological   | 1     | (0, 1)        |       |           |
# | wx     |            | indicator-to-water compartment map                                | Physiological   |       |               |       |           |
# +-------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------+
# |                              ConcToSignal - all outputs (n = 10)                               |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-----------+
# | Key | Unit | Name                        | Group           | Init | Bounds | DICOM | OSIPI     |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-----------+
# | S   | a.u. | signal                      | Signal          | 1.0  | (0, 5) |       |           |
# | S0  | a.u. | signal scaling factor       | Signal          | 1.0  | (0, 5) |       | Q.MS1.010 |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-----------+
# | M   | A/cm | magnetization               | Electromagnetic | 1    | (0, 5) |       |           |
# | R1  | Hz   | tissue R1                   | Electromagnetic | 0.65 | (0, 5) |       |           |
# | R1i | Hz   | inlet R1                    | Electromagnetic | 0.65 | (0, 5) |       |           |
# | R2  | Hz   | tissue R2                   | Electromagnetic | 2.0  | (0, 5) |       |           |
# | R2s | Hz   | tissue R2*                  | Electromagnetic | 20   | (0, 5) |       |           |
# | tM  | sec  | magnetization time points   | Electromagnetic | 0.0  |        |       |           |
# | tR  | sec  | relaxation rate time points | Electromagnetic | 0.0  |        |       |           |
# | tS  | sec  | signal time points          | Electromagnetic | 0.0  |        |       |           |
# +------------------------------------------------------------------------------------------------+

class ConcToSignal(Module): 
    configs = ConcToRelax.configs | RelaxToSignal.configs 
    defaults = ConcToRelax.defaults | RelaxToSignal.defaults

    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        self._conc_to_relax = ConcToRelax(**self.config)
        self._relax_to_signal = RelaxToSignal(**self.config)
        self.map_io(imap, omap, iomap)  
        
    def inputs(self):
        inputs = {'tC'}
        inputs |= self._conc_to_relax.mapped_inputs()
        inputs |= self._relax_to_signal.mapped_inputs()
        inputs -= self._conc_to_relax.new_mapped_outputs()
        inputs -= self._relax_to_signal.new_mapped_outputs()
        inputs -= {'tR'}
        return inputs 
   
    def outputs(self):
        outputs = {'tR'}
        outputs |= self._conc_to_relax.mapped_outputs() 
        outputs |= self._relax_to_signal.mapped_outputs()
        return outputs

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs) 
        p['tR'] = p['tC']
        p |= self._conc_to_relax(p)
        p |= self._relax_to_signal(p)
        return self.map_results(p)

    def dummy_data(self, nc=2, nt=5):
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            'tacq': nt-1,
            'r1i': np.ones(nc),
            'tC': np.arange(nt),
            'C': np.ones((nc, nt)),
            'ci': np.ones((nc, nt)),
            'R1b': np.ones(nc),
            'R2b': np.ones(nc),
            'R1ib': np.ones(nc),
            'r1': np.ones(nc),
            'r2': np.ones(nc),
            'Fwi': np.ones(nc),
            'inlets': np.arange(nc),
            'Kw': np.eye(nc),
            'RM': np.eye(nc),
            'v': np.ones(nc) / nc,
            'vw': np.ones(nc) / nc,
            # 'wx': [[i] for i in range(nc)],
            'iScal': np.zeros(n0, dtype=int),
            'Scal': Scal, 
            'iStrig': np.zeros(n0, dtype=int),
        }
        return data
