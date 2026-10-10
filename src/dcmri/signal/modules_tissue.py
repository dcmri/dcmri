
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
# | tS  | sec  | signal time points    | Signal          | 0.0  |        |       |           |
# +------------------------------------------------------------------------------------------+

class Signal(Module):
    """Convert magnetization time courses into a measured MR signal.

    The transverse (x, y) components of the magnetization are summed over
    compartments and scaled by the equilibrium signal ``S0``. Optionally, the
    magnitude is taken and Rician noise is added, the signal is sampled at
    triggered time points only, and ``S0`` is estimated from a calibration
    measurement instead of being supplied.

    Configurations
    --------------
    magnitude : bool, default True
        If True, return the magnitude of the (x, y) signal with Rician noise
        added, giving one component. The noise standard deviation is
        ``NSR`` times the baseline signal (see ``NSR``). If False, return
        the noise-free signal as two components (x, y).
    trigger : bool, default False
        If True, keep only the time points listed in ``iStrig`` (e.g. to
        model triggered acquisitions). If ``iStrig`` is None, no time points
        are removed.
    calibrate : bool, default False
        If False, ``S0`` is an input and the signal is ``S0 * M_xy``. If True,
        the signal is first computed with ``S0 = 1``, then ``S0`` is estimated
        as the mean ratio between the measured calibration signal ``Scal`` and
        the simulated signal at time points ``iScal``, and returned as an
        output. The signal is then rescaled by this ``S0``.

    Inputs
    ------
    tM : array_like, shape (times,)
        Time points of the magnetization.
    M : ndarray, shape (channels, components, compartments, times)
        Magnetization. Only the first two components (x, y) are used; they
        are summed over compartments.
    S0 : array_like
        Equilibrium signal. Required if ``calibrate`` is False.
    NSR : float
        Noise-to-signal ratio. The standard deviation of the Rician noise is
        ``NSR * Sb``, where ``Sb`` is the baseline signal: the magnitude
        signal at the first time point, averaged over channels and taken
        before any triggering. Noise is therefore proportional to the
        baseline signal, and is zero if the baseline is zero. When
        ``calibrate`` is True, the noise is added to the normalized
        (``S0 = 1``) signal and scales with ``S0`` afterwards, so the
        relative noise level is preserved. Required if ``magnitude`` is True.
    iScal : ndarray of int
        Indices of the time points used for calibration. Required if
        ``calibrate`` is True. If ``trigger`` is also True, the indices refer
        to the time points that remain after triggering.
    Scal : ndarray, shape (channels, components, len(iScal))
        Measured signal at the calibration time points. Required if
        ``calibrate`` is True.
    iStrig : ndarray of int or bool
        Indices (or mask) of the time points to keep. Required if ``trigger``
        is True.

    Outputs
    -------
    tS : ndarray, shape (times,)
        Time points of the signal (a subset of ``tM`` if ``trigger`` is True).
    S : ndarray, shape (channels, components, times)
        Signal, with one component if ``magnitude`` is True, otherwise two.
    S0 : float
        Estimated equilibrium signal. Only returned if ``calibrate`` is True.

    Raises
    ------
    ValueError
        If ``M`` is not a 4D array.
    """
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
    _all_inputs = {'M', 'S0', 'NSR', 'iScal', 'Scal', 'tM', 'iStrig'}
    _all_outputs = {'S0', 'tS', 'S'}

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
            noise_sdev = p['NSR'] * Sb
            S = signal_rice(S, noise_sdev)
            # (channels, 1, times)

        if self.config['trigger']:
            if p['iStrig'] is not None:
                accept = p['iStrig']
                n = np.size(accept)
                tS = tS[:n][accept]
                S = S[:, :, :n][:, :, accept]
                # (channels, components, times)

        if self.config['calibrate']:
            s_cal_norm = S[:, :, p['iScal']]
            s_cal = p['Scal']
            nozero = np.where(s_cal_norm != 0)
            results['S0'] = np.mean(s_cal[nozero] / s_cal_norm[nozero])
            S *= results['S0']

        results |= {'tS': tS, 'S': S}  # (channels, components, times)
        return self.map_results(results)
    
    def test_data(self, nt=5, nch=1):
        data = self.init_data()
        n_channels = nch
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            'tM': np.zeros(nt),
            'M': np.ones((n_channels, 3, 1, nt)), # (channels, components, compartments, times)
            'iScal': np.zeros(n0, dtype=int),
            'Scal': Scal, 
            'iStrig': np.zeros(n0, dtype=int),
        }
        return self.input_data(data)


# +--------------------------------------------------------------------------------------------------+
# |                               RelaxToSignal - all configs (n = 6)                                |
# +-----------+-------------------------------------------------------------------------+------------+
# | Key       | Values                                                                  | Default    |
# +-----------+-------------------------------------------------------------------------+------------+
# | sequence  | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,       | 3D-SPGR-SS |
# |           | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS, 3D-PR-SPGR,  |            |
# |           | 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR,    |            |
# |           | 3D-SR-SPGR-SS, 3D-SR-SS, ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS              |            |
# | tof_corr  | False, True                                                             | False      |
# | inflow    | inlet, none, pool                                                       | none       |
# | magnitude | False, True                                                             | True       |
# | trigger   | False, True                                                             | False      |
# | calibrate | False, True                                                             | False      |
# +--------------------------------------------------------------------------------------------------+

# +---------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                     RelaxToSignal - all inputs (n = 35)                                                     |
# +--------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Key    | Unit       | Name                                                    | Group           | Init  | Bounds        | DICOM | OSIPI     |
# +--------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | NSR    |            | noise-to-signal ratio                                   | Signal          | 0.0   | (0, 100000.0) |       |           |
# | S0     | a.u.       | signal scaling factor                                   | Signal          | 1.0   | (0, 5)        |       | Q.MS1.010 |
# | Scal   | a.u.       | calibration signal                                      | Signal          | 1.0   | (0, 5)        |       | Q.MS1.002 |
# | iScal  |            | indices of calibration signal                           | Signal          | 0     |               |       |           |
# | iStrig |            | indices of the signal trigger                           | Signal          | None  |               |       |           |
# +--------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | FA     | deg        | flip angle                                              | Sequence        | 15    | (0, 180)      |       |           |
# | Nk0    |            | number of acquired phase lines to the center of k-space | Sequence        | 64    | (0, 1000)     |       |           |
# | Nph    |            | number of acquired phase lines in k-space               | Sequence        | 128   | (0, 1000)     |       |           |
# | Nz     |            | number of slices in a multi-slice acquisition           | Sequence        | 64    | (0, 1000)     |       |           |
# | PA     | deg        | preparation Pulse Flip Angle                            | Sequence        | 90    | (0, 180)      |       |           |
# | SA     | deg        | saturation Slab Flip Angle                              | Sequence        | 0     | (0, 180)      |       |           |
# | TA     | sec        | acquisition time                                        | Sequence        | 2.0   | (0, 30)       |       |           |
# | TD     | sec        | prepulse delay                                          | Sequence        | 0.05  | (0, 1)        |       |           |
# | TE     | sec        | echo time                                               | Sequence        | 0.001 | (0, 10)       |       |           |
# | TE1    | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001 | (0, 1)        |       |           |
# | TE2    | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005 | (0, 1)        |       |           |
# | TP     | sec        | preparation delay                                       | Sequence        | 0.05  | (0, 1)        |       |           |
# | TR     | sec        | repetition time                                         | Sequence        | 0.005 | (0, 1)        |       |           |
# | iz     |            | slice number in a multi-slice acquisition               | Sequence        | 0     | (0, 1000)     |       |           |
# | tacq   | sec        | acquisition duration                                    | Sequence        | 240   | (0, 10000.0)  |       |           |
# | tstart | sec        | start of the acquisition                                | Sequence        | 0     | (0, 10000.0)  |       |           |
# +--------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | B1corr |            | B1-correction factor                                    | Electromagnetic | 1     | (0, 5)        |       |           |
# | Mzi    | A/cm       | longitudinal inlet magnetization                        | Electromagnetic | 1     | (0, 5)        |       |           |
# | R1     | Hz         | tissue R1                                               | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | R1i    | Hz         | inlet R1                                                | Electromagnetic | 0.65  | (0, 5)        |       |           |
# | R2     | Hz         | tissue R2                                               | Electromagnetic | 2.0   | (0, 5)        |       |           |
# | R2s    | Hz         | tissue R2*                                              | Electromagnetic | 20    | (0, 5)        |       |           |
# | me     | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1     | (0, 5)        |       |           |
# | tMi    | sec        | inlet magnetization time points                         | Electromagnetic | 0.0   |               |       |           |
# | tR     | sec        | relaxation rate time points                             | Electromagnetic | 0.0   |               |       |           |
# +--------+------------+---------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Fwi    | mL/sec/cm3 | inflow in all water compartments                        | Physiological   | 0.02  | (0, 1)        |       |           |
# | Kw     | mL/sec/cm3 | water exchange matrix                                   | Physiological   | 0     | (0, 1)        |       |           |
# | TF     | sec        | inflow time                                             | Physiological   | 0.5   | (0, 10)       |       |           |
# | inlets |            | water inlet compartments                                | Physiological   | (0,)  |               |       |           |
# | vw     | mL/cm3     | water volume fraction                                   | Physiological   | 1     | (0, 1)        |       |           |
# +---------------------------------------------------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------+
# |                             RelaxToSignal - all outputs (n = 5)                              |
# +-----+------+---------------------------+-----------------+------+--------+-------+-----------+
# | Key | Unit | Name                      | Group           | Init | Bounds | DICOM | OSIPI     |
# +-----+------+---------------------------+-----------------+------+--------+-------+-----------+
# | S   | a.u. | signal                    | Signal          | 1.0  | (0, 5) |       |           |
# | S0  | a.u. | signal scaling factor     | Signal          | 1.0  | (0, 5) |       | Q.MS1.010 |
# | tS  | sec  | signal time points        | Signal          | 0.0  |        |       |           |
# +-----+------+---------------------------+-----------------+------+--------+-------+-----------+
# | M   | A/cm | magnetization             | Electromagnetic | 1    | (0, 5) |       |           |
# | tM  | sec  | magnetization time points | Electromagnetic | 0.0  |        |       |           |
# +----------------------------------------------------------------------------------------------+


class RelaxToSignal(Module): 
    """Compute the MR signal from tissue relaxation rates.

    Chains two modules: :class:`Magnetization`, which computes the
    magnetization time course ``M`` for a given acquisition sequence, and
    :class:`Signal`, which converts ``M`` into the measured signal ``S``
    (magnitude and noise, triggering, calibration). The configurations of
    both modules are combined. ``M`` and ``tM`` are produced by the first
    stage and consumed by the second, so they are outputs of this module
    and not inputs.

    The tables below list *all* configurations, inputs and outputs. Which
    of them are active depends on the configuration. To list the ones that
    apply to a given instance, use its ``inputs()`` and ``outputs()``
    methods.

    Parameters
    ----------
    imap, omap, iomap, cmap : dict, optional
        Optional mappings for input, output, input and output, and
        configuration names (see :class:`Module`).
    **config
        Configuration options as listed under *Configurations*, for example
        ``sequence='2D-SPGR'`` or ``calibrate=True``. Options that are not
        specified take their default value.

    See Also
    --------
    Magnetization : First stage, relaxation rates to magnetization.
    Signal : Second stage, magnetization to signal.

    Notes
    -----
    .. rubric:: Configurations

    .. list-table::
       :header-rows: 1
       :widths: 12 50 14 24

       * - Key
         - Values
         - Default
         - Effect
       * - ``sequence``
         - ``2D-DE-EPI``, ``2D-GE-EPI``, ``2D-SE-EPI``, ``2D-SPGR``,
           ``2D-SPGR-SS``, ``2D-SR-SPGR``, ``3D-DE-EPI``, ``3D-GE-EPI``,
           ``3D-IR-SPGR``, ``3D-IR-SPGR-SS``, ``3D-IR-SS``, ``3D-PR-SPGR``,
           ``3D-PR-SPGR-SS``, ``3D-PR-SS``, ``3D-SE-EPI``, ``3D-SPGR``,
           ``3D-SPGR-SS``, ``3D-SR-SPGR``, ``3D-SR-SPGR-SS``, ``3D-SR-SS``,
           ``ZTE-3D-IR-SPGR-SS``, ``ZTE-3D-SPGR-SS``
         - ``3D-SPGR-SS``
         - Acquisition sequence.
       * - ``tof_corr``
         - ``False``, ``True``
         - ``False``
         - Time-of-flight correction (see :class:`Magnetization`).
       * - ``inflow``
         - ``inlet``, ``none``, ``pool``
         - ``none``
         - Treatment of inflow (see :class:`Magnetization`).
       * - ``magnitude``
         - ``False``, ``True``
         - ``True``
         - Return the magnitude signal with Rician noise (``True``) or the
           noise-free (x, y) signal (``False``).
       * - ``trigger``
         - ``False``, ``True``
         - ``False``
         - Keep only the time points in ``iStrig``.
       * - ``calibrate``
         - ``False``, ``True``
         - ``False``
         - Estimate ``S0`` from the calibration signal ``Scal`` instead of
           taking it as input.

    .. rubric:: Inputs

    Signal inputs:

    .. csv-table::
       :header: Key | Unit | Name | Default | Bounds | OSIPI | Used when
       :delim: |

       NSR | | noise-to-signal ratio | 0.0 | (0, 100000.0) | | magnitude=True
       S0 | a.u. | signal scaling factor | 1.0 | (0, 5) | Q.MS1.010 | calibrate=False
       Scal | a.u. | calibration signal | 1.0 | (0, 5) | Q.MS1.002 | calibrate=True
       iScal | | indices of calibration signal | 0 | | | calibrate=True
       iStrig | | indices of the signal trigger | None | | | trigger=True

    Sequence inputs:

    .. csv-table::
       :header: Key | Unit | Name | Default | Bounds
       :delim: |

       FA | deg | flip angle | 15 | (0, 180)
       Nk0 | | number of acquired phase lines to the center of k-space | 64 | (0, 1000)
       Nph | | number of acquired phase lines in k-space | 128 | (0, 1000)
       Nz | | number of slices in a multi-slice acquisition | 64 | (0, 1000)
       PA | deg | preparation Pulse Flip Angle | 90 | (0, 180)
       SA | deg | saturation Slab Flip Angle | 0 | (0, 180)
       TA | sec | acquisition time | 2.0 | (0, 30)
       TD | sec | prepulse delay | 0.05 | (0, 1)
       TE | sec | echo time | 0.001 | (0, 10)
       TE1 | sec | first echo time in a multi-echo sequence | 0.001 | (0, 1)
       TE2 | sec | second echo time in a multi-echo sequence | 0.005 | (0, 1)
       TP | sec | preparation delay | 0.05 | (0, 1)
       TR | sec | repetition time | 0.005 | (0, 1)
       iz | | slice number in a multi-slice acquisition | 0 | (0, 1000)
       tacq | sec | acquisition duration | 240 | (0, 10000.0)
       tstart | sec | start of the acquisition | 0 | (0, 10000.0)

    Electromagnetic inputs:

    .. csv-table::
       :header: Key | Unit | Name | Default | Bounds
       :delim: |

       B1corr | | B1-correction factor | 1 | (0, 5)
       Mzi | A/cm | longitudinal inlet magnetization | 1 | (0, 5)
       R1 | Hz | tissue R1 | 0.65 | (0, 5)
       R1i | Hz | inlet R1 | 0.65 | (0, 5)
       R2 | Hz | tissue R2 | 2.0 | (0, 5)
       R2s | Hz | tissue ``R2*`` | 20 | (0, 5)
       me | A cm2/mL | equilibrium magnetization | 1 | (0, 5)
       tMi | sec | inlet magnetization time points | 0.0 |
       tR | sec | relaxation rate time points | 0.0 |

    Physiological inputs:

    .. csv-table::
       :header: Key | Unit | Name | Default | Bounds
       :delim: |

       Fwi | mL/sec/cm3 | inflow in all water compartments | 0.02 | (0, 1)
       Kw | mL/sec/cm3 | water exchange matrix | 0 | (0, 1)
       TF | sec | inflow time | 0.5 | (0, 10)
       inlets | | water inlet compartments | (0,) |
       vw | mL/cm3 | water volume fraction | 1 | (0, 1)

    .. rubric:: Outputs

    .. csv-table::
       :header: Key | Unit | Name | Returned when
       :delim: |

       S | a.u. | signal | always
       tS | sec | signal time points | always
       S0 | a.u. | signal scaling factor | calibrate=True
       M | A/cm | magnetization | always
       tM | sec | magnetization time points | always

    Examples
    --------
    >>> import dcmri
    >>> model = dcmri.RelaxToSignal(sequence='2D-SPGR', calibrate=True)
    >>> inputs = model.inputs()    # inputs needed for this configuration
    >>> outputs = model.outputs()  # outputs returned for this configuration
    """
    configs = Magnetization.configs | Signal.configs
    defaults = Magnetization.defaults | Signal.defaults

    _all_inputs = {'TR', 'TE2', 'Nz', 'Scal', 'TE', 'iStrig', 'FA', 'vw', 'Fwi', 'PA', 'tacq', 'tMi', 'TF', 'TA', 'TD', 'R2', 'iScal', 'TE1', 'SA', 'R1', 'inlets', 'NSR', 'me', 'Kw', 'Mzi', 'Nk0', 'R2s', 'iz', 'B1corr', 'S0', 'R1i', 'TP', 'tstart', 'Nph', 'tR'}
    _all_outputs = {'S', 'tS', 'S0', 'M', 'tM'}

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

    def test_data(self, nc=2, nt=5):
        n_channels = channels(self.config['sequence'])

        p = self.init_data()

        p |= self._magn.test_data(nc, nt)
        p |= self._signal.test_data(nt, n_channels)

        return self.input_data(p)


# +--------------------------------------------------------------------------------------------------+
# |                                ConcToSignal - all configs (n = 9)                                |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, leakage, lin, quad                                           | lin        |
# | inflow         | inlet, none, pool                                                  | none       |
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
# |                                                           ConcToSignal - all inputs (n = 46)                                                          |
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
# | Mzi    | A/cm       | longitudinal inlet magnetization                                  | Electromagnetic | 1     | (0, 5)        |       |           |
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
# | tMi    | sec        | inlet magnetization time points                                   | Electromagnetic | 0.0   |               |       |           |
# +--------+------------+-------------------------------------------------------------------+-----------------+-------+---------------+-------+-----------+
# | Fwi    | mL/sec/cm3 | inflow in all water compartments                                  | Physiological   | 0.02  | (0, 1)        |       |           |
# | Kw     | mL/sec/cm3 | water exchange matrix                                             | Physiological   | 0     | (0, 1)        |       |           |
# | RM     |            | relaxivity mapping                                                | Physiological   |       |               |       |           |
# | TF     | sec        | inflow time                                                       | Physiological   | 0.5   | (0, 10)       |       |           |
# | inlets |            | water inlet compartments                                          | Physiological   | (0,)  |               |       |           |
# | v      | mL/cm3     | volume fraction                                                   | Physiological   | 1     | (0, 1)        |       |           |
# | vw     | mL/cm3     | water volume fraction                                             | Physiological   | 1     | (0, 1)        |       |           |
# +-------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------+
# |                              ConcToSignal - all outputs (n = 10)                               |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-----------+
# | Key | Unit | Name                        | Group           | Init | Bounds | DICOM | OSIPI     |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-----------+
# | S   | a.u. | signal                      | Signal          | 1.0  | (0, 5) |       |           |
# | S0  | a.u. | signal scaling factor       | Signal          | 1.0  | (0, 5) |       | Q.MS1.010 |
# | tS  | sec  | signal time points          | Signal          | 0.0  |        |       |           |
# +-----+------+-----------------------------+-----------------+------+--------+-------+-----------+
# | M   | A/cm | magnetization               | Electromagnetic | 1    | (0, 5) |       |           |
# | R1  | Hz   | tissue R1                   | Electromagnetic | 0.65 | (0, 5) |       |           |
# | R1i | Hz   | inlet R1                    | Electromagnetic | 0.65 | (0, 5) |       |           |
# | R2  | Hz   | tissue R2                   | Electromagnetic | 2.0  | (0, 5) |       |           |
# | R2s | Hz   | tissue R2*                  | Electromagnetic | 20   | (0, 5) |       |           |
# | tM  | sec  | magnetization time points   | Electromagnetic | 0.0  |        |       |           |
# | tR  | sec  | relaxation rate time points | Electromagnetic | 0.0  |        |       |           |
# +------------------------------------------------------------------------------------------------+


class ConcToSignal(Module): 
    configs = ConcToRelax.configs | RelaxToSignal.configs 
    defaults = ConcToRelax.defaults | RelaxToSignal.defaults

    _all_inputs = {'TR', 'Scal', 'R1ib', 'r2se', 'FA', 'r2', 'TD', 'iScal', 'TE1', 'inlets', 'NSR', 'me', 'Kw', 'Mzi', 'iz', 'B1corr', 'S0', 'TP', 'r2sq', 'C', 'TE2', 'Nz', 'R1b', 'v', 'TE', 'RM', 'iStrig', 'r2s', 'vw', 'Fwi', 'tC', 'PA', 'tacq', 'tMi', 'ci', 'TF', 'r1i', 'TA', 'SA', 'r1', 'Nk0', 'r2sv', 'R2b', 'R2sb', 'Nph', 'tstart'}
    _all_outputs = {'S', 'R2', 'R1', 'M', 'R2s', 'tS', 'S0', 'R1i', 'tM', 'tR'}

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

    def test_data(self, nc=2, nt=5):
        p = self.init_data()

        p |= self._conc_to_relax.test_data(nc, nt)
        p |= self._relax_to_signal.test_data(nc, nt)

        return self.input_data(p)
