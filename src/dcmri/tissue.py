"""PK models built from PK blocks defined in dcmri.pk"""

import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils
import dcmri.rel as rel
import dcmri.sig as sig



def params_tissue(kinetics='2CX', water_exchange='RR') -> list:
    """Parameters characterizing a 2-site exchange tissue. 
    For more detail see :ref:`two-site-exchange`.

    Args:
        kinetics (str, optional): Tracer-kinetic regime. Possible values are
         '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to '2CX'.
        water_exchange (str, optional): Water exchange regime, Any combination
          of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'RR'.

    Returns: 
        list: tissue parameters

    Raises:
        ValueError: if the configuration is not recognized.

    Example:

        Print the parameters of a HFU tissue with restricted water 
        exchange:

        >>> import dcmri as dc

        >>> dc.params_tissue('HFU', 'RR')
        ['PSe', 'PSc', 'H', 'vb', 'vi', 'PS']

    Notes:

        Table :ref:`tissue-kinetic-regimes` list the water compartments and free 
        parameters for all configurations. Regimes without water exchange 
        across one or both of the barriers are not listed 
        explicitly (FN, NF, FR, NR and NN). They differ from restricted water 
        exchange only in the sense that the respective water permeabilities 
        *PSe* and/or *PSc* are zero. 

        .. _tissue-kinetic-regimes:
        .. list-table:: **Parameters by configuration** 
            :widths: 15 15 30 40
            :header-rows: 1 

            * - Water exchange
              - Indicator exchange
              - Water compartments
              - Free parameters
            * - **FF**
              - 
              - 
              - 
            * - FF
              - 2CX
              - vb + vi + vc
              - H, vb, vi, Fb, PS
            * - FF
              - 2CU
              - vb + vi + vc
              - H, vb, Fb, PS
            * - FF
              - HF
              - vb + vi + vc
              - H, vb, vi, PS
            * - FF
              - HFU
              - vb + vi + vc
              - H, vb, PS
            * - FF
              - FX
              - vb + vi + vc 
              - H, ve, Fb  
            * - FF
              - NX
              - vb + vi + vc
              - vb, Fb  
            * - FF
              - U
              - vb + vi + vc
              - Fb
            * - FF
              - WV
              - vb + vi + vc
              - H, vi, Ktrans
            * - **RR**
              - 
              - 
              - 
            * - RR
              - 2CX
              - vb, vi, vc
              - PSe, PSc, H, vb, vi, Fb, PS
            * - RR
              - 2CU
              - vb, vi, vc
              - PSe, PSc, H, vb, vi, Fb, PS
            * - RR
              - HF
              - vb, vi, vc
              - PSe, PSc, H, vb, vi, PS
            * - RR
              - HFU
              - vb, vi, vc
              - PSe, PSc, H, vb, vi, PS
            * - RR
              - FX
              - vb, vi, vc 
              - PSe, PSc, H, vb, vi, Fb 
            * - RR
              - NX
              - vb, vi, vc 
              - PSe, vb, vi, Fb  
            * - RR
              - U
              - vb, vi, vc 
              - PSe, vb, vi, Fb 
            * - RR
              - WV
              - vi, vi+vc
              - PSc, H, vi, Ktrans
            * - **RF**
              - 
              - 
              - 
            * - RF
              - 2CX
              - vb, vi+vc
              - PSe, H, vb, vi, Fb, PS
            * - RF
              - 2CU
              - vb, vi+vc
              - PSe, H, vb, Fb, PS
            * - RF
              - HF
              - vb, vi+vc
              - PSe, H, vb, vi, PS
            * - RF
              - HFU
              - vb, vi+vc
              - PSe, H, vb, PS
            * - RF
              - FX
              - vb, vi+vc
              - PSe, H, vb, vi, Fb
            * - RF
              - NX
              - vb, vi+vc
              - PSe, vb, Fb
            * - RF
              - U
              - vb, vi+vc
              - PSe, vb, Fb 
            * - RF
              - WV
              - vi+vc
              - H, vi, Ktrans
            * - **FR**
              - 
              - 
              -  
            * - FR
              - 2CX
              - vb+vi, vc
              - PSc, H, vb, vi, Fb, PS
            * - FR
              - 2CU
              - vb+vi, vc
              - PSc, H, vb, vi, Fb, PS
            * - FR
              - HF
              - vb+vi, vc
              - PSc, H, vb, vi, PS
            * - FR
              - HFU
              - vb+vi, vc
              - PSc, H, vb, vi, PS
            * - FR
              - FX
              - vb+vi, vc 
              - PSc, H, vb, vi, Fb
            * - FR
              - NX
              - vb+vi, vc 
              - PSc, vb, vi, Fb
            * - FR
              - U
              - vb+vi, vc 
              - PSc, vc, Fb
            * - FR
              - WV
              - vi, vc
              - PSc, H, vi, Ktrans
    """

    if kinetics not in ['2CX', '2CU', 'HF', 'HFU', 'WV', 'FX', 'NX', 'U']:
        raise ValueError(
            'The value ' + str(kinetics) + ' for the kinetics argument \
            is not recognised. \n Possible values are 2CX, 2CU, HF, HFU, WV, \
            FX, NX, U.'
        )

    if water_exchange not in ['FF', 'NF','RF', 'FN', 'NN', 
                              'RN', 'FR', 'NR', 'RR']:
        raise ValueError(
            'The value ' + str(water_exchange) + ' of the \
            water_exchange argument is not recognised.\n It must be a \
            2-element string composed of characters N, F, R.'
        )
    
    pars = _relax_pars(kinetics, water_exchange)

    if water_exchange in ['FF', 'NN', 'NF', 'FN']:
        return pars
    if water_exchange in ['NR', 'FR']:
        return ['PSc'] + pars
    if water_exchange in ['RN', 'RF']:
        if kinetics == 'WV':
            return pars
        else:
            return ['PSe'] + pars
    if water_exchange == 'RR':
        if kinetics == 'WV':
            return ['PSc'] + pars
        else:
            return ['PSe', 'PSc'] + pars


def _relax_pars(kin, wex) -> list:

    if kin == '2CX':
        return ['H', 'vb', 'vi', 'Fb', 'PS']
    if kin == 'HF':
        return ['H', 'vb', 'vi', 'PS']
    if kin == 'WV':
        return ['H', 'vi', 'Ktrans']

    if wex == 'FF':
        return _kin_pars(kin)

    if wex in ['RR', 'NN', 'NR', 'RN']:

        if kin == '2CU':
            return ['H', 'vb', 'vi', 'Fb', 'PS']
        if kin == 'HFU':
            return ['H', 'vb', 'vi', 'PS']
        if kin == 'FX':
            return ['H', 'vb', 'vi', 'Fb']
        if kin == 'NX':
            return ['vb', 'vi', 'Fb']
        if kin == 'U':
            return ['vb', 'vi', 'Fb']

    if wex in ['RF', 'NF']:

        if kin == '2CU':
            return ['H', 'vb', 'Fb', 'PS']
        if kin == 'HFU':
            return ['H', 'vb', 'PS']
        if kin == 'FX':
            return ['H', 'vb', 'vi', 'Fb']
        if kin == 'NX':
            return ['vb', 'Fb']
        if kin == 'U':
            return ['vb', 'Fb']

    if wex in ['FR', 'FN']:

        if kin == '2CU':
            return ['H', 'vb', 'vi', 'Fb', 'PS']
        if kin == 'HFU':
            return ['H', 'vb', 'vi', 'PS']
        if kin == 'FX':
            return ['H', 'vb', 'vi', 'Fb']
        if kin == 'NX':
            return ['vb', 'vi', 'Fb']
        if kin == 'U':
            return ['vc', 'Fb']

def _kin_pars(kin):

    if kin == '2CX':
        return ['H', 'vb', 'vi', 'Fb', 'PS']
    if kin == 'HF':
        return ['H', 'vb', 'vi', 'PS']
    if kin == 'WV':
        return ['H', 'vi', 'Ktrans']
    if kin == '2CU':
        return ['H', 'vb', 'Fb', 'PS']
    if kin == 'HFU':
        return ['H', 'vb', 'PS']
    if kin == 'FX':
        return ['H', 've', 'Fb']
    if kin == 'NX':
        return ['vb', 'Fb']
    if kin == 'U':
        return ['Fb']


def signal_tissue(
        ca: np.ndarray, R10: float, r1: float, t=None, dt=1.0, 
        kinetics='2CX', water_exchange='FF', sequence=None, inflow=None,
        sum=True, **params) -> np.ndarray:
    
    """Signal for a 2-site exchange tissue. For more detail see
    :ref:`two-site-exchange`.

    Args:
        ca (array-like): concentration in the blood of the arterial input.
        R10 (float): precontrast relaxation rate. The tissue is assumed to be 
          in fast exchange before injection of contrast agent.
        r1 (float): contrast agent relaxivity. 
        t (array_like, optional): the time points in sec of the input function 
          *ca*. If *t* is not provided, the time points are assumed to be 
          uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          explicity provided. Defaults to 1.0.
        kinetics (str, optional): Tracer-kinetic model. Possible values are
         '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to '2CX'.
        water_exchange (str, optional): Water exchange regime, Any combination
          of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'FF'.
        sequence (dict): the sequence model and its parameters. The 
          dictionary has one required key 'model' which specifies the signal 
          model. Currently either 'SS' or 'SR'. The other keys are the values 
          of the signal parameter, which depend on the model. See table 
          :ref:`Tissue-signal-parameters` for detail. 
        inflow (dict, optional): inflow model. If not provided, the in- and 
          outflow of magnetization is ignored. To include 
          inflow effects, **inflow** must be dictionary with the signal model 
          parameters for the arterial input. For the 'SS' signal model, 
          required parameters are 'R10a' and 'B1corr_a'. Defaults to None.
        sum (bool, optional): If True, the total signal is returned. If False, 
          the signal in individual tissue compartments is returned. Defaults to
          True.
        params (dict): model parameters. See :ref:`Tissue-signal-parameters` 
          for more detail. Note: the tissue parameters are keyword 
          arguments for convenience, but a value is required.

    Raises:
        ValueError: if a required parameter has no value assigned.
        NotImplementedError: if a combination of regimes is not yet 
          implemented. Currently the sequence type 'SR' only accepts fast 
          water exchange 'FF'.

    Returns:
        ndarray: tissue signal. 
          In the fast water exchange limit, or whenever 
          sum = True, the signal is a 1D array. In all other situations, the 
          signal is a 2D-array with dimensions (k,n), where k is the number 
          of compartments and n is the number of time points in ca. 

    Example:

        We verify that the effect of inflow is negligible in a steady state 
        sequence:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Define constants and model parameters: 

        >>> R10, r1 = 1, 5000
        >>> seq = {'model': 'SS', 'S0':1, 'FA':15, 'TR': 0.001, 'B1corr':1}
        >>> pars = {
        >>>     'sequence':seq, 'kinetics':'2CX', 'water_exchange':'NN', 
        >>>     'H':0.045, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005} 
        >>> inflow = {'R10a': 0.7, 'B1corr_a':1}

        Generate arterial blood concentrations:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)/(1-0.45) 

        Calculate the signal with and without inflow:

        >>> Sf = dc.signal_tissue(ca, R10, r1, t=t, inflow=inflow, **pars)
        >>> Sn = dc.signal_tissue(ca, R10, r1, t=t, **pars)

        Compare them in a plot:

        >>> plt.figure()
        >>> plt.plot(t/60, Sn, label='Without inflow correction', linewidth=3)
        >>> plt.plot(t/60, Sf, label='With inflow correction')
        >>> plt.xlabel('Time (min)')
        >>> plt.ylabel('Concentration (mM)')
        >>> plt.legend()
        >>> plt.show()

    Notes:

        .. _Tissue-signal-parameters:
        .. list-table:: **Tissue signal parameters**
            :widths: 20 30 30
            :header-rows: 1

            * - Parameters
              - When to use
              - Further detail
            * - Fb, PS, Ktrans, vb, H, vi,
                ve, vc, PSe, PSc.
              - Depends on **kinetics** and **water_exchange**
              - :ref:`tissue-kinetic-regimes`
            * - S0, FA, TR, B1corr
              - Always
              - :ref:`params-per-sequence`
            * - TP, TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - R10a, B1corr_a
              - If **inflow** is not None
              - :ref:`relaxation-params`, :ref:`params-per-sequence`

    """
    if sequence is None:
        raise ValueError(
            'sequence is required. Please specify a model \
             and appropriate sequence parameters.')
    
    R1, v, PSw = relax_tissue(
        ca, R10, r1, t=t, dt=dt, 
        kinetics=kinetics, water_exchange=water_exchange, **params)
    
    if sequence['model'] == 'SS':
        if inflow is None:
            Ji = None
        else:
            if kinetics != '2CX':
                raise ValueError('Inflow correction is currently only \
                                 available for 2CX tissues')
            FAa = inflow['B1corr_a'] * sequence['FA']
            R1a = rel.relax(ca, inflow['R10a'], r1)
            na = sig.signal_ss(R1a, 1, sequence['TR'], FAa)
            if np.isscalar(v):
                Ji = PSw*na
            else:
                Ji = np.zeros((len(v), len(na)))
                Ji[0, :] = PSw[0,0]*na
        FA = sequence['B1corr'] * sequence['FA']
        return sig.signal_ss(
            R1, sequence['S0'], sequence['TR'], FA,
            v=v, PSw=PSw, Ji=Ji, sum=sum)
    
    elif sequence['model'] == 'SR':
        if inflow is not None:
            raise NotImplementedError(
                'Inflow correction is currently not \
                available for signal model SR')
        if not sum:
            raise NotImplementedError(
                "Separate signals for signal model SR \
                are not yet implemented.")
        FA = sequence['B1corr'] * sequence['FA']
        return sig.signal_sr(
            R1, sequence['S0'], sequence['TR'], FA, sequence['TC'], 
            sequence['TP'], v=v, PSw=PSw)
    


def relax_tissue(ca: np.ndarray, R10: float, r1: float, t=None, dt=1.0, 
                 kinetics='2CX', water_exchange='FF', **params):
    """Free relaxation rates for a 2-site exchange tissue. For more detail see
    :ref:`two-site-exchange`.

    Note: the free relaxation rates are the relaxation rates of the tissue 
    compartments in the absence of water exchange between them.

    Args:
        ca (array-like): concentration in the blood of the arterial input.
        R10 (float): precontrast relaxation rate. The tissue is assumed to be 
          in fast exchange before injection of contrast agent.
        r1 (float): contrast agent relaxivity. 
        t (array_like, optional): the time points in sec of the input function 
          *ca*. If *t* is not provided, the time points are assumed to be 
          uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          explicity provided. Defaults to 1.0.
        kinetics (str, optional): Tracer-kinetic model. Possible values are
         '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to '2CX'.
        water_exchange (str, optional): Water exchange regime, Any combination
          of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'FF'.
        params (dict): values for the parameters of the tissue,
          specified as keyword parameters. See table :ref:`tissue-kinetic-regimes` 
          for more detail on the parameters that are relevant in each regime. 

    Returns:
        numpy.ndarray: relaxation rates
          In the fast water exchange limit, the 
          relaxation rates are a 1D array. In all other situations, 
          relaxation rates are a 2D-array with dimensions (k,n), where k is 
          the number of compartments and n is the number of time points 
          in ca.
        volume fractions
          the volume fractions of the tissue compartments. 
          Returns None in 'FF' regime. 
        water flows
          2D array with water exchange 
          rates between tissue compartments. Returns None in 'FF' regime.

    Example:

        Compare the free relaxation rates without water exchange against 
        relaxation rates in fast exchange:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define constants and model parameters: 

        >>> R10, r1 = 1/dc.T1(), dc.relaxivity()     
        >>> pf = {'H':0.5, 'vb':0.05, 'vi':0.3, 'Fb':0.01, 'PS':0.005}   
        >>> pn = {'H':0.5, 'vb':0.1, 'vi':0.3, 'Fb':0.01, 'PS':0.005}

        Calculate tissue relaxation rates without water exchange, 
        and also in the fast exchange limit for comparison:

        >>> R1f, _, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='FF', **pf)
        >>> R1n, _, _ = dc.relax_tissue(ca, R10, r1, t=t, water_exchange='NN', **pn)

        Plot the relaxation rates in the three compartments, and compare 
        against the fast exchange result:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        Plot restricted water exchange in the left panel:

        >>> ax0.set_title('Restricted water exchange')
        >>> ax0.plot(t/60, R1n[0,:], linestyle='-', 
        >>>          linewidth=2.0, color='darkred', label='Blood')
        >>> ax0.plot(t/60, R1n[1,:], linestyle='-', 
        >>>          linewidth=2.0, color='darkblue', label='Interstitium')
        >>> ax0.plot(t/60, R1n[2,:], linestyle='-', 
        >>>          linewidth=2.0, color='grey', label='Cells')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Compartment relaxation rate (1/sec)')
        >>> ax0.legend()

        Plot fast water exchange in the right panel:

        >>> ax1.set_title('Fast water exchange')
        >>> ax1.plot(t/60, R1f, linestyle='-', 
        >>>          linewidth=2.0, color='black', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue relaxation rate (1/sec)')
        >>> ax1.legend()
        >>> plt.show()

    """

    # Check configuration
    if kinetics not in ['U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU', '2CX']:
        raise ValueError(
            "Kinetic model '" + str(kinetics) + 
            "' is not recognised.\n Possible values are: " +
            "'U', 'FX', 'NX', 'WV', 'HFU', 'HF', '2CU' and '2CX'."
        )
    
    if (water_exchange[0] not in ['F', 'R', 'N']) or (
        water_exchange[1] not in ['F', 'R', 'N']):
        raise ValueError(
            "Water exchange regime '" +
            str(water_exchange) + "' is not recognised.\n" + 
            "Possible values are: 'FF','RF','NF','FR','RR','NR','FN','RN','NN'"
        )

    if water_exchange[0] == 'N':
        if kinetics != 'WV':
            params['PSe'] = 0

    if water_exchange[1] == 'N':
        params['PSc'] = 0    

    wex = water_exchange.replace('N','R')

    # Distribute cases
    if wex == 'FF':

        if kinetics == 'U':
            return _relax_u_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_ff(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_ff(ca, R10, r1, t=t, dt=dt, **params)

    elif wex == 'RF':

        if kinetics == 'U':
            return _relax_u_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_rf(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_rf(ca, R10, r1, t=t, dt=dt, **params)

    elif wex == 'FR':

        if kinetics == 'U':
            return _relax_u_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_fr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_fr(ca, R10, r1, t=t, dt=dt, **params)

    elif wex == 'RR':

        if kinetics == 'U':
            return _relax_u_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'FX':
            return _relax_fx_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'NX':
            return _relax_nx_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'WV':
            return _relax_wv_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HFU':
            return _relax_hfu_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == 'HF':
            return _relax_hf_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CU':
            return _relax_2cu_rr(ca, R10, r1, t=t, dt=dt, **params)
        elif kinetics == '2CX':
            return _relax_2cx_rr(ca, R10, r1, t=t, dt=dt, **params)


def _c(C,v):
    if v==0:
        # In this case the result does not matter
        return C*0
    else:
        return C/v
    

# FF

# For water flow modelling
def _relax_2cx_ff(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vi=None, vb=None, Fb=None, PS=None):
    C = _conc_2cx(ca, t=t, dt=dt, vi=vi, H=H, vb=vb, Fb=Fb, PS=PS)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, Fb

def __relax_2cx_ff(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vi=None, vb=None, Fb=None, PS=None):
    C = _conc_2cx(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_2cu_ff(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, Fb=None, PS=None):
    C = _conc_2cu(ca, t=t, dt=dt, H=H, vb=vb, Fb=Fb, PS=PS)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_hf_ff(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vi=None, vb=None, PS=None):
    C = _conc_hf(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, PS=PS)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_hfu_ff(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, PS=None):
    C = _conc_hfu(ca, t=t, dt=dt, H=H, vb=vb, PS=PS)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_nx_ff(ca, R10, r1, t=None, dt=1.0, 
                 vb=None, Fb=None):
    C = _conc_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_wv_ff(ca, R10, r1, t=None, dt=1.0,
                 H=None, vi=None, Ktrans=None):
    C = _conc_wv(ca, t=t, dt=dt, H=H, vi=vi, Ktrans=Ktrans)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_u_ff(ca, R10, r1, t=None, dt=1.0, 
                Fb=None):
    C = _conc_u(ca, t=t, dt=dt, Fb=Fb)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_fx_ff(ca, R10, r1, t=None, dt=1.0, 
                 H=None, ve=None, Fb=None):
    C = _conc_fx(ca / (1-H), t=t, dt=dt, H=H, ve=ve, Fb=Fb)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

# FR

# For water flow modelling
def _relax_2cx_fr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, Fb=None, PS=None, PSc=None):
    C = _conc_2cx(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    v = [vb+vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[Fb, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def __relax_2cx_fr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, Fb=None, PS=None, PSc=None):
    C = _conc_2cx(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    v = [vb+vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_2cu_fr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vi=None, vb=None, Fb=None, PS=None, PSc=None):
    C = _conc_2cu(ca, t=t, dt=dt, H=H, vb=vb, Fb=Fb, PS=PS)
    vc = 1-vb-vi
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hf_fr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, PS=None, PSc=None):
    C = _conc_hf(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, PS=PS)
    v = [vb+vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hfu_fr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vi=None, vb=None, PS=None, PSc=None):
    C = _conc_hfu(ca, t=t, dt=dt, H=H, vb=vb, PS=PS)
    vc = 1-vb-vi
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_wv_fr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vi=None, Ktrans=None, PSc=None):
    C = _conc_wv(ca, t=t, dt=dt, H=H, vi=vi, Ktrans=Ktrans)
    v = [vi, 1-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_fx_fr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, Fb=None, PSc=None):
    vp = vb*(1-H)
    ve = vi+vp
    vc = 1-vb-vi
    C = _conc_fx(ca, t=t, dt=dt, H=H, ve=ve, Fb=Fb)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_nx_fr(ca, R10, r1, t=None, dt=1.0, 
                 vi=None, vb=None, Fb=None, PSc=None):
    vc = 1-vb-vi
    C = _conc_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_u_fr(ca, R10, r1, t=None, dt=1.0, 
                vc=None, Fb=None, PSc=None):
    C = _conc_u(ca, t=t, dt=dt, Fb=Fb)
    v = [1-vc, vc]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

# RF

# For water flow modelling
def _relax_2cx_rf(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, Fb=None, PS=None, PSe=None):
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, 
                  H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[Fb, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def __relax_2cx_rf(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, Fb=None, PS=None, PSe=None):
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, 
                  H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_2cu_rf(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, Fb=None, PS=None, PSe=None):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, 
                  H=H, vb=vb, Fb=Fb, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hf_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, PS=None, PSe=None):
    C = _conc_hf(ca, t=t, dt=dt, sum=False, H=H, vb=vb, vi=vi, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hfu_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, PS=None, PSe=None):
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, H=H, vb=vb, PS=PS)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_wv_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vi=None, Ktrans=None):
    C = _conc_wv(ca, t=t, dt=dt, H=H, vi=vi, Ktrans=Ktrans)
    R1 = rel.relax(C, R10, r1)
    return R1, 1, None

def _relax_fx_rf(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, Fb=None, PSe=None):
    vp = vb * (1-H)
    ve = vp + vi
    C = _conc_fx(ca, t=t, dt=dt, H=H, ve=ve, Fb=Fb)
    v = [vb, 1-vb]
    Cp = C*vp/ve
    Ci = C*vi/ve
    R1 = (
        rel.relax(_c(Cp, v[0]), R10, r1),
        rel.relax(_c(Ci, v[1]), R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_nx_rf(ca, R10, r1, t=None, dt=1.0, 
                 vb=None, Fb=None, PSe=None):
    C = _conc_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_u_rf(ca, R10, r1, t=None, dt=1.0, 
                vb=None, Fb=None, PSe=None):
    C = _conc_u(ca, t=t, dt=dt, Fb=Fb)
    v = [vb, 1-vb]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1),
    )
    PSw = [[0, PSe], [PSe, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)


# RR

# For water flow modelling
def _relax_2cx_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, 
                  Fb=None, PS=None, PSe=None, PSc=None):
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, 
                  H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1),
        rel.relax(ca*0, R10, r1),
    )
    PSw = [[Fb, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)


def __relax_2cx_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, 
                  Fb=None, PS=None, PSe=None, PSc=None):
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, 
                  H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_2cu_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, 
                  Fb=None, PS=None, PSe=None, PSc=None):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, H=H, vb=vb, Fb=Fb, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hf_rr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, PS=None, PSe=None, PSc=None):
    C = _conc_hf(ca, t=t, dt=dt, sum=False, H=H, vb=vb, vi=vi, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_hfu_rr(ca, R10, r1, t=None, dt=1.0, 
                  H=None, vb=None, vi=None, PS=None, 
                  PSe=None, PSc=None):
    C = _conc_hfu(ca, t=t, dt=dt, sum=False, H=H, vb=vb, PS=PS)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C[0,:], v[0]), R10, r1),
        rel.relax(_c(C[1,:], v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_wv_rr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vi=None, Ktrans=None, PSc=None):
    C = _conc_wv(ca, t=t, dt=dt, H=H, vi=vi, Ktrans=Ktrans)
    v = [vi, 1-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSc], [PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_fx_rr(ca, R10, r1, t=None, dt=1.0, 
                 H=None, vb=None, vi=None, Fb=None, 
                 PSe=None, PSc=None):
    vp = vb * (1-H)
    ve = vp + vi
    C = _conc_fx(ca, t=t, dt=dt, H=H, ve=ve, Fb=Fb)
    v = [vb, vi, 1-vb-vi]
    Cp = C*vp/ve
    Ci = C*vi/ve
    R1 = (
        rel.relax(_c(Cp, v[0]), R10, r1),
        rel.relax(_c(Ci, v[1]), R10, r1), 
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_nx_rr(ca, R10, r1, t=None, dt=1.0, 
                 vb=None, vi=None, Fb=None, 
                 PSe=None, PSc=None):
    C = _conc_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
    v = [vb, vi, 1-vb-vi]
    R1 = (
        rel.relax(_c(C, v[0]), R10, r1),
        rel.relax(ca*0, R10, r1),
        rel.relax(ca*0, R10, r1), 
    )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)

def _relax_u_rr(ca, R10, r1, t=None, dt=1.0, 
                vb=None, vi=None, Fb=None, PSe=None, PSc=None):
    v = [vb, vi, 1-vb-vi]
    if Fb==0:
        R1 = (
            rel.relax(ca*0, R10, r1),
            rel.relax(ca*0, R10, r1), 
            rel.relax(ca*0, R10, r1), 
        )
    else:
        C = _conc_u(ca, t=t, dt=dt, Fb=Fb)
        R1 = (
            rel.relax(_c(C, v[0]), R10, r1),
            rel.relax(ca*0, R10, r1), 
            rel.relax(ca*0, R10, r1), 
        )
    PSw = [[0, PSe, 0], [PSe, 0, PSc], [0, PSc, 0]]
    return np.stack(R1), np.array(v), np.array(PSw)



def conc_tissue(ca: np.ndarray, t=None, dt=1.0, kinetics='2CX', sum=True, 
                **params) -> np.ndarray:
    """Tissue concentration in a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points of the input function *ca*. 
          If *t* is not provided, the time points are assumed to be uniformly 
          spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          provided. Defaults to 1.0.
        kinetics (str, optional): Tracer-kinetic model. Possible values are 
          '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U' (see 
          table :ref:`two-site-exchange-kinetics` for detail). Defaults to 
          '2CX'.
        sum (bool, optional): For two-compartment tissues, set to True to 
          return the total tissue concentration, and False to return the 
          concentrations in the compartments separately. In one-compartment 
          tissues this keyword has no effect. Defaults to True.
        params (dict): free model parameters provided as keyword arguments. 
          Possible parameters depend on **kinetics** as detailed in Table 
          :ref:`two-site-exchange-kinetics`. 

    Returns:
        numpy.ndarray: concentration
          If sum=True, or the tissue is one-compartmental, this 
          is a 1D array with the total concentration at each time point. If 
          sum=False this is the concentration in each compartment, and at each 
          time point, as a 2D array with dimensions *(2,k)*, where *k* is the 
          number of time points in *ca*. 

    Raises:
        ValueError: if values are not provided for one or more of the model 
          parameters.

    Example:

        We plot the concentrations of 2CX and WV models with the same values 
        for the shared tissue parameters. 

    .. plot::
        :include-source:

        Start by importing the packages:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define some tissue parameters: 

        >>> p2x = {'H': 0.5, 'vb':0.1, 'vi':0.4, 'Fb':0.02, 'PS':0.005}
        >>> pwv = {'H': 0.5, 'vi':0.4, 'Ktrans':0.005*0.01/(0.005+0.01)}

        Generate plasma and extravascular tissue concentrations with the 2CX 
        and WV models:

        >>> C2x = dc.conc_tissue(ca, t=t, sum=False, kinetics='2CX', **p2x)
        >>> Cwv = dc.conc_tissue(ca, t=t, kinetics='WV', **pwv)

        Compare them in a plot:

        >>> fig, (ax0, ax1) = plt.subplots(1,2,figsize=(12,5))

        Plot 2CX results in the left panel:

        >>> ax0.set_title('2-compartment exchange model')
        >>> ax0.plot(t/60, 1000*C2x[0,:], linestyle='-', linewidth=3.0, 
        >>>          color='darkred', label='Plasma')
        >>> ax0.plot(t/60, 1000*C2x[1,:], linestyle='-', linewidth=3.0, 
        >>>          color='darkblue', 
        >>>          label='Extravascular, extracellular space')
        >>> ax0.plot(t/60, 1000*(C2x[0,:]+C2x[1,:]), linestyle='-', 
        >>>          linewidth=3.0, color='grey', label='Tissue')
        >>> ax0.set_xlabel('Time (min)')
        >>> ax0.set_ylabel('Tissue concentration (mM)')
        >>> ax0.legend()

        Plot WV results in the right panel:

        >>> ax1.set_title('Weakly vascularised model')
        >>> ax1.plot(t/60, Cwv*0, linestyle='-', linewidth=3.0, 
        >>>          color='darkred', label='Plasma')
        >>> ax1.plot(t/60, 1000*Cwv, linestyle='-', 
        >>>          linewidth=3.0, color='grey', label='Tissue')
        >>> ax1.set_xlabel('Time (min)')
        >>> ax1.set_ylabel('Tissue concentration (mM)')
        >>> ax1.legend()
        >>> plt.show()
    """

    if kinetics == 'U':
        return _conc_u(ca, t=t, dt=dt, **params)
    elif kinetics == 'FX':
        return _conc_fx(ca, t=t, dt=dt, **params)
    elif kinetics == 'NX':
        return _conc_nx(ca, t=t, dt=dt, **params)
    elif kinetics == 'WV':
        return _conc_wv(ca, t=t, dt=dt, **params)
    elif kinetics == 'HFU':
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics == 'HF':
        return _conc_hf(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics == '2CU':
        return _conc_2cu(ca, t=t, dt=dt, sum=sum, **params)
    elif kinetics == '2CX':
        return _conc_2cx(ca, t=t, dt=dt, sum=sum, **params)
    # elif model=='2CF':
    #     return _conc_2cf(ca, *params, t=t, dt=dt, sum=sum)
    else:
        raise ValueError(
            'Kinetic model ' + kinetics + ' is not currently implemented.')



def _conc_u(ca, t=None, dt=1.0, Fb=None):
    if Fb is None:
        raise ValueError(
            'Fp is a required parameter for the tissue concentration '
            + 'in an uptake model. \nPlease provide a value.')
    C = pk.conc_trap(Fb*ca, t=t, dt=dt)
    return C
    

def _conc_fx(ca, t=None, dt=1.0, 
             H=None, ve=None, Fb=None):
    # Te = ve/Fp
    if Fb is None:
        raise ValueError(
            'Fb is a required parameter for the tissue concentration '
            + 'in a fast exchange model. \nPlease provide a value.')
    if Fb == 0:
        ce = ca*0
    else:
        if ve is None:
            raise ValueError(
                've is a required parameter for the tissue '
                + 'concentration in a fast exchange model. '
                + '\nPlease provide a value.')
        Fp = (1-H)*Fb
        ce = pk.flux_comp(ca/(1-H), ve/Fp, t=t, dt=dt)
    return ve*ce


def _conc_nx(ca, t=None, dt=1.0, 
             vb=None, Fb=None):
    if Fb is None:
        raise ValueError(
            'Fb is a required parameter for the tissue concentration '
            + 'in a no exchange model. \nPlease provide a value.')
    if Fb == 0:
        Cp = ca*0
    else:
        if vb is None:
            raise ValueError(
                'vb is a required parameter for the tissue ' 
                + 'concentration in a no exchange model. '
                + '\nPlease provide a value.')
        Cp = pk.conc_comp(Fb*ca, vb/Fb, t=t, dt=dt)
    return Cp 


def _conc_wv(ca, t=None, dt=1.0, 
             H=None, vi=None, Ktrans=None):
    if Ktrans == 0:
        ci = ca*0
    else:
        ci = pk.flux_comp(ca/(1-H), vi/Ktrans, t=t, dt=dt)
    return vi*ci


def _conc_hfu(ca, t=None, dt=1.0, sum=True, 
              H=None, vb=None, PS=None):
    vp = vb*(1-H)
    cp = ca/(1-H)
    Ci = pk.conc_trap(PS*cp, t=t, dt=dt)
    if sum:
        return vp*cp + Ci
    else:
        return np.stack((vp*cp, Ci)) 


def _conc_hf(ca, t=None, dt=1.0, sum=True,
             H=None, vi=None, vb=None, PS=None):
    vp = vb*(1-H)
    ca = ca/(1-H)
    Cp = vp*ca
    if PS == 0:
        Ci = 0*ca
    else:
        Ci = pk.conc_comp(PS*ca, vi/PS, t=t, dt=dt)
    if sum:
        return Cp+Ci
    else:
        return np.stack((Cp, Ci))


def _conc_2cu(ca, t=None, dt=1.0, sum=True,
              H=None, vb=None, Fb=None, PS=None):
    vp = (1-H)*vb
    Fp = (1-H)*Fb
    if np.isinf(Fp):
        return _conc_hfu(ca, t=t, dt=dt, sum=sum, vp=vp, PS=PS)
    ca = ca/(1-H)
    if Fp+PS == 0:
        return np.zeros((2, len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp == 0:
        Ktrans = PS*Fp/(PS+Fp)
        Ci = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ci = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    if sum:
        return Cp+Ci
    else:
        return np.stack((Cp, Ci))


def _conc_2cx(ca, t=None, dt=1.0, sum=True,
              H=None, vi=None, vb=None, Fb=None, PS=None):

    vp = (1-H)*vb
    Fp = (1-H)*Fb
    if np.isinf(Fp):
        return _conc_hf(ca, t=t, dt=dt, sum=sum, 
                        H=H, vi=vi, vb=vb, PS=PS)

    ca = ca/(1-H)
    J = Fp*ca

    if Fp+PS == 0:
        Cp = np.zeros(len(ca))
        Ce = np.zeros(len(ca))
        if sum:
            return Cp+Ce
        else:
            return np.stack((Cp, Ce))

    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)

    if PS == 0:
        Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
        Ci = np.zeros(len(ca))
        if sum:
            return Cp+Ci
        else:
            return np.stack((Cp, Ci))

    Ti = vi/PS

    C = pk.conc_2cxm(J, [Tp, Ti], E, t=t, dt=dt)
    if sum:
        return np.sum(C, axis=0)
    else:
        return C
    

def _conc_2cf(ca, t=None, dt=1.0, sum=True, 
              vp=None, Fp=None, PS=None, Te=None):
    if Fp+PS == 0:
        if sum:
            return np.zeros(len(ca))
        else:
            return np.zeros((2, len(ca)))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)
    J = Fp*ca
    T = [Tp, Te]
    # Solve the system explicitly
    t = utils.tarray(len(J), t=t, dt=dt)
    C0 = pk.conc_comp(J, T[0], t)
    if E == 0:
        C1 = np.zeros(len(t))
    elif T[0] == 0:
        J10 = E*J
        C1 = pk.conc_comp(J10, T[1], t)
    else:
        J10 = C0*E/T[0]
        C1 = pk.conc_comp(J10, T[1], t)
    if sum:
        return C0+C1
    else:
        return np.stack((C0, C1))
    

def _lconc_fx(ca, t=None, dt=1.0, Te=None):
    # Te = ve/Fp
    if Te is None:
        msg = ('Te is a required parameter for the concentration'
                + 'in a fast exchange model. \nPlease provide a value.')
        raise ValueError(msg)
    ce = pk.flux_comp(ca, Te, t=t, dt=dt)
    return ce


def _lconc_u(ca, t=None, dt=1.0, Tb=None):
    # Tb = vp/Fp
    if Tb is None:
        msg = ('Tb is a required parameter for the concentration'
                + 'in an uptake model. \nPlease provide a value.')
        raise ValueError(msg)
    if Tb==0:
        msg = ('An uptake tissue with Tb=0 is not well-defined. \n'
                + 'Consider constraining the parameters.')
        raise ValueError(msg)
    cp = pk.conc_trap(ca, t=t, dt=dt)/Tb
    return cp

    
def _lconc_nx(ca, t=None, dt=1.0, Tb=None):
    cp = pk.flux_comp(ca, Tb, t=t, dt=dt)
    return cp

    
def _lconc_wv(ca, t=None, dt=1.0, Ti=None):
    # Note cp is non-zero and equal to (1-E)*ca
    # But is not returned as it sits in a compartment without dimensions
    # Ti = vi/Ktrans
    ci = pk.flux_comp(ca, Ti, t=t, dt=dt)
    return ci

        
def _lconc_hfu(ca, t=None, dt=1.0, Ti=None):
    # Ti=vi/PS
    # up = vp / (vp + vi)
    cp = ca
    if Ti==0:
        msg = 'An uptake tissue with Ti=0 is not well-defined. \n'
        msg += 'Consider constraining the parameters.'
        raise ValueError(msg)
    ci = pk.conc_trap(cp, t=t, dt=dt)/Ti
    return np.stack((cp, ci))

        
def _lconc_hf(ca, t=None, dt=1.0, Ti=None):
    # Ti = vi/PS
    # up = vp/ve
    cp = ca
    ci = pk.flux_comp(cp, Ti, t=t, dt=dt)
    return np.stack((cp, ci))


def _lconc_2cu(ca, t=None, dt=1.0, 
               Tp=None, E=None, Ti=None):
    # Ti = vi/PS
    cp = (1-E)*pk.flux_comp(ca, Tp, t=t, dt=dt)
    ci = pk.conc_trap(cp, t=t, dt=dt)/Ti
    return np.stack((cp, ci))

        
def _lconc_2cx(ca, t=None, dt=1.0, Tp=None, Ti=None, E=None):
    # c = C/Fp
    # cp = C0/vp = c0*Fp/vp = c0 * (1-E)/Tp
    # ci = C1/vi = c1*Fp/vi = c1 * (1-E)/E/Ti
    c = pk.conc_2cxm(ca, [Tp, Ti], E, t=t, dt=dt)
    cp = c[0,:] * (1-E) / Tp
    ci = c[1,:] * (1-E) / E / Ti
    return np.stack((cp, ci))


def flux_tissue(ca: np.ndarray, t=None, dt=1.0, kinetics='2CX', **params) -> np.ndarray:
    """Indicator flux out of a 2-site exchange tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        t (array_like, optional): the time points of the input function *ca*. 
          If *t* is not provided, the time points are assumed to be uniformly 
          spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          provided. Defaults to 1.0.
        kinetics (str, optional): The kinetic model of the tissue (see below 
          for possible values). Defaults to '2CX'. 
        params (dict): free model parameters and their values (see below for 
          possible).

    Returns: 
        numpy.ndarray: outflux
          For a one-compartmental tissue, outflux out of the 
          compartment as a 1D array in units of mmol/sec/mL or M/sec. For a 
          multi=compartmental tissue, outflux out of each compartment, and at 
          each time point, as a 3D array with dimensions *(2,2,k)*, where *2* 
          is the number of compartments and *k* is the number of time points 
          in *J*. Encoding of the first two indices is the same as for *E*: 
          *J[j,i,:]* is the flux from compartment *i* to *j*, and *J[i,i,:]* 
          is the flux from *i* directly to the outside. The flux is returned in 
          units of mmol/sec/mL or M/sec.
    """
    if kinetics == 'U':
        return _flux_u(ca, **params)
    elif kinetics == 'NX':
        return _flux_nx(ca, t=t, dt=dt, **params)
    elif kinetics == 'FX':
        return _flux_fx(ca, t=t, dt=dt, **params)
    elif kinetics == 'WV':
        return _flux_wv(ca, t=t, dt=dt, **params)
    elif kinetics == 'HFU':
        return _flux_hfu(ca, **params)
    elif kinetics == 'HF':
        return _flux_hf(ca, t=t, dt=dt, **params)
    elif kinetics == '2CU':
        return _flux_2cu(ca, t=t, dt=dt, **params)
    elif kinetics == '2CX':
        return _flux_2cx(ca, t=t, dt=dt, **params)
    # elif model=='2CF':
    #     return _flux_2cf(ca, t=t, dt=dt, **params)
    else:
        raise ValueError('Kinetic model ' + kinetics +
                         ' is not currently implemented.')


def _flux_u(ca, Fb=None):
    return pk.flux(Fb*ca, model='trap')


def _flux_nx(ca, t=None, dt=1.0, vb=None, Fb=None):
    if Fb == 0:
        return np.zeros(len(ca))
    return pk.flux(Fb*ca, vb/Fb, t=t, dt=dt, model='comp')

def _flux_fx(ca, t=None, dt=1.0, H=None, ve=None, Fb=None):
    if Fb == 0:
        return np.zeros(len(ca))
    Fp = Fb*(1-H)
    return pk.flux(Fb*ca, ve/Fp, t=t, dt=dt, model='comp')

def _flux_wv(ca, t=None, dt=1.0, H=None, vi=None, Ktrans=None):
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = Ktrans*ca
    if Ktrans != 0:
        J[0, 1, :] = pk.flux(Ktrans*ca, vi/Ktrans, t=t, dt=dt, model='comp')
    return J

def _flux_hfu(ca, H=None, PS=None):
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = PS*ca/(1-H)
    return J


def _flux_hf(ca, t=None, dt=1.0, H=None, vi=None, PS=None):
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.inf
    J[1, 0, :] = PS*ca
    if PS == 0:
        J[0, 1, :] = 0*ca
    else:
        J[0, 1, :] = pk.flux(PS*ca, vi/PS, t=t, dt=dt, model='comp')
    return J


def _flux_2cu(ca, t=None, dt=1.0, H=None, vb=None, Fb=None, PS=None):
    C = _conc_2cu(ca, t=t, dt=dt, sum=False, H=H, vb=vb, Fb=Fb, PS=PS)
    ca = ca/(1-H)
    Fp = Fb*(1-H)
    J = np.zeros(((2, 2, len(ca))))
    if vb == 0:
        if Fp+PS != 0:
            Ktrans = Fp*PS/(Fp+PS)
            J[0, 0, :] = Fp*ca
            J[1, 0, :] = Ktrans*ca
    else:
        J[0, 0, :] = Fp*C[0, :]/vb
        J[1, 0, :] = PS*C[0, :]/vb
    return J


def _flux_2cx(ca, t=None, dt=1.0, H=None, vb=None, vi=None, Fb=None, PS=None):

    if np.isinf(Fb):
        return _flux_hf(ca, t=t, dt=dt, H=H, vi=vi, PS=PS)

    if Fb == 0:
        return np.zeros((2, 2, len(ca)))

    Fp = Fb*(1-H)
    if Fp+PS == 0:
        return np.zeros((2, 2, len(ca)))
    if PS == 0:
        Jp = _flux_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
        J = np.zeros((2, 2, len(ca)))
        J[0, 0, :] = Jp
        return J
    C = _conc_2cx(ca, t=t, dt=dt, sum=False, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    # Derive standard parameters
    vp = vb*(1-H)
    Tp = vp/(Fp+PS)
    Te = vi/PS
    E = PS/(Fp+PS)
    # Build the system matrix K
    T = [Tp, Te]
    E = [
        [1-E, 1],
        [E,   0],
    ]
    return pk._J_ncomp(C, T, E)


def _flux_2cf(ca, t=None, dt=1.0, vp=None, Fp=None, PS=None, Te=None):
    if Fp+PS == 0:
        return np.zeros((2, 2, len(ca)))
    # Derive standard parameters
    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)
    J = Fp*ca
    T = [Tp, Te]
    # Solve the system explicitly
    t = utils.tarray(len(J), t=t, dt=dt)
    Jo = np.zeros((2, 2, len(t)))
    J0 = pk.flux(J, T[0], t=t, model='comp')
    J10 = E*J0
    Jo[1, 0, :] = J10
    Jo[1, 1, :] = pk.flux(J10, T[1], t=t, model='comp')
    Jo[0, 0, :] = (1-E)*J0
    return Jo
