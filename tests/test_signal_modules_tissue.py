import numpy as np
from tqdm import tqdm

from dcmri.core.exceptions import InvalidConfiguration
from dcmri import Signal, CalibrateSignal, QVALUES, Magnetization
from dcmri.bloch.functions_sequences import repetition_time
from dcmri.signal.modules_tissue import signal_rice, RelaxToSignal, ConcToSignal


def test_signal_rice():
    """Test signal_rice for clean math path and its fallback logic."""
    # Case 1: Zero noise standard deviation returns input directly
    res_zero_sigma = signal_rice(np.array([10.0]), sigma=0.0)
    np.testing.assert_array_almost_equal(res_zero_sigma, np.array([10.0]))
    
    # Case 2: Standard math path
    res_standard = signal_rice(np.array([2.0, 5.0]), sigma=1.0)
    assert res_standard.shape == (2,)
   
    # Case 3: Extreme inputs trigger the fallback logic cleanly
    res_fallback = signal_rice(np.array([1e10]), sigma=1e-10)
    np.testing.assert_array_almost_equal(res_fallback, np.array([1e10]))

def test_coverage_signal():
    for config in Signal.configurations():
        print(config)
        sig = Signal(**config)
        sig.inputs()
        sig.outputs()

        p = QVALUES | sig.lexicon_data(QVALUES)
        S = sig(p)['S']


def test_coverage_calibrate_signal():
    # # Inputs not in Lexicon
    # print(CalibrateSignal.all_inputs() - set(QVALUES.keys()))
    # # Inputs in Lexicon
    # print(CalibrateSignal.all_inputs() & set(QVALUES.keys()))
    # return

    for config in CalibrateSignal.configurations():
        print(config)
        cal = CalibrateSignal(**config)
        cal.inputs()
        cal.outputs()

        p = QVALUES | cal.lexicon_data(QVALUES)
        S0 = cal(p)['S0']


def test_coverage_relax_to_signal():
    for config in RelaxToSignal.configurations():
        print('relax_to_signal', config)
        sig = RelaxToSignal(**config)
        sig.inputs()
        sig.outputs()

        p = QVALUES | sig.lexicon_data(QVALUES)
        S = sig(p)['S']


def test_coverage_conc_to_signal():
    # # Inputs not in Lexicon
    # print(ConcToSignal.all_inputs() - set(QVALUES.keys()))
    # # Inputs in Lexicon
    # print(ConcToSignal.all_inputs() & set(QVALUES.keys()))
    # return

    configs = ConcToSignal.configurations()
    
    for config in tqdm(list(configs)):
        # For debugging
        # cnfg = {'t1_relaxation': 'lin', 't2_relaxation': None, 't2s_relaxation': 'quad', 'sequence': '3D-SR-SS', 'inflow': False, 'magnitude': False, 'calibrate': True}
        # if config != cnfg:
        #     continue

        try:
            sig = ConcToSignal(**config)
        except InvalidConfiguration:
            continue
        print('conc_to_signal', config)
        i = sig.inputs()
        o = sig.outputs()
        p = sig.lexicon_data(QVALUES)
        S = sig(p)['S']


def test_function_calibrate_signal():
    config = {'sequence': '3D-SPGR', 'magnitude': False, 'inflow': True, 'trigger': False}
    imap = {'Sb':'Sb_a', 'R1b':'R1b_a', 'R2b':'R2b_a', 'R2sb':'R2sb_a', 'R1ib':'R1b_a'}

    # Compute S0 with default inputs
    scal = CalibrateSignal(imap=imap, **config)
    pars = scal.lexicon_data(QVALUES) 
    p = scal(pars) 
    S0_1 = p['S0']

    # Generate signal with the derived S0
    magn = Magnetization(**config)
    signal = Signal(**config)

    nc, nt = 2, 10 
    TR = repetition_time(config['sequence'], pars)
    pars |= {
        'tR': TR * np.arange(nt),
        'R2s': QVALUES['R2sb'] * np.ones(nt),
        'R1': QVALUES['R1b'] * np.ones((nc, nt)),
        'R2': QVALUES['R2b'] * np.ones((nc, nt)),
        'R1i': QVALUES['R1ib'] * np.ones((nc, nt)),
        'S0': p['S0'],
    }
    p |= magn(pars) 
    s = signal(p, noise_sdev=0) 

    # Set signal as baseline and compute S0 again
    n0 = 5
    scal = CalibrateSignal(imap=imap, **config)
    pars = scal.lexicon_data(QVALUES)
    pars['Sb'] = s['S'][..., :n0]
    p = scal(pars)  
    S0_2 = p['S0']

    # Check that the two S0 values are the same
    print(S0_1, S0_2) 
    assert np.abs(S0_1 - S0_2) / np.mean([S0_1, S0_2]) < 1e-9


if __name__ == "__main__":
    test_signal_rice()
    test_coverage_signal()
    test_coverage_calibrate_signal()
    test_function_calibrate_signal()
    test_coverage_relax_to_signal()
    test_coverage_conc_to_signal()
    
    print('All signal tests passing!')