import numpy as np

from dcmri import Signal, CalibrateSignal, QVALUES, Magnetization
from dcmri.signal.modules_tissue import signal_rice, RelaxToSignal


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

        p = sig.map_lexicon(QVALUES)
        S = sig(p)['S']


def test_coverage_calibrate_signal():
    for config in CalibrateSignal.configurations():
        print(config)
        cal = CalibrateSignal(**config)
        cal.inputs()
        cal.outputs()

        p = cal.map_lexicon(QVALUES)
        S0 = cal(p)['S0']
        print(S0)


def test_coverage_relax_to_signal():
    for config in RelaxToSignal.configurations():
        print('relax_to_signal', config)
        sig = RelaxToSignal(**config)
        sig.inputs()
        sig.outputs()

        p = sig.map_lexicon(QVALUES)
        S = sig(p)['S']


def test_function_calibrate_signal():
    # Replicates a bug from AortaModel
    config = {'sequence': '3D-SPGR', 'magnitude': False, 'inflow': True, 'trigger': False}
    imap = {'tSb': 'tSb_a', 'Sb':'Sb_a', 'R1b':'R1b_a', 'R2b':'R2b_a', 'R2sb':'R2sb_a', 'R1ib':'R1b_a'}

    # Compute S0
    scal = CalibrateSignal(imap=imap, **config)
    pars = scal.map_lexicon(QVALUES) | {'v': 1, 'Fw': 10, 'me': 1, 'Fi': 10} 
    p = scal(pars) 
    S0_1 = p['S0']

    # Generate signal with constant baseline
    magn = Magnetization(**config)
    signal = Signal(**config)

    n0 = 10 
    # Clue: need to simulate longer then truncate to 5 to get the exact result. 
    # Simulating n0=5 gives a slightly different result in the final value
    # This would be VERY relevant for n0=1 simulation
    pars |= p | {
        'tR': 0.5 * np.arange(n0),
        'R2s': 20 * np.ones(n0),
        'R1': 0.65 * np.ones(n0),
        'R1i': 0.65 * np.ones(n0),
    }
    p |= magn(pars) 
    s = signal(p, noise_sdev=0) 

    # Set signals and compute S0 again
    n0=5
    scal = CalibrateSignal(imap=imap, **config)
    pars = scal.map_lexicon(QVALUES) | {'v': 1, 'Fw': 10, 'me': 1, 'Fi': 10} 
    pars['Sb_a'] = s['S'][..., :n0]
    pars['tSb_a'] = s['tS'][:n0]
    p = scal(pars)  

    S0_2 = p['S0']

    print(S0_1, S0_2) # These should be the same exactly
    assert np.abs(S0_1 - S0_2) / np.mean([S0_1, S0_2]) < 1e-9


if __name__ == "__main__":
    test_signal_rice()
    test_coverage_signal()
    test_coverage_calibrate_signal()
    test_coverage_relax_to_signal()
    test_function_calibrate_signal()
    
    print('All signal tests passing!')