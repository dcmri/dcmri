import numpy as np

from dcmri import Signal, CalibrateSignal, QVALUES
from dcmri.signal.modules_tissue import signal_rice, RelaxToSignal
from dcmri.bloch.functions_sequences import channels

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

        M = np.ones((2, 3, 2, 10))
        S = sig(QVALUES, M=M)['S']


def test_coverage_calibrate_signal():

    def map_lexicon(cal, p):
        # Set baseline if not provided
        n_channels = channels(cal.config['sequence'])
        components = 1 if cal.config['magnitude'] else 2
        nt = 1
        return p | {
            'tSb': np.arange(nt),
            'Sb': np.full((n_channels, components, nt), p['Sb'])
        }
    
    for config in CalibrateSignal.configurations():
        print(config)
        cal = CalibrateSignal(**config)
        cal.inputs()
        cal.outputs()

        p = map_lexicon(cal, QVALUES)
        S0 = cal(p)['S0']
        print(S0)


def test_coverage_relax_to_signal():
    def map_lexicon(sig, p):
        nc, nt = 3, 5
        return p | {
            'v': np.ones(nc) / nc,
            'Fw': np.eye(nc),
            'tR': np.arange(nt),
            'R1': np.full((nc, nt), p['R1']),
            'R2': np.full((nc, nt), p['R2']),
            'R2s': np.full(nt, p['R2s']),
            'Fi': np.ones(nc),
            'R1i': np.full((nc, nt), p['R1i']),
        }
    
    for config in RelaxToSignal.configurations():
        print('relax_to_signal', config)
        sig = RelaxToSignal(**config)
        sig.inputs()
        sig.outputs()

        p = map_lexicon(sig, QVALUES)
        S = sig(p)['S']


if __name__ == "__main__":
    test_signal_rice()
    test_coverage_signal()
    test_coverage_calibrate_signal()
    test_coverage_relax_to_signal()
    
    print('All signal tests passing!')