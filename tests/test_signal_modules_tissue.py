import numpy as np

from dcmri import Signal, CalibrateSignal, QVALUES



def test_coverage_signal():
    for config in Signal.configurations():
        print(config)
        sig = Signal(**config)
        sig.inputs()
        sig.outputs()

        M = np.ones((3, 2, 10))
        time = 2 * np.ones(3)
        S = sig(QVALUES, M=M, time=time)['S']


def test_coverage_calibrate_signal():
    for config in CalibrateSignal.configurations():
        print(config)
        cal = CalibrateSignal(**config)
        cal.inputs()
        cal.outputs()

        p = cal.map_lexicon(QVALUES)
        S0 = cal(p)['S0']
        print(S0)


if __name__ == "__main__":
    test_coverage_signal()
    test_coverage_calibrate_signal()
    
    print('All signal tests passing!')