import itertools
import numpy as np

from dcmri import Signal, QVALUES


def test_coverage_signal():
    values = Signal.configs.values()
    for cnfgs in itertools.product(*values):
        print(cnfgs)
        config = {k: cnfgs[i] for i, k in enumerate(Signal.configs)}
        sig = Signal(**config)
        sig.inputs()
        sig.outputs()

        M = np.ones((3, 2, 10))
        time = 2 * np.ones(3)
        S = sig(QVALUES, M=M, time=time)['S']


if __name__ == "__main__":
    test_coverage_signal()
    
    print('All magnetization tests passing!')