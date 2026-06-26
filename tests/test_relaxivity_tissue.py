import itertools
import numpy as np

import dcmri as dc

def test_coverage():

    nt = 10

    DEFAULTS = dc.QVALUES | {'c': np.ones(nt), 'C': np.ones(nt)}
    DEFAULTS2 = dc.QVALUES | {'c': np.ones((5, nt)), 'C': np.ones((5, nt))}

    # Run for coverage
    for func in [dc.R1, dc.R2, dc.R2s, dc.Relax]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            print(cnfgs)
            func(*cnfgs, defaults=DEFAULTS)()
            func(*cnfgs, defaults=DEFAULTS2)()

    for func in [dc.R1]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            func(*cnfgs, defaults=DEFAULTS2)(R1b=np.ones(5))
            func(*cnfgs, defaults=DEFAULTS2)(r1=np.ones(5))

    for func in [dc.R2]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            func(*cnfgs, defaults=DEFAULTS2)(R2b=np.ones(5))
            func(*cnfgs, defaults=DEFAULTS2)(r2=np.ones(5))



if __name__ == "__main__":
    test_coverage()
    
    print('All relaxivity models tests passing!')