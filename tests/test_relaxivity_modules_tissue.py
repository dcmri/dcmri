import itertools
import numpy as np

import dcmri as dc

def test_coverage():

    nt = 10

    DEFAULTS = dc.QVALUES | {'c': np.ones(nt), 'C': np.ones(nt), 'R1b': 1, 'R2b': 1, 'R2sb': 1}
    DEFAULTS2 = dc.QVALUES | {'c': np.ones((5, nt)), 'C': np.ones((5, nt)), 'R1b': np.ones(5), 'R2b': np.ones(5), 'R2sb': 1, 'r1': 1e3 * np.ones(5), 'r2': 1e3 * np.ones(5)}

    # Run for coverage
    for module in [dc.R1, dc.R2, dc.R2s, dc.Relax]:
        values = module.configs.values()
        for cnfgs in itertools.product(*values):
            print(cnfgs)
            config = {k: cnfgs[i] for i, k in enumerate(module.configs)}
            module(**config)(DEFAULTS)
            module(**config)(DEFAULTS2)



if __name__ == "__main__":
    test_coverage()
    
    print('All relaxivity models tests passing!')