import itertools
import numpy as np

import dcmri as dc

def test_coverage():

    nc, nt = 5, 10

    DEFAULTS = dc.QVALUES | {'v': 1, 'c': np.ones(nt), 'R1b': 1, 'R2b': 1, 'R2sb': 1}
    DEFAULTS2 = dc.QVALUES | {'v': np.ones(nc) / nc, 'c': np.ones((nc, nt)), 'R1b': np.ones(nc), 'R2b': np.ones(nc), 'R2sb': 1, 'r1': 1e3 * np.ones(nc), 'r2': 1e3 * np.ones(nc)}

    # Run for coverage
    for module in [dc.R1, dc.R2, dc.R2s, dc.Relax]:
        for cnfg in module.configurations():
            print(cnfg)
            module(**cnfg)(DEFAULTS)
            module(**cnfg)(DEFAULTS2)


if __name__ == "__main__":
    test_coverage()
    
    
    print('All relaxivity models tests passing!')