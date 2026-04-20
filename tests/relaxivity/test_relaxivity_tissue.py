import itertools
import numpy as np

from dcmri.relaxivity import R1, R2, R2s, Relax



def test_coverage():

    nt = 10
    c = np.ones(nt)

    # Run for coverage
    for func in [R1, R2, R2s, Relax]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            print(func, cnfgs)
            func(*cnfgs)(c)



if __name__ == "__main__":
    test_coverage()
    
    print('All relaxivity models tests passing!')