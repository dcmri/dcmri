import itertools
import numpy as np

from dcmri import R1, R2, R2s, Relax



def test_coverage():

    nt = 10
    c = np.ones(nt)
    c2 = np.ones((5, nt))

    # Run for coverage
    for func in [R1, R2, R2s, Relax]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            func(*cnfgs)(c)
            func(*cnfgs)(c2)

    for func in [R1]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            func(*cnfgs)(c2, R10=np.ones(c2.shape[0]))
            func(*cnfgs)(c2, r1=np.ones(c2.shape[0]))

    for func in [R2]:
        values = func.configs.values()
        for cnfgs in itertools.product(*values):
            func(*cnfgs)(c2, R20=np.ones(c2.shape[0]))
            func(*cnfgs)(c2, r2=np.ones(c2.shape[0]))



if __name__ == "__main__":
    test_coverage()
    
    print('All relaxivity models tests passing!')