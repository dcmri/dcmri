import numpy as np
import dcmri

def test_conc_spgress():
    dcmri.conc_spgress(1,1,1,1,1,1)

def test_conc_lin():
    dcmri.conc_lin(1,1,1,1)

if __name__ == "__main__":

    test_conc_spgress()
    test_conc_lin()

    print('All inv tests passed!!')