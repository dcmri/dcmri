import numpy as np
import dcmri as dc



def test_conc_t2w():
    S = np.ones(10)
    TE = np.inf
    C = dc.conc('T2w', S, TE=TE, r2=1, n0=1)
    assert 0 == np.linalg.norm(C)
    
def test_conc_ss():
    S = np.ones(10)
    C = dc.conc('SS', S, R10=0, TR=1, FA=45, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

    # Data without solution
    S = [1,2,0]
    C = dc.conc('SS', S, TR=1, FA=45, R10=1, r1=1, n0=1)
    assert C[-1] == -1

def test_conc_src():
    S = np.ones(10)
    C = dc.conc('SRC', S, TC=1, R10=0, r1=1, n0=1)
    assert 0 == np.linalg.norm(C)

def test_conc_lin():
    S = np.ones(10)
    C = dc.conc('lin', S, R10=1, r1=1, n0=1)
    assert np.linalg.norm(C)==0



if __name__ == "__main__":

    test_conc_t2w()
    test_conc_ss()
    test_conc_src()
    test_conc_lin()

    print('All sig tests passing!')