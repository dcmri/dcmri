import numpy as np
import dcmri

def test_inj_step():
    dcmri.inj_step(np.arange(5),1,1,1,1,1,1,1)
    dcmri.inj_step(np.arange(5),1,1,1,1,1)
    assert True


if __name__ == "__main__":

    test_inj_step()

    print('All inj tests passed!!')