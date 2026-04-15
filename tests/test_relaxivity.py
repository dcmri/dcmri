import numpy as np
import dcmri as dc

from dcmri import relaxivity

def test_conc_t1():
    R1 = np.arange(10) + 5
    c = relaxivity.conc_t1(R1, 0.1)
    assert np.sum(c) == 450.0
    R1 = np.stack((R1, 2*R1))
    c = relaxivity.conc_t1(R1, [5,4])
    assert np.sum(c) == 31.5
    c = relaxivity.conc_t1(R1, 5)
    assert np.sum(c) == 27


def test_relax_t1():

    # One compartment
    #################
    r1 = 1

    # One time point - ROI
    R10 = 1
    c = 1
    assert relaxivity.relax_t1(c, R10, r1) == 2

    # One time point - image
    shape = (3,3)
    R10 = np.ones(shape)
    c = np.ones(shape)
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full(shape,2))

    # 4 time points - ROI
    R10 = 1
    c = np.ones(4)
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full((4,),2))

    # 4 time points - image
    shape = (3,3,4)
    R10 = np.ones(shape[:2])
    c = np.ones(shape)
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full(shape,2))

    # Two compartments
    ##################
    r1 = [1,1]

    # One time point - ROI
    R10 = [1,1]
    c = [1,1]
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full((2,),2))

    # One time point - image
    shape = (2,3,3)
    R10 = np.ones(shape)
    c = np.ones(shape)
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full(shape,2))

    # 4 time points - ROI
    R10 = [1,1]
    c = np.ones((2,4))
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full((2,4),2))

    # 4 time points - image
    shape = (2,3,3,4)
    R10 = np.ones(shape[:3])
    c = np.ones(shape)
    assert np.array_equal(relaxivity.relax_t1(c, R10, r1), np.full(shape,2))


if __name__ == "__main__":

    test_conc_t1()
    test_relax_t1()

    print('All relaxivity tests passing!')