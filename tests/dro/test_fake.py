
import numpy as np
from scipy.integrate import trapezoid

import dcmri as dc


def test_fake_aif():
    time, aif, _ = dc.fake.aif()
    assert 1800 < trapezoid(aif, time) < 1900
    time, aif, _ = dc.fake.aif(model='3D-SPGR-SS')
    assert 1800 < trapezoid(aif, time) < 2000

def test_fake_brain():
    time, signal, aif, gt = dc.fake.brain(n=64)
    assert 1800 < trapezoid(aif, time) < 1900
    assert 445 < trapezoid(signal[32,32,:], time) < 460
    time, signal, aif, gt = dc.fake.brain(n=8, model='3D-SR-SPGR-SS', verbose=1)
    assert 1600 < trapezoid(aif, time) < 1700
    assert 170 < trapezoid(signal[3,3,:], time) < 180

def test_fake_tissue():
    time, aif, roi, gt = dc.fake.tissue()
    assert 1230 < trapezoid(aif, time) < 1240
    assert 1800 < trapezoid(roi, time) < 1900
    time, aif, roi, gt = dc.fake.tissue(model='2D-SR-SPGR-SS')
    assert 1000 < trapezoid(aif, time) < 1100
    assert 1000 < trapezoid(roi, time) < 1100

def test_fake_liver():
    time, aif, vif, roi, gt = dc.fake.liver()
    assert 1500 < trapezoid(aif, time) < 1600
    assert 1500 < trapezoid(vif, time) < 1600
    assert 2000 < trapezoid(roi, time) < 2020
    time, aif, vif, roi, gt = dc.fake.liver(sequence='3D-SPGR-SSI')
    assert 1500 < trapezoid(aif, time) < 1600
    assert 1500 < trapezoid(vif, time) < 1600
    assert 2000 < trapezoid(roi, time) < 2020

def test_fake_tissue2scan():
    time, aif, roi, gt = dc.fake.tissue2scan()
    assert 1230 < trapezoid(aif[0], time[0]) < 1240
    assert 3120 < trapezoid(aif[1], time[1]) < 3145
    assert 1750 < trapezoid(roi[0], time[0]) < 1850
    assert 4250 < trapezoid(roi[1], time[1]) < 4350
    time, aif, roi, gt = dc.fake.tissue2scan(model='2D-SR-SPGR-SS')
    assert 1000 < trapezoid(aif[0], time[0]) < 1100
    assert 3500 < trapezoid(roi[1], time[1]) < 3600

def test_fake_kidney():
    time, aif, roi, gt = dc.fake.kidney()
    assert 950 < trapezoid(aif, time) < 16050
    assert 3100 < trapezoid(roi[0], time) < 3200
    assert 1850 < trapezoid(roi[1], time) < 1950
    time, aif, roi, gt = dc.fake.kidney(model='3D-SPGR-SS')
    assert 1145 < trapezoid(aif, time) < 1155
    assert 2300 < trapezoid(roi[1], time) < 2400



if __name__ == "__main__":
    test_fake_aif()
    test_fake_tissue()
    test_fake_brain()
    test_fake_tissue2scan()
    test_fake_liver()
    test_fake_kidney()


    print('All fake tests passed!!')