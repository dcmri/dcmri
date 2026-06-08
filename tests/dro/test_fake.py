
import numpy as np
from scipy.integrate import trapezoid

import dcmri as dc


def test_fake_aif():
    time, aif, _ = dc.aif()
    assert 1800 < trapezoid(aif, time) < 1900
    time, aif, _ = dc.aif(model='3D-SPGR-SS')
    assert 1800 < trapezoid(aif, time) < 2000
    time, aif, _ = dc.aif(model='3D-SR-SPGR-SS')
    assert 1550 < trapezoid(aif, time) < 1650

def test_fake_brain():
    time, signal, aif, gt = dc.brain(n=64)
    assert 1700 < trapezoid(aif, time) < 1800
    assert 430 < trapezoid(signal[32,32,:], time) < 440
    time, signal, aif, gt = dc.brain(n=8, model='3D-SR-SPGR-SS', verbose=1)
    assert 1500 < trapezoid(aif, time) < 1600
    assert 160 < trapezoid(signal[3,3,:], time) < 180

def test_fake_tissue():
    time, aif, roi, gt = dc.tissue()
    assert 1230 < trapezoid(aif, time) < 1240
    assert 1100 < trapezoid(roi, time) < 1200
    time, aif, roi, gt = dc.tissue(model='2D-SR-SPGR-SS')
    assert 1000 < trapezoid(aif, time) < 1100
    assert 1000 < trapezoid(roi, time) < 1100

def test_fake_liver():
    time, aif, vif, roi, gt = dc.liver()
    assert 1500 < trapezoid(aif, time) < 1600
    assert 1500 < trapezoid(vif, time) < 1600
    assert 2000 < trapezoid(roi, time) < 2020
    time, aif, vif, roi, gt = dc.liver(sequence='3D-SPGR-SSI')
    assert 1500 < trapezoid(aif, time) < 1600
    assert 1500 < trapezoid(vif, time) < 1600
    assert 2000 < trapezoid(roi, time) < 2020

def test_fake_tissue2scan():
    time, aif, roi, gt = dc.tissue2scan()
    assert 1230 < trapezoid(aif[0], time[0]) < 1240
    assert 3120 < trapezoid(aif[1], time[1]) < 3145
    assert 1100 < trapezoid(roi[0], time[0]) < 1200
    assert 2900 < trapezoid(roi[1], time[1]) < 3000
    time, aif, roi, gt = dc.tissue2scan(model='2D-SR-SPGR-SS')
    assert 1000 < trapezoid(aif[0], time[0]) < 1100
    assert 2400 < trapezoid(roi[1], time[1]) < 2500

def test_fake_kidney():
    time, aif, roi, gt = dc.kidney()
    assert 950 < trapezoid(aif, time) < 16050
    assert 1500 < trapezoid(roi[0], time) < 1600
    assert 900 < trapezoid(roi[1], time) < 1000
    time, aif, roi, gt = dc.kidney(model='3D-SPGR-SS')
    assert 1100 < trapezoid(aif, time) < 1200
    assert 1100 < trapezoid(roi[1], time) < 1200



if __name__ == "__main__":
    test_fake_aif()
    test_fake_tissue()
    test_fake_brain()
    test_fake_tissue2scan()
    test_fake_liver()
    test_fake_kidney()


    print('All fake tests passed!!')