
import numpy as np
import dcmri as dc


def test_fake_aif():
    time, aif, _ = dc.fake_aif()
    assert 1800 < np.trapezoid(aif, time) < 1900
    time, aif, _ = dc.fake_aif(model='SR')
    assert 2680 < np.trapezoid(aif, time) < 2690

def test_fake_brain():
    time, signal, aif, gt = dc.fake_brain(n=64)
    assert 1800 < np.trapezoid(aif, time) < 1900
    assert 445 < np.trapezoid(signal[32,32,:], time) < 460
    time, signal, aif, gt = dc.fake_brain(n=8, model='SR', verbose=1)
    assert 2680 < np.trapezoid(aif, time) < 2690
    assert 170 < np.trapezoid(signal[3,3,:], time) < 180

def test_fake_tissue():
    time, aif, roi, gt = dc.fake_tissue()
    assert 1230 < np.trapezoid(aif, time) < 1240
    assert 1150 < np.trapezoid(roi, time) < 1170
    time, aif, roi, gt = dc.fake_tissue(model='SR')
    assert 1785 < np.trapezoid(aif, time) < 1795
    assert 935 < np.trapezoid(roi, time) < 945

def test_fake_liver():
    time, aif, vif, roi, gt = dc.fake_liver()
    assert 1910 < np.trapezoid(aif, time) < 1920
    assert 1925 < np.trapezoid(vif, time) < 1935
    assert 2740 < np.trapezoid(roi, time) < 2750
    time, aif, vif, roi, gt = dc.fake_liver(sequence='SSI')
    assert 1925 < np.trapezoid(aif, time) < 1935
    assert 1925 < np.trapezoid(vif, time) < 1935
    assert 2740 < np.trapezoid(roi, time) < 2750

def test_fake_tissue2scan():
    time, aif, roi, gt = dc.fake_tissue2scan()
    assert 1230 < np.trapezoid(aif[0], time[0]) < 1240
    assert 3120 < np.trapezoid(aif[1], time[1]) < 3145
    assert 1130 < np.trapezoid(roi[0], time[0]) < 1145
    assert 2930 < np.trapezoid(roi[1], time[1]) < 2945
    time, aif, roi, gt = dc.fake_tissue2scan(model='SR')
    assert 1785 < np.trapezoid(aif[0], time[0]) < 1795
    assert 2425 < np.trapezoid(roi[1], time[1]) < 2435

def test_fake_kidney():
    time, aif, roi, gt = dc.fake_kidney()
    assert 1660 < np.trapezoid(aif, time) < 1680
    assert 1555 < np.trapezoid(roi[0], time) < 1565
    assert 900 < np.trapezoid(roi[1], time) < 915
    time, aif, roi, gt = dc.fake_kidney(model='SS')
    assert 1145 < np.trapezoid(aif, time) < 1155
    assert 1110 < np.trapezoid(roi[1], time) < 1125



if __name__ == "__main__":

    test_fake_aif()
    test_fake_tissue()
    test_fake_brain()
    test_fake_tissue2scan()
    test_fake_liver()
    test_fake_kidney()


    print('All fake tests passed!!')