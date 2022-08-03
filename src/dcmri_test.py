import numpy as np
import matplotlib.pyplot as plt
import dcmri

tmax = 120 # sec
dt = 0.1 # sec
MTT = 20 # sec

weight = 70.0           # Patient weight in kg
conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
dose = 0.025            # mL per kg bodyweight (quarter dose)
rate = 1                # Injection rate (mL/sec)
start = 20.0        # sec
dispersion = 90    # %

def test_convolve():

    t = np.arange(0, tmax, dt)
    P = dcmri.compartment_propagator(t, MTT)
    J = dcmri.injection_gv(t, weight, conc, dose, rate, start, dispersion=dispersion)

    ref = dcmri.propagate_compartment(t, J, MTT)
    new = dcmri.convolve(t, t, J, t, P)

    plt.plot(t, ref, 'b-')
    plt.plot(t, new, 'rx')
    plt.show()

    dif = ref-new
    error = 100*np.sqrt(np.mean(dif**2))/ np.sqrt(np.mean(ref**2))
    print('Error (%)', error)

if __name__ == "__main__":
    test_convolve()

