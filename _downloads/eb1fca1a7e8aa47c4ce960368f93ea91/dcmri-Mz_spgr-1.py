# Import packages:
#
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc
#
# Define constants:
#
FA, TR, TP = 12, 0.005, 0
TI = np.linspace(0,3,100)
R1 = [1, 0.5]
v = [0.3, 0.7]
f, PS = 0.5, 0.1
Fw = [[f, PS], [PS, 0]]
#
# Compute magnetization:
#
Mspgr = dc.Mz_spgr(R1, TI, TR, FA, TP, v, Fw, j=[f, 0], n0=-1)
Mfree = dc.Mz_free(R1, TI, v, Fw, j=[f, 0], n0=-1)
Mss = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f, 0])
#
# Plot the results for the peripheral compartment:
#
c = 1
plt.title('Peripheral compartment')
plt.plot(TI, Mfree[c,:], label='Free', linewidth=3)
plt.plot(TI, v[c]+0*TI, label='Equilibrium', linewidth=3)
plt.plot(TI, Mspgr[c,:], label='SPGR', linewidth=3)
plt.plot(TI, Mss[c]+0*TI, label='Steady-state', linewidth=3)
plt.xlabel('Time since start of pulse sequence (sec)')
plt.ylabel('Tissue magnetization (A/cm/cm3)')
plt.legend()
plt.show()
#
# This verifies that the free recovery magnetization goes to equilibrium,
# and that the SPGR magnetization relaxes to the steady state at a
