import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc
#
# Plot magnetization recovery for the first 10 seconds after an
# inversion pulse, for a closed tissue with R1 = 1 sec, and for an open
# tissue with equilibrium inflow and inverted inflow:
#
TI = 0.1*np.arange(100)
R1 = 1
f = 0.5
#
Mz = dc.Mz_free(R1, TI, n0=-1)
Mz_e = dc.Mz_free(R1, TI, n0=-1, Fw=f, j=f)
Mz_i = dc.Mz_free(R1, TI, n0=-1, Fw=f, j=-f)
#
plt.plot(TI, Mz, label='No flow', linewidth=3)
plt.plot(TI, Mz_e, label='Equilibrium inflow', linewidth=3)
plt.plot(TI, Mz_i, label='Inverted inflow', linewidth=3)
plt.xlabel('Inversion time (sec)')
plt.ylabel('Magnetization (A/cm)')
plt.legend()
plt.show()
#
# Now consider a two-compartment model, with a central compartment
# that has in- and outflow, and a peripheral compartment that only
# exchanges with the central compartment:
#
R1 = [1,2]
v = [0.3, 0.7]
PS = 0.1
Fw = [[f, PS], [PS, 0]]
Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=[f, 0])
#
plt.plot(TI, Mz[0,:], label='Central compartment', linewidth=3)
plt.plot(TI, Mz[1,:], label='Peripheral compartment', linewidth=3)
plt.xlabel('Inversion time (sec)')
plt.ylabel('Magnetization (A/cm)')
plt.legend()
plt.show()
#
# In DC-MRI the more usual situation is one where TI is fixed and the
# relaxation rates are variable due to the effect of a contrast agent.
# As an illustration, consider the previous result again at TI=500 msec
# and an R1 that is linearly declining in the central compartment and
# constant in the peripheral compartment:
#
TI = 0.5
nt = 1000
t = 0.1*np.arange(nt)
R1 = np.stack((1-t/np.amax(t), np.ones(nt)))
j = np.stack((f*np.ones(nt), np.zeros(nt)))
Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=j)
#
plt.plot(t, Mz[0,:], label='Central compartment', linewidth=3)
plt.plot(t, Mz[1,:], label='Peripheral compartment', linewidth=3)
plt.xlabel('Time (sec)')
plt.ylabel('Magnetization (A/cm)')
plt.legend()
plt.show()
#
# The function allows for R1 and TI to be both variable. Computing the
# result for 10 different TI values and extracting the result
# corresponding to TI=0.5 gives again the same result:
#
TI = 0.1*np.arange(10)
Mz = dc.Mz_free(R1, TI, v, Fw, n0=-1, j=j)
#
plt.plot(t, Mz[0,:,5], label='Central compartment', linewidth=3)
plt.plot(t, Mz[1,:,5], label='Peripheral compartment', linewidth=3)
plt.xlabel('Time (sec)')
plt.ylabel('Magnetization (A/cm)')
plt.legend()
plt.show()
