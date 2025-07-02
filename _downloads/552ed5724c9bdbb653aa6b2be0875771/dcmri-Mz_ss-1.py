# Import packages:
#
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc
#
# Define constants:
#
FA, TR = 12, 0.005
R1 = 1
f, v = 0.5, 0.7
mi = np.linspace(-1, 1, 100)
#
# Compute magnetization without/with inflow:
#
m_c = dc.Mz_ss(R1, TR, FA, v)/v
m_f = [dc.Mz_ss(R1, TR, FA, v, f, j=f*m)/v for m in mi]
#
# Plot the results:
#
plt.plot(mi, mi*0+m_c, label='No inflow', linewidth=3)
plt.plot(mi, m_f, label='Inflow', linewidth=3)
plt.xlabel('Inflow magnetization (A/cm/mL)')
plt.ylabel('Steady-state magnetization (A/cm/mL)')
plt.legend()
plt.show()
#
# Note the magnetization of the two tissues is the same when the
# inflow is at the steady-state of the isolated tissue:
#
m_f = dc.Mz_ss(R1, TR, FA, v, f, j=f*m_c)/v
print(m_f - m_c)
# Expected:
## 0.00023950879616352339
#
# Now we consider the same situation again, this time for a two-
# compartment tissue with one central compartment that exchanges with
# the enviroment.
#
R1 = [0.5, 1.5]
v = [0.3, 0.6]
PS = 0.1
#
# Compute magnetization without inflow:
#
Fw = [[0, PS], [PS, 0]]
M_c = dc.Mz_ss(R1, TR, FA, v, Fw)
#
# Compute magnetization with flow through the first compartment:
#
Fw = [[f, PS], [PS, 0]]
M_f = np.zeros((2, 100))
for i, m in enumerate(mi):
    M_f[:,i] = dc.Mz_ss(R1, TR, FA, v, Fw, j=[f*m, 0])
#
# Plot the results for the central compartment:
#
plt.plot(mi, M_c[0]/v[0] + mi*0, label='No inflow', linewidth=3)
plt.plot(mi, M_f[0,:]/v[0], label='Inflow', linewidth=3)
plt.xlabel('Inflow magnetization (A/cm/mL)')
plt.ylabel('Steady-state magnetization (A/cm/mL)')
plt.legend()
plt.show()
#
# We can verify again that the magnetization is the same as that of the
# isolated tissue when the inflow is at the isolated steady-state:
#
m_c = M_c[0]/v[0]
M_f = dc.Mz_ss(R1, TR, FA, v, Fw=f, j=f*m_c)
print(M_f[0]/v[0] - m_c)
# Expected:
## 0.0002984954839493209
