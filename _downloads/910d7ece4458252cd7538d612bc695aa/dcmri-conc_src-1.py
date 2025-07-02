import matplotlib.pyplot as plt
import numpy as np
import dcmri as dc
#
# First define some constants:
#
T10 = 1         # sec
TC = 0.2        # sec
r1 = 0.005      # Hz/M
FA = 15         # deg
#
# Generate ground truth concentrations and signal data:
#
t = np.arange(0, 5*60, 0.1)     # sec
C = 0.003*(1-np.exp(-t/60))     # M
R1 = 1/T10 + r1*C               # Hz
S = dc.signal_free(100, R1, TC, FA)  # au
#
# Reconstruct the concentrations from the signal data:
#
Crec = dc.conc_src(S, TC, T10, r1)
#
# Check results by plotting ground truth against reconstruction:
#
plt.plot(t/60, 1000*C, 'ro', label='Ground truth')
plt.plot(t/60, 1000*Crec, 'b-', label='Reconstructed')
plt.title('SRC signal inverse')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.show()
