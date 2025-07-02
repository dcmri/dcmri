import dcmri as dc
import matplotlib.pyplot as plt
#
t, aif, _ = dc.fake_aif()
tissue = dc.Tissue('HFU', 'RR', aif=aif, t=t)
C = tissue.conc(sum=False)
#
_ = plt.figure()
_ = plt.plot(t/60, 1e3*C[0,:], label='Plasma')
_ = plt.plot(t/60, 1e3*C[1,:], label='Interstitium')
_ = plt.xlabel('Time (min)')
_ = plt.ylabel('Concentration (mM)')
_ = plt.legend()
_ = plt.show()
