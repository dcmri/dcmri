import dcmri as dc
t, aif, _ = dc.fake_aif()
tissue = dc.Tissue('2CX', 'RR', aif=aif, t=t)
R1, v, Fw = tissue.relax()
#
v
# Expected:
## array([0.1, 0.3, 0.6])
#
Fw
# Expected:
## array([[0.02, 0.03, 0.  ],
##        [0.03, 0.  , 0.03],
##        [0.  , 0.03, 0.  ]])
#
import matplotlib.pyplot as plt
_ = plt.figure()
_ = plt.plot(t/60, R1[0,:], label='Blood')
_ = plt.plot(t/60, R1[1,:], label='Interstitium')
_ = plt.plot(t/60, R1[2,:], label='Cells')
_ = plt.xlabel('Time (min)')
_ = plt.ylabel('Relaxation rate (Hz)')
_ = plt.legend()
plt.show()
