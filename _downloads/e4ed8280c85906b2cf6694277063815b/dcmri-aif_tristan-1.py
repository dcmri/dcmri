import matplotlib.pyplot as plt
import dcmri as dc

# Time points in sec
t = np.arange(0, 180, 2.0)

# Plot aifs with different levels of cardiac output, including the
# default of 100mL/sec.
plt.plot(t/60, 1000*dc.aif_tristan(t, BAT=30), 'r-', label='AIF (default)')
plt.plot(t/60, 1000*dc.aif_tristan(t, BAT=30, CO=150), 'g-',
         label='AIF (increased cardiac output)')
plt.plot(t/60, 1000*dc.aif_tristan(t, BAT=30, CO=75), 'b-',
         label='AIF (reduced cardiac output)')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (mmol/mL)')
plt.legend()
plt.show()