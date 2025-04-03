import matplotlib.pyplot as plt
import dcmri as dc

# Generate a stepwise injection:
t = np.arange(0, 120, 2.0)
Ji = dc.ca_injection(t, 70, 0.5, 0.2, 3, 30)

# Calculate the fluxes in mmol/sec:
Ja = dc.flux_aorta(Ji, t)

# Plot the fluxes:
plt.plot(t/60, Ja, 'r-', label='Aorta')
plt.xlabel('Time (min)')
plt.ylabel('Indicator flux (mmol/sec)')
plt.legend()
plt.show()