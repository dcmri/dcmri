"""
=====================================
A comparison of convolution functions
=====================================

Using the convolution functions `~dcmri.conv`, `~dcmri.expconv`, `~dcmri.biexpconv` and `~dcmri.nexpconv`. 
"""

# %%
import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import dcmri as dc

# %%
# Generate two normalized gaussian distributions f(t) and h(t) and use `~dcmri.conv` to convolve them. Compare the result to `numpy.convolve`:
t = np.linspace(0, 100, 50)
f = norm.pdf(t, 30, 5)
h = norm.pdf(t, 60, 10)
g = dc.conv(h, f, t)
g1 = np.convolve(h, f, mode='same')
plt.plot(t, f, 'r-', label='f(t)')
plt.plot(t, h, 'b-', label='h(t)')
plt.plot(t, g, 'k-', label='dcmri.conv')
plt.plot(g1, 'k:', label='numpy.convolve')
plt.title('Convolution of two gaussian distributions')
plt.legend()
plt.show()

# %%
# While there is clearly some relation between both results, they are not in any way similar. The `~numpy.convolve` result is shifted compared to `~dcmri.conv` and has a lower amplitude. This shows that caution is needed when applying convolution formulae from different libraries in a tracer-kinetic setting.


# %%
# In the special case where one of the factors is an exponential function, the function `~dcmri.expconv` is more accurate than `~dcmri.conv`, though the difference is small at this time resolution:
Tf = 20
f = np.exp(-t/Tf)/Tf
g0 = dc.conv(h, f, t)
g1 = dc.expconv(h, Tf, t)
plt.plot(t, f, 'r-', label='f(t)')
plt.plot(t, h, 'b-', label='h(t)')
plt.plot(t, g0, 'k-', label='conv()')
plt.plot(t, g1, color='gray', linestyle='-', label='expconv()')
plt.title('Comparison of conv() and expconv()')
plt.legend()
plt.show()

# %%
# However, `~dcmri.expconv` comes with a major improvement in computation time compared to `~dcmri.conv`, showing the that `~dcmri.expconv` should be used whenever applicable. We illustrate the effect by applying the functions 500 times and measuring the total computation time in each case:
start = time.time()
for _ in range(500):
    dc.conv(h, f, t)
print('Computation time for conv(): ', time.time()-start, 'sec')
start = time.time()
for _ in range(500):
    dc.expconv(h, Tf, t)
print('Computation time for expconv(): ', time.time()-start, 'sec')

# %%
# Incidentally since the time array in this case is uniform, `~dcmri.conv` can be accelerated by specifying dt instead of t in the arguments. However the performance remains far below `~dcmri.expconv`:
start = time.time()
for i in range(500):
    dc.conv(h, f, dt=t[1])
print('Computation time for conv() with uniform times: ', time.time()-start, 'sec')

# %%
# The difference in accuracy between `~dcmri.conv` and `~dcmri.expconv` becomes more apparent at lower temporal resolution but generally remains minor. Using 10 time points instead of 50 as above we start seeing some effect:
t = np.linspace(0, 120, 10)
h = norm.pdf(t, 60, 10)
f = np.exp(-t/Tf)/Tf
g0 = dc.conv(h, f, t)
g1 = dc.expconv(h, Tf, t)
plt.plot(t, f, 'r-', label='f(t)')
plt.plot(t, h, 'b-', label='h(t)')
plt.plot(t, g0, 'k-', label='conv()')
plt.plot(t, g1, color='gray', linestyle='-', label='expconv()')
plt.title('Comparison of conv() and expconv() at lower resolution')
plt.legend()
plt.show()

# %%
# If both functions are exponentials, convolution can be accelerated further with `~dcmri.biexpconv`, which uses an analytical formula to calculate the convolution: 
Th = 10
start = time.time()
for i in range(1000):
    dc.expconv(h, Tf, t)
print('Computation time for expconv(): ', time.time()-start, 'sec')
start = time.time()
for i in range(1000):
    dc.biexpconv(Th, Tf, t)
print('Computation time for biexpconv(): ', time.time()-start, 'sec')

# %%
# Using an analytical formula also comes with some improvements in accuracy, which is apparent at lower time resolution:
h = np.exp(-t/Th)/Th
g0 = dc.expconv(h, Tf, t)
g1 = dc.biexpconv(Th, Tf, t)
plt.plot(t, f, 'r-', label='f(t)')
plt.plot(t, h, 'b-', label='h(t)')
plt.plot(t, g0, 'k-', label='expconv()')
plt.plot(t, g1, color='gray', linestyle='-', label='biexpconv()')
plt.title('Comparison of expconv() and biexpconv()')
plt.legend()
plt.show()

# %%
# The final convolution function `~dcmri.nexpconv` convolves n indentical exponentials with mean transit time T analytically. We illustrate the result by keeping the total mean transit time MTT=nT constant, and increasing n from 1 to 100:
MTT = 30
t = np.linspace(0, 120, 500)
g1 = dc.nexpconv(1, MTT/1, t)
g10 = dc.nexpconv(10, MTT/10, t)
g100 = dc.nexpconv(100, MTT/100, t)
plt.plot(t, g1, 'r-', label='1 exponential')
plt.plot(t, g10, 'g-', label='10 exponentials')
plt.plot(t, g100, 'b-', label='100 exponentials')
plt.title('Convolutions of identical gaussian distributions')
plt.legend()
plt.show()

# %%
# As the number of exponentials increases, the convolution converges to a delta function positioned on t=MTT. This is because the system models a bolus transit through n identical compartments, which become increasing smaller at higher n and thus allow less bolus dispersion. A system like this is referred to as a chain in `~dcmri` which has the compartment (n=1) and the plug flow system (n=infty) as special cases.


# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
