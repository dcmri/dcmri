import numpy as np
import dcmri as dc
t = [0, 5, 15, 30, 60]
J = [1, 2, 3, 3, 2]
print(dc.flux_pfcomp(J, 5, 0.2, t))

