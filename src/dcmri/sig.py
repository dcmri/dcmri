"""
Signals and measurement
"""
import math
import numpy as np

def signal_SPGRESS(TR: int,
                   FA: int,
                   R1: int,
                   S0: int
                   ) -> np.ndarray:
    """Signal aof a spoiled gradient echo sequence in the steady state"""
    E = np.exp(-TR*R1)
    cFA = np.cos(FA*math.pi/180)
    return S0 * (1-E) / (1-cFA*E)

def sample(t: np.ndarray,
           S: np.ndarray,
           ts: np.ndarray,
           dts: int
           ) -> np.ndarray:  
    """Sample the signal"""
    Ss = np.empty(len(ts)) 
    for k, tk in enumerate(ts):
        tacq = (t > tk) & (t < tk+dts)
        Ss[k] = np.average(S[np.nonzero(tacq)[0]])
    return Ss 


if __name__ == "__main__":
    pass