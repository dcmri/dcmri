import numpy as np

class Input:
    
    def __init__(
        self, 
        signal: np.ndarray=None, 
        time:np.ndarray=None,
        dt=1.0,
        R10=0.7,
        B1corr=1.0,
    ):
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        if time is None:
            time = dt * np.arange(signal.size)

        self.signal = signal
        self.time = time
        self.R10 = R10
        self.B1corr = B1corr