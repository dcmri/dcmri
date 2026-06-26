import numpy as np

class Input:
    
    def __init__(self, aif: dict):

        self.signal = None
        self.time = None
        self.R1b = 0.7
        self.B1corr = 1.0

        if 'signal' in aif:
            self.signal = np.array(aif['signal'])
        else:
            raise ValueError('signal is required to construct an aif.')
        
        if 'time' in aif:
            self.time = np.array(aif['time'])
        elif 'dt' in aif:
            self.time = aif['dt'] * np.arange(self.signal.size)
        else:
            raise ValueError('Either time or dt must be provided to construct an AIF.')

        if 'R1b' in aif:
            self.R1b = aif['R1b']

        if 'B1corr' in aif:
            self.B1corr = aif['B1corr']