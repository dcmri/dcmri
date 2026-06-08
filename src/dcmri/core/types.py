import numpy as np

class Input:
    
    def __init__(self, aif: dict):

        self.signal = None
        self.time = None
        self.R10 = 0.7
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

        if 'R10' in aif:
            self.R10 = aif['R10']

        if 'B1corr' in aif:
            self.B1corr = aif['B1corr']