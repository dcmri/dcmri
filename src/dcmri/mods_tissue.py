import numpy as np
import dcmri as dc

class TissueSignal3(dc.Model):
    """Extended Tofts model with fast water exchange and spoiled gradient echo in steady state.

    Attributes:
        pars (nump.ndarray): free parameters (vp, Ktrans, ve). Defaults to np.zeros(3).
        cb (numpy.ndarray): uniformly sampled blood concentrations at the inlet in M.
        dt (float): time interval between points in cb in dec. Defaults to 0.5 sec.
        Hct (float): Hematocrit. Defaults to 0.45.
        R10 (float): baseline R1 in 1/sec. Defaults to 1/sec.
        field_strength (float): field strength in Tesla. Defaults to 3T.
        agent (str): contrast agent generic name. Defaults to 'gadoterate'.
        S0 (float): Signal scaling factor. Defaults to 1.
        TR (float): Repetition time or time between excitation pulses in sec. Defaults to 5 msec.
        FA (float): Flip angle in degrees. Defaults to 15 deg.
    """

    pars = np.zeros(3)

    cb = None
    dt = 0.5                    # Internal time resolution (sec)
    Hct = 0.45
    R10 = 1
    field_strength = 3.0        # Field strength (T)
    agent = 'gadoterate'    
    S0 = 1
    TR = 3.0/1000.0            # Repetition time (sec)
    FA = 15.0                   # Nominal flip angle (degrees)

    def predict(self, tacq:np.ndarray, return_conc=False)->np.ndarray:
        """Predict signal-time curve.

        Args:
            tacq (np.ndarray): time points (sec) where the signals are to be calculated.

        Returns:
            np.ndarray: signal-time curve, same length as tacq.
        """
        t = self.dt*np.arange(len(self.cb))
        if np.amax(t) <= np.amax(tacq):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce tacq.'
            raise ValueError(msg)
        vp, Ktrans, ve = self.pars
        ca = self.cb/(1-self.Hct)
        C = dc.conc_etofts(ca, vp, Ktrans, ve, dt=self.dt)
        if return_conc:
            return dc.sample(t, C, tacq, tacq[2]-tacq[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*C
        signal = dc.signal_spgress(R1, self.S0, self.TR, self.FA)
        return dc.sample(t, signal, tacq, tacq[2]-tacq[1])
    
    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([0.1, 0.1/60, 0.2])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings='default'):
        if settings=='default':
            ub = [1, np.inf, 1]
            lb = [0, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
    
    def train(self, tacq, signal, baseline=1, **kwargs):
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(signal[:baseline]) / Sref
        return super().train(tacq, signal, **kwargs)

    # def pretrain(self, signal, baseline=1):
    #     Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
    #     self.S0 = np.mean(signal[:baseline]) / Sref
    #     return self

    def pfree(self, units='standard'):
        # vp, Ktrans, ve
        pars = [
            ['vp','Plasma volume',self.pars[0],'mL/mL'],
            ['Ktrans','Volume transfer constant',self.pars[1],'1/sec'],
            ['ve','Extravascular extracellular volume',self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*100, 'mL/100mL']
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Te','Extracellular mean transit time',self.pars[2]/self.pars[1],'sec'],
            ['kep','Extravascular transfer constant',self.pars[1]/self.pars[2],'1/sec'],
            ['v','Extracellular volume',self.pars[0]+self.pars[2],'mL/mL'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]/60, 'min']
            pars[1][2:] = [pars[1][2]*60, '1/min']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
        return pars
    
class TissueSignal3b(TissueSignal3):
    # No water exchange

    def predict(self, tacq:np.ndarray)->np.ndarray:
        t = self.dt*np.arange(len(self.cb))
        if np.amax(t) <= np.amax(tacq):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce tacq.'
            raise ValueError(msg)
        vp, Ktrans, ve = self.pars
        vb = vp/(1-self.Hct)
        ca = self.cb/(1-self.Hct)
        C = dc.conc_etofts(ca, vp, Ktrans, ve, dt=self.dt, sum=False)
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1b = self.R10 + rp*C[0,:]/vb
        R1e = self.R10 + rp*C[1,:]/ve
        R1c = self.R10 + np.zeros(C.shape[1])
        v = [vb, ve, 1-vb-ve]
        R1 = np.stack((R1b, R1e, R1c))
        signal = dc.signal_spgress_nex(v, R1, self.S0, self.TR, self.FA)
        return dc.sample(t, signal, tacq, tacq[2]-tacq[1])


class TissueSignal5(dc.Model):
    # Intermediate water exchange

    pars = np.zeros(5)

    cb = None
    dt = 0.5                    # Internal time resolution (sec)
    Hct = 0.45
    R10 = 1
    field_strength = 3.0        # Field strength (T)
    agent = 'gadoterate'    
    S0 = 1
    TR = 4.0/1000.0            # Repetition time (sec)
    FA = 15.0                   # Nominal flip angle (degrees)

    def predict(self, tacq:np.ndarray)->np.ndarray:
        t = self.dt*np.arange(len(self.cb))
        if np.amax(t) <= np.amax(tacq):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'Possible solutions: (1) increase dt; (2) extend cb; (3) reduce tacq.'
            raise ValueError(msg)
        vp, Ktrans, ve, PSbe, PSec = self.pars
        vb = vp/(1-self.Hct)
        ca = self.cb/(1-self.Hct)
        C = dc.conc_etofts(ca, vp, Ktrans, ve, dt=self.dt, sum=False)
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1b = self.R10 + rp*C[0,:]/vb
        R1e = self.R10 + rp*C[1,:]/ve
        R1c = self.R10 + np.zeros(C.shape[1])
        PS = np.array([[0,PSbe,0],[PSbe,0,PSec],[0,PSec,0]])
        v = [vb, ve, 1-vb-ve]
        R1 = np.stack((R1b, R1e, R1c))
        signal = dc.signal_spgress_wex(PS, v, R1, self.S0, self.TR, self.FA)
        return dc.sample(t, signal, tacq, tacq[2]-tacq[1])

    def pars0(self, settings='default'):
        if settings == 'default':
            return np.array([0.1, 0.1/60, 0.2, 10, 10])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings='default'):
        if settings == 'default':
            ub = [1, np.inf, 1, np.inf, np.inf]
            lb = [0, 0, 0, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
    
    def train(self, tacq, signal, baseline=1, **kwargs):
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(signal[:baseline]) / Sref
        return super().train(tacq, signal, **kwargs)