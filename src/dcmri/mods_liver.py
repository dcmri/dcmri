import numpy as np
import dcmri as dc


class LiverSignal5(dc.Model):
    """ 
    Plug-flow two-compartment filtration model in fast wwater exchange acquired with a steady-state spoiled gradient echo sequence in steady state.
    """

    pars = np.zeros(5)

    # Constants
    dt = 0.5                    # Internal time resolution (sec)
    cb = None
    Hct = 0.45
    TR = 3.71/1000.0            # Repetition time (sec)
    FA = 15.0                   # Nominal flip angle (degrees)
    agent = 'Gadoxetate'
    R10 = 1
    field_strength = 3.0        # Field strength (T)  
    S0 = 1
    liver_volume = 800          # mL
    BAT = None                  # sec

    def predict(self, tacq,
            return_conc = False,
            return_rel = False,
        ): 
        Te, De, ve, k_he, Th = self.pars
        # Propagate through the gut
        ca = dc.flux_pfcomp(self.cb, Te, De, dt=self.dt, solver='interp')
        # Tissue concentration in the extracellular space
        Ce = ve*ca/(1-self.Hct)
        # Tissue concentration in the hepatocytes
        Ch = dc.conc_comp(k_he*ca, Th, dt=self.dt)
        t = self.dt*np.arange(len(self.cb))
        if return_conc:
            return (
                dc.sample(tacq, t, Ce, tacq[2]-tacq[1]),
                dc.sample(tacq, t, Ch, tacq[2]-tacq[1]))
        # Return R
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        R1 = self.R10 + rp*Ce + rh*Ch
        if return_rel:
            return dc.sample(tacq, t, R1, tacq[2]-tacq[1])
        signal = dc.signal_ss(R1, self.S0, self.TR, self.FA)
        return dc.sample(tacq, t, signal, tacq[1]-tacq[0])
    
    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([30.0, 0.85, 0.3, 20/6000, 30*60])
        else:
            return np.zeros(5)

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [60, 1, 0.6, np.inf, 10*60*60]
            lb = [0.1, 0, 0.01, 0, 10*60]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
    
    def pretrain(self, time, signal):
        BAT = self.BAT
        if BAT is None:
            BAT = time[np.argmax(signal)]-20
        baseline = time[time <= BAT].size
        baseline = max([baseline, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(signal[:baseline]) / Sref

    def pfree(self, units='standard'):
        pars = [
            # Inlets
            ['Te', "Extracellular transit time", self.pars[0], 'sec'], 
            ['De', "Extracellular dispersion", self.pars[1], ''],
            # Liver tissue
            ['ve', "Liver extracellular volume fraction", self.pars[2], 'mL/mL'],
            ['k_he', "Hepatocellular uptake rate", self.pars[3], 'mL/sec/mL'],
            ['Th', "Hepatocellular transit time", self.pars[4], 'sec'],
        ]
        if units == 'custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['k_bh', "Biliary excretion rate", (1-self.pars[2])/self.pars[4], 'mL/sec/mL'],
            ['Khe', "Hepatocellular tissue uptake rate", self.pars[3]/self.pars[2], 'mL/sec/mL'],
            ['Kbh', "Biliary tissue excretion rate", np.divide(1, self.pars[4]), 'mL/sec/mL'],
            ['CL_l', 'Liver blood clearance', self.pars[3]*self.liver_volume, 'mL/sec'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*60, 'mL/min']
        return pars
       

class LiverSignal9(dc.Model):
    """ 
    Serial arrangement of a plug-flow system and a compartment for the extracellular space with a non-stationary compartment for the hepatocytes, fast water exchange and a two-scan spoiled gradient echo acquisition in steady-state.
    """

    pars = np.zeros(9)

    dt = 0.5                   # Internal time resolution (sec)
    cb = None
    Hct = 0.45
    TR = 3.71/1000.0          # Repetition time (sec)
    FA = 15.0                  # Nominal flip angle (degrees)
    field_strength = 3.0      # Field strength (T)
    R10 = 1
    tR12 = 1
    agent = 'Gadoxetate'
    liver_volume = 800
    BAT1 = None
    R12 = None

    def predict(self, xdata,
            return_conc = False,
            return_rel = False,
            ):
        Te, De, ve, k_he_i, k_he_f, Th_i, Th_f, S01, S02 = self.pars
        #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
        t = self.dt*np.arange(len(self.cb))
        k_he = dc.interp([k_he_i, k_he_f], t)
        # Interpolating Kbh here for consistency with original model
        Kbh = dc.interp([1/Th_i, 1/Th_f], t)
        # Propagate through the gut
        ca = dc.flux_pfcomp(self.cb, Te, De, dt=self.dt, solver='interp')
        # Tissue concentration in the extracellular space
        Ce = ve*ca/(1-self.Hct)
        # Tissue concentration in the hepatocytes
        Ch = dc.conc_nscomp(k_he*ca, 1/Kbh, t)
        if return_conc:
            return (
                dc.sample(xdata, t, Ce, xdata[2]-xdata[1]),
                dc.sample(xdata, t, Ch, xdata[2]-xdata[1]))
        # Return R
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        R1 = self.R10 + rp*Ce + rh*Ch
        if return_rel:
            return dc.sample(xdata, t, R1, xdata[2]-xdata[1])
        signal = dc.signal_ss(R1, S01, self.TR, self.FA)
        t2 = (t >= self.tR12)
        signal[t2] = dc.signal_ss(R1[t2], S02, self.TR, self.FA)
        return dc.sample(xdata, t, signal, xdata[1]-xdata[0])
    
    def pretrain(self, xdata, ydata):

        # Use some notations
        t1 = xdata < self.tR12
        tdce1 = xdata[t1]
        Sdce1 = ydata[t1]
        t2 = xdata > self.tR12
        tdce2 = xdata[t2]
        Sdce2 = ydata[t2]

        # Estimate S01 from data
        baseline = tdce1[tdce1 <= self.BAT1]
        baseline = max([baseline.size, 1])
        Sref = dc.signal_ss(self.R10, 1, self.TR, self.FA)
        S01 = np.mean(Sdce1[:baseline]) / Sref

        # Estimate S02 from data
        baseline = int(np.floor(60/(tdce2[1]-tdce2[0])))
        Sref = dc.signal_ss(self.R12, 1, self.TR, self.FA)
        S02 = np.mean(Sdce2[:baseline]) / Sref

        self.pars[-2] = S01
        self.pars[-1] = S02

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([30, 0.85, 0.3, 20/6000, 20/6000, 30*60, 30*60, 1, 1])
        else:
            return np.zeros(9)

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [60, 1, 0.6, np.inf, np.inf, 10*60*60, 10*60*60, np.inf, np.inf]
            lb = [0, 0, 0.01, 0, 0, 10*60, 10*60, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            # Inlets
            ['Te', "Extracellular transit time", self.pars[0], 'sec'],
            ['De', "Extracellular dispersion", self.pars[1], ''],  
            # Liver tissue
            ['ve', "Liver extracellular volume fraction", self.pars[2], 'mL/mL'],
            ['k_he_i', "Hepatocellular uptake rate (initial)", self.pars[3], 'mL/sec/mL'],
            ['k_he_f', "Hepatocellular uptake rate (final)", self.pars[4], 'mL/sec/mL'],
            ['Th_i', "Hepatocellular transit time (initial)", self.pars[5], 'sec'],
            ['Th_f', "Hepatocellular transit time (final)", self.pars[6], 'sec'],
            ['S01', "Signal amplitude S01", self.pars[7], "a.u."],
            ['S02', "Signal amplitude S02", self.pars[8], "a.u."],
        ]
        if units=='custom':
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*100, 'mL/100mL']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]*6000, 'mL/min/100mL']
            pars[5][2:] = [pars[5][2]/60, 'min']
            pars[6][2:] = [pars[6][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        k_he = [self.pars[3], self.pars[4]]
        Kbh = [1/self.pars[5], 1/self.pars[6]]
        k_he_avr = np.mean(k_he)
        Kbh_avr = np.mean(Kbh)
        k_he_var = (np.amax(k_he)-np.amin(k_he))/k_he_avr
        Kbh_var = (np.amax(Kbh)-np.amin(Kbh))/Kbh_avr 
        k_bh = np.mean((1-self.pars[2])*Kbh_avr)
        Th = np.mean(1/Kbh_avr)
        pars = [
            ['k_he', "Hepatocellular uptake rate", k_he_avr, 'mL/sec/mL'],
            ['k_he_var', "Hepatocellular uptake rate variance", k_he_var, ''],
            ['Kbh', "Biliary tissue excretion rate", Kbh_avr, 'mL/sec/mL'],
            ['Kbh_var', "Biliary tissue excretion rate variance", Kbh_var, ''],
            ['k_bh', "Biliary excretion rate", k_bh, 'mL/sec/mL'],
            ['k_bh_i', "Biliary excretion rate (initial)", (1-self.pars[2])/self.pars[5], 'mL/sec/mL'],
            ['k_bh_f', "Biliary excretion rate (final)", (1-self.pars[2])/self.pars[6], 'mL/sec/mL'],
            ['Khe', "Hepatocellular tissue uptake rate", k_he_avr/self.pars[2], 'mL/sec/mL'],
            ['Th', "Hepatocellular transit time", Th, 'sec'],
            ['CL_l', 'Liver blood clearance', k_he_avr*self.liver_volume, 'mL/sec'],
        ]
        if units=='custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]*100, '%']
            pars[4][2:] = [pars[4][2]*6000, 'mL/min/100mL']
            pars[5][2:] = [pars[5][2]*6000, 'mL/min/100mL']
            pars[6][2:] = [pars[6][2]*6000, 'mL/min/100mL']
            pars[7][2:] = [pars[7][2]*6000, 'mL/min/100mL']
            pars[8][2:] = [pars[8][2]/60, 'min']
            pars[9][2:] = [pars[9][2]*60, 'mL/min']
        return pars