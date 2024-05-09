import numpy as np
import dcmri as dc


class AortaSignal8(dc.Model):

    pars = np.zeros(8)

    # Constants
    S0 = 1                     # Baseline signal (a.u.)
    dt = 0.5                   # Internal time resolution (sec)
    weight = 70.0              # Patient weight in kg
    dose = 0.025               # mL per kg bodyweight (quarter dose)
    rate = 1                   # Injection rate (mL/sec)
    dose_tolerance = 0.1
    field_strength = 3.0      # Field strength (T)
    agent = 'gadoterate'
    TR = 3.71/1000.0          # Repetition time (sec)
    FA = 15.0                 # Nominal flip angle (degrees)
    R10 = 1.0                 # Precontrast relaxation rate (1/sec)

    def predict(self, tacq,
            return_conc = False,
            return_rel = False):
        
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(tacq)+tacq[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) # mmol/sec
        Jb = dc.aorta_flux_6(Ji, # mmol/sec
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO  # M = (mmol/sec) / (mL/sec) 
        if return_conc:
            return dc.sample(t, cb, tacq, tacq[2]-tacq[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(t, R1, tacq, tacq[2]-tacq[1])
        signal = dc.signal_spgress(R1, self.S0, self.TR, self.FA)
        return dc.sample(t, signal, tacq, tacq[2]-tacq[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 100, 10, 0.2, 0.15, 20, 120, 0.05])
        else:
            return np.zeros(8)

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            lb = [0, 0, 0, 0.05, 0, 0, 0, 0.01]
            ub = [np.inf, np.inf, 30, 0.95, 0.5, 60, 800, 0.15]
        else:
            lb = -np.inf
            ub = +np.inf
        return (lb, ub)

    def pretrain(self, time, signal):

        # Estimate BAT from data
        T, D = self.pars[2], self.pars[3]
        BAT = time[np.argmax(signal)] - (1-D)*T
        self.pars[0] = BAT

        # Estimate S0 from data
        baseline = time[time <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        self.S0 = np.mean(signal[:baseline]) / Sref

    def pfree(self, units='standard'):
        pars = [
            ['BAT', 'Bolus arrival time', self.pars[0], "sec"], 
            ['CO', 'Cardiac output', self.pars[1], "mL/sec"], 
            ['T_hl', 'Heart-lung mean transit time', self.pars[2], "sec"],
            ['D_hl', 'Heart-lung transit time dispersion', self.pars[3], ""],
            ['E_o', "Extracellular extraction fraction", self.pars[4], ""],
            ['Tp_o', "Organs mean transit time", self.pars[5], "sec"],
            ['Te_o', "Extracellular mean transit time", self.pars[6], "sec"],
            ['E_b', "Body extraction fraction", self.pars[7], ""],
        ]
        if units == 'custom':
            pars[3][2:] = [100*self.pars[3], '%']
            pars[4][2:] = [100*self.pars[4], '%']
            pars[7][2:] = [100*self.pars[7], '%']
        return pars
    
    # rename to pdep
    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[5]+self.pars[6], 'sec'],
        ]
    
  
class AortaSignal8b(AortaSignal8):
    # Same as 8 except using chain model for Heart-Lung system

    def predict(self, tacq,
            return_conc = False,
            return_rel = False,
        ):
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(tacq)+tacq[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) #mmol/sec
        Jb = dc.aorta_flux_6b(Ji, # mmol/sec
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO  # M = (mmol/sec) / (mL/sec) 
        if return_conc:
            return dc.sample(t, cb, tacq, tacq[2]-tacq[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(t, R1, tacq, tacq[2]-tacq[1])
        signal = dc.signal_spgress(R1, self.S0, self.TR, self.FA)
        return dc.sample(t, signal, tacq, tacq[2]-tacq[1])
    

class AortaSignal8c(dc.Model):
    # Same as 8 except using SR signal model

    pars = np.zeros(8)

    # Constants
    S0 = 1                     # Baseline signal (a.u.)
    dt = 0.5                   # Internal time resolution (sec)
    weight = 70.0              # Patient weight in kg
    dose = 0.025               # mL per kg bodyweight (quarter dose)
    rate = 1                   # Injection rate (mL/sec)
    dose_tolerance = 0.1
    field_strength = 3.0       # Field strength (T)
    agent = 'gadoterate'
    TD = 180/1000.0            # Delay time (sec)
    R10 = 1.0                  # Precontrast relaxation rate (1/sec)

    def predict(self, xdata,
            return_conc = False,
            return_rel = False,
        ):
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent) #mmol/mL
        Ji = dc.influx_step(t, self.weight, conc, self.dose, self.rate, BAT) #mmol/sec
        Jb = dc.aorta_flux_6(Ji, #mmol/sec
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO  # M #mmol/sec / (mL/sec)
        if return_conc:
            return dc.sample(t, cb, xdata, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(t, R1, xdata, xdata[2]-xdata[1])
        signal = dc.signal_sr(R1, self.S0, self.TD)
        return dc.sample(t, signal, xdata, xdata[1]-xdata[0])
    
    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 100, 10, 0.2, 0.15, 20, 120, 0.05])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            lb = [0, 0, 0, 0.05, 0, 0, 0, 0.01]
            ub = [np.inf, np.inf, 30, 0.95, 0.5, 60, 800, 0.15]
        else:
            lb = -np.inf
            ub = +np.inf
        return (lb, ub)

    def pretrain(self, time, signal):

        # Estimate BAT from data
        T, D = self.pars[2], self.pars[3]
        BAT = time[np.argmax(signal)] - (1-D)*T
        self.pars[0] = BAT

        # Estimate S0 from data
        baseline = time[time <= BAT-20].size
        baseline = max([baseline, 1])
        Sref = dc.signal_sr(self.R10, 1, self.TD)
        self.S0 = np.mean(signal[:baseline]) / Sref

    def pfree(self, units='standard'):
        pars = [
            ['BAT', 'Bolus arrival time', self.pars[0], "sec"], 
            ['CO', 'Cardiac output', self.pars[1], "mL/sec"], 
            ['T_hl', 'Heart-lung mean transit time', self.pars[2], "sec"],
            ['D_hl', 'Heart-lung transit time dispersion', self.pars[3], ""],
            ['E_o', "Extracellular extraction fraction", self.pars[4], ""],
            ['Tp_o', "Organs mean transit time", self.pars[5], "sec"],
            ['Te_o', "Extracellular mean transit time", self.pars[6], "sec"],
            ['E_b', "Body extraction fraction", self.pars[7], ""],
        ]
        if units == 'custom':
            pars[3][2:] = [100*self.pars[3], '%']
            pars[4][2:] = [100*self.pars[4], '%']
            pars[7][2:] = [100*self.pars[7], '%']
        return pars
    
    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[5]+self.pars[6], 'sec'],
        ]
    

class AortaSignal11(dc.Model):

    pars = np.zeros(11)

    # Constants
    dt = 0.5                   # Internal time resolution (sec)
    weight = 70.0              # Patient weight in kg
    dose = [0.025, 0.025]               # mL per kg bodyweight (quarter dose)
    rate = 1                   # Injection rate (mL/sec)
    dose_tolerance = 0.1
    field_strength = 3.0      # Field strength (T)
    agent = 'gadoterate'
    TR = 3.71/1000.0          # Repetition time (sec)
    FA = 15.0                 # Nominal flip angle (degrees)
    R10 = 1.0                 # Precontrast relaxation rate (1/sec)
    R12 = 1.0
    tR12 = 1                  # 

    def predict(self, xdata,
            return_conc = False,
            return_rel = False,
        ):
        BAT1, BAT2, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b, S01, S02 = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        J1 = dc.influx_step(t, self.weight, conc, self.dose[0], self.rate, BAT1)
        J2 = dc.influx_step(t, self.weight, conc, self.dose[1], self.rate, BAT2)
        Jb = dc.aorta_flux_6(J1 + J2,
                T_hl, D_hl,
                E_o, Tp_o, Te_o,
                E_b, 
                dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/CO  # (mM)
        if return_conc:
            return dc.sample(t, cb, xdata, xdata[2]-xdata[1])
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*cb
        if return_rel:
            return dc.sample(t, R1, xdata, xdata[2]-xdata[1])
        signal = dc.signal_spgress(R1, S01, self.TR, self.FA)
        t2 = (t >= self.tR12)
        signal[t2] = dc.signal_spgress(R1[t2], S02, self.TR, self.FA)
        return dc.sample(t, signal, xdata, xdata[2]-xdata[1])

    def pars0(self, settings=None):
        if settings == 'TRISTAN':
            return np.array([60, 1200, 100, 10, 0.2, 0.15, 20, 120, 0.05, 1, 1])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings=None):
        if settings == 'TRISTAN':
            ub = [np.inf, np.inf, np.inf, 30, 0.95, 0.50, 60, 800, 0.15, np.inf, np.inf]
            lb = [0,0,0,0,0.05,0,0,0,0.01,0,0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
        
    def pretrain(self, xdata, ydata):

        # Use some notations
        k = xdata < self.tR12
        tdce1 = xdata[k]
        Sdce1 = ydata[k]
        k = xdata >= self.tR12
        tdce2 = xdata[k]
        Sdce2 = ydata[k]
        T, D = self.pars[3], self.pars[4]

        # Estimate BAT1 and S01 from data
        BAT1 = tdce1[np.argmax(Sdce1)] - (1-D)*T
        baseline = tdce1[tdce1 <= BAT1-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
        S01 = np.mean(Sdce1[:baseline]) / Sref

        # Estimate BAT2 and S02 from data
        BAT2 = tdce2[np.argmax(Sdce2)] - (1-D)*T
        baseline = tdce2[tdce2 <= BAT2-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_spgress(self.R12, 1, self.TR, self.FA)
        S02 = np.mean(Sdce2[:baseline]) / Sref

        # Format data in standard form
        self.pars[0] = BAT1
        self.pars[1] = BAT2
        self.pars[-2] = S01
        self.pars[-1] = S02

    def pfree(self, units='standard'):
        pars = [
            ['BAT1', "Bolus arrival time 1", self.pars[0], "sec"],
            ['BAT2', "Bolus arrival time 2", self.pars[1], "sec"],
            ['CO', "Cardiac output", self.pars[2], "mL/sec"], # 6 L/min = 100 mL/sec
            ['T_hl', "Heart-lung mean transit time", self.pars[3], "sec"],
            ['D_hl', "Heart-lung transit time dispersion", self.pars[4], ""],
            ['E_o', "Extracellular extraction fraction", self.pars[5], ""],
            ['Tp_o', "Organs mean transit time", self.pars[6], "sec"],
            ['Te_o', "Extracellular mean transit time", self.pars[7], "sec"],
            ['E_b',"Body extraction fraction", self.pars[8], ""],
            ['S01', "Signal amplitude S01", self.pars[9], "a.u."],
            ['S02', "Signal amplitude S02", self.pars[10], "a.u."],
        ]
        if units=='custom':
            pars[4][2:] = [pars[4][2]*100, '%']
            pars[5][2:] = [pars[5][2]*100, '%']
            pars[8][2:] = [pars[8][2]*100, '%']
        return pars

    def pdep(self, units='standard'):
        return [
            ['Tc', "Mean circulation time", self.pars[3]+self.pars[6], 'sec'],
        ]


