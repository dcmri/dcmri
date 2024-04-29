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
    agent = 'Dotarem'
    TR = 3.71/1000.0          # Repetition time (sec)
    FA = 15.0                 # Nominal flip angle (degrees)
    R10 = 1.0                 # Precontrast relaxation rate (1/sec)

    def predict(self, tacq,
            return_conc = False,
            return_rel = False):
        
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(tacq)+tacq[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.inj_step(t, self.weight, conc, self.dose, self.rate, BAT) # mmol/sec
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
        signal = dc.signal_spgress(self.TR, self.FA, R1, self.S0)
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
        Sref = dc.signal_spgress(self.TR, self.FA, self.R10, 1)
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
        Ji = dc.inj_step(t, self.weight, conc, self.dose, self.rate, BAT) #mmol/sec
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
        signal = dc.signal_spgress(self.TR, self.FA, R1, self.S0)
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
    agent = 'Dotarem'
    TD = 180/1000.0            # Delay time (sec)
    R10 = 1.0                  # Precontrast relaxation rate (1/sec)

    def predict(self, xdata,
            return_conc = False,
            return_rel = False,
        ):
        BAT, CO, T_hl, D_hl, E_o, Tp_o, Te_o, E_b = self.pars
        t = np.arange(0, max(xdata)+xdata[1]+self.dt, self.dt)
        conc = dc.ca_conc(self.agent) #mmol/mL
        Ji = dc.inj_step(t, self.weight, conc, self.dose, self.rate, BAT) #mmol/sec
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
    agent = 'Dotarem'
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
        Ji = dc.inj_step(t, 
            self.weight, conc, self.dose[0], self.rate, BAT1,
            self.dose[1], BAT2)
        Jb = dc.aorta_flux_6(Ji,
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
        signal = dc.signal_spgress(self.TR, self.FA, R1, S01)
        t2 = (t >= self.tR12)
        signal[t2] = dc.signal_spgress(self.TR, self.FA, R1[t2], S02)
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
        Sref = dc.signal_spgress(self.TR, self.FA, self.R10, 1)
        S01 = np.mean(Sdce1[:baseline]) / Sref

        # Estimate BAT2 and S02 from data
        BAT2 = tdce2[np.argmax(Sdce2)] - (1-D)*T
        baseline = tdce2[tdce2 <= BAT2-20]
        baseline = max([baseline.size,1])
        Sref = dc.signal_spgress(self.TR, self.FA, self.R12, 1)
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


class KidneySignal6(dc.Model):

    pars = np.zeros(6)

    R10 = 1
    dt = 0.5                    # Internal time resolution (sec)
    cb = None
    TR = 3.71/1000.0            # Repetition time (sec)
    FA = 15.0                   # Nominal flip angle (degrees)
    Tsat = 0                    # time before start of readout
    TD = 85/1000                # Time to the center of the readout pulse
    field_strength = 3.0        # Field strength (T)
    agent = 'Dotarem'
    Hct = 0.45
    vol = 300 #mL
    CO = 120 #mL/sec

    def predict(self, xdata,
            return_conc = False):
        
        Fp, Tp, Ft, Tt, Ta, S0 = self.pars
        t = self.dt*np.arange(len(self.cb))
        vp = Tp*(Fp+Ft)
        ca = self.cb/(1-self.Hct)
        ca = dc.flux_plug(ca, Ta, t)
        cp = dc.conc_comp(Fp*ca, Tp, t)
        Cp = vp*cp
        Ct = dc.conc_comp(Ft*cp, Tt, t)
        if return_conc:
            return (
                dc.sample(t, Cp, xdata, xdata[2]-xdata[1]),
                dc.sample(t, Ct, xdata, xdata[2]-xdata[1]))
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*Cp + rp*Ct
        signal = dc.signal_srspgre(R1, S0, self.TR, self.FA, self.Tsat, self.TD)
        return dc.sample(t, signal, xdata, xdata[1]-xdata[0])
    
    def pars0(self, settings=None):
        if settings == 'iBEAt':
            return np.array([200/6000, 5, 30/6000, 120, 0, 1])
        else:
            return np.zeros(len(self.pars))

    def bounds(self, settings=None):
        if settings == 'iBEAt':
            ub = [np.inf, 8, np.inf, np.inf, 3, np.inf]
            lb = [0, 0, 0, 1, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)

    def pretrain(self, time, signal):

        # Estimate S0 from data
        t = self.dt*np.arange(len(self.cb))
        TTP = t[np.argmax(self.cb)]
        n0 = max([time[time<=TTP-30].size, 1])
        Sref = dc.signal_srspgre(self.R10, 1, self.TR, self.FA, self.Tsat, self.TD)
        self.pars[5] = np.mean(signal[:n0]) / Sref

    def pfree(self, units='standard'):
        # Fp, Tp, Ft, Tt, Ta, S0
        pars = [
            ['Fp','Plasma flow',self.pars[0],'mL/sec/mL'],
            ['Tp','Plasma mean transit time',self.pars[1],'sec'],
            ['Ft','Tubular flow',self.pars[2],'mL/sec/mL'],
            ['Tt','Tubular mean transit time',self.pars[3],'sec'],
            ['Ta','Arterial mean transit time',self.pars[4],'sec'],
            ['S0','Signal scaling factor',self.pars[5],'a.u.'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[2][2:] = [pars[2][2]*6000, 'mL/min/100mL']
            pars[3][2:] = [pars[3][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        pars = [
            ['Fb','Blood flow',self.pars[0]/(1-self.Hct),'mL/sec/mL'],
            ['ve', 'Extracellular volume', self.pars[0]*self.pars[1], ''],
            ['SKGFR', 'Single-kidney glomerular filtration rate', self.pars[2]*self.vol, 'mL/sec'],
            ['SKRBF', 'Single-kidney renal blood flow', self.pars[0]*self.vol/(1-self.Hct), 'mL/sec'],
            ['FF', 'Filtration fraction', self.pars[2]/self.pars[0], ''],
            ['E', 'Extraction fraction', self.pars[2]/(self.pars[0]+self.pars[2]), ''],
            ['fCO', 'Cardiac output fraction', self.vol*self.pars[0]/(1-self.Hct)/self.CO, ''],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, 'mL/100mL']
            pars[2][2:] = [pars[2][2]*60, 'mL/min']
            pars[3][2:] = [pars[3][2]*60, 'mL/min']
            pars[4][2:] = [pars[4][2]*100, 'mL/100mL']
            pars[5][2:] = [pars[5][2]*100, 'mL/100mL']
            pars[6][2:] = [pars[6][2]*100, 'mL/100mL']
        return pars


class KidneySignal9(dc.Model):

    pars = np.zeros(9)

    R10 = 1
    S0 = 1                     # Baseline signal (a.u.)
    dt = 0.5                   # Internal time resolution (sec)
    J_aorta = None
    TR = 3.71/1000.0           # Repetition time (sec)
    FA = 15.0                  # Nominal flip angle (degrees)
    field_strength = 3.0       # Field strength (T)
    agent = 'Dotarem'
    kidney_volume = None
    TT = [15,30,60,90,150,300,600]
    BAT = None
    CO = None
    Hct = 0.45

    def predict(self, xdata,
            return_conc = False,
            ):
        FF_k, F_k, Tv, h0,h1,h2,h3,h4,h5 = self.pars
        #t = np.arange(0, max(xdata)+xdata[1]+dt, dt)
        t = self.dt*np.arange(len(self.J_aorta))
        E_k = F_k/(1+F_k)
        Kvp = 1/Tv
        Kp = Kvp/(1-E_k)
        H = [h0,h1,h2,h3,h4,h5]
        J_kidneys = FF_k*self.J_aorta
        Np = dc.conc_plug(J_kidneys, Tv, t, solver='interp') 
        Nt = dc.conc_free(E_k*Kp*Np, H, dt=self.dt, TT=self.TT, solver='step')
        Cp = Np/self.kidney_volume # mM
        Ct = Nt/self.kidney_volume # mM
        if return_conc:
            return (
                dc.sample(t, Cp, xdata, xdata[2]-xdata[1]),
                dc.sample(t, Ct, xdata, xdata[2]-xdata[1]))
        # Return R
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1 = self.R10 + rp*Cp + rp*Ct
        signal = dc.signal_spgress(self.TR, self.FA, R1, self.S0)
        return dc.sample(t, signal, xdata, xdata[1]-xdata[0])
    
    def pars0(self, settings=None):
        if settings == 'Amber':
            return np.array([0.2, 0.1, 3.0, 1, 1, 1, 1, 1, 1])
        else:
            return np.zeros(9)

    def bounds(self, settings=None):
        if settings == 'Amber':
            ub = [1, 1, 10, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            lb = [0.01, 0, 1.0, 0, 0, 0, 0, 0, 0]
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)

    def pfree(self, units='standard'):
        pars = [
            ['FF_k', "Kidney flow fraction", self.pars[0], ""],
            ['F_k', "Kidney filtration fraction", self.pars[1], ""],
            ['Tv', "Vascular mean transit time", self.pars[2],"sec"],
            ['h0', "Transit time weight 1", self.pars[3], '1/sec'],
            ['h1', "Transit time weight 2", self.pars[4], '1/sec'],
            ['h2', "Transit time weight 3", self.pars[5], '1/sec'],
            ['h3', "Transit time weight 4", self.pars[6], '1/sec'],
            ['h4', "Transit time weight 5", self.pars[7], '1/sec'],
            ['h5', "Transit time weight 6", self.pars[8], '1/sec'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*100, '%']
            pars[1][2:] = [pars[1][2]*100, '%']
        return pars

    def pdep(self, units='standard'):
        E_k = np.divide(self.pars[1], 1+self.pars[1])
        H = self.pars[3:]
        tmax = self.dt*len(self.J_aorta)
        TTdesc = dc.res_free_desc(tmax, H, self.TT)
        pars = [
            ['E_k', "Kidney extraction fraction", E_k, ''],
            ['GFR', "Glomerular Filtration Rate", self.pars[0]*self.pars[1]*self.CO*(1-self.Hct), 'mL/sec'],  
            ['RBF', "Renal blood flow", self.pars[0]*self.CO, 'mL/sec'],
            ['RP', "Renal perfusion", self.pars[0]*self.CO/self.kidney_volume, 'mL/sec/mL'],
            ['RBV', "Renal blood volume", self.pars[0]*self.CO*self.pars[2]/self.kidney_volume, 'mL/mL'],
            ['MTTt', "Tubular mean transit time", TTdesc['mean'], 'sec'],
            ['TTDt', "Tubular transit time dispersion", 100*TTdesc['stdev']/TTdesc['mean'], '%'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*100, '%']
            pars[1][2:] = [pars[1][2]*60, 'mL/min']
            pars[2][2:] = [pars[2][2]*60, 'mL/min']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]*100, 'mL/100mL']
            pars[5][2:] = [pars[5][2]/60, 'min']
            pars[6][2:] = [pars[6][2]*100, '%']
        return pars

    def pretrain(self, xdata, ydata):
        # Estimate S0 from data
        baseline = xdata[xdata <= self.BAT]
        baseline = max([baseline.size, 1])
        Sref = dc.signal_spgress(self.TR, self.FA, self.R10, 1)
        self.S0 = np.mean(ydata[:baseline]) / Sref
    

class KidneyCMSignal9(dc.Model):

    pars = np.zeros(9)

    R10c = 1
    R10m = 1
    S0c = 1                    # Baseline signal cortex (a.u.)
    S0m = 1                    # Baseline signal medulla (a.u.)
    dt = 0.5                   # Internal time resolution (sec)
    cb = None
    TR = 3.71/1000.0           # Repetition time (sec)
    FAc = 15.0                  # Flip angle cortex (degrees)
    FAm = 15.0                  # Flip angle medulla (degrees)
    Tsat = 0                  # time before start of readout
    TD = 85/1000               # Time to the center of the readout pulse
    field_strength = 3.0       # Field strength (T)
    agent = 'Dotarem'
    Hct = 0.45
    CO = None
    vol = None

    def predict(self, tacq, return_conc=False):
        Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd = self.pars
        t = self.dt*np.arange(len(self.cb))
        ca = self.cb/(1-self.Hct)
        Cc, Cm = dc.kidney_conc_9(
            ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, dt=self.dt)
        if return_conc:
            return (
                dc.sample(t, Cc, tacq, tacq[2]-tacq[1]),
                dc.sample(t, Cm, tacq, tacq[2]-tacq[1]))
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        R1c = self.R10c + rp*Cc
        R1m = self.R10m + rp*Cm
        Sc = dc.signal_srspgre(R1c, self.S0c, self.TR, self.FAc, self.Tsat, self.TD)
        Sm = dc.signal_srspgre(R1m, self.S0m, self.TR, self.FAm, self.Tsat, self.TD)
        nt = int(len(tacq)/2)
        Sc = dc.sample(t, Sc, tacq[:nt], tacq[2]-tacq[1])
        Sm = dc.sample(t, Sm, tacq[nt:], tacq[2]-tacq[1])
        return np.concatenate((Sc, Sm))
    
    def pars0(self, settings=None):
        if settings == 'iBEAt':
            return np.array([200/6000, 0.15, 0.8, 4, 10, 60, 60, 30, 30])
        else:
            return np.zeros(9)

    def bounds(self, settings=None):
        if settings == 'iBEAt':
            ub = [1, 1, 1, 10, 30, np.inf, np.inf, np.inf, np.inf]
            lb = [0.01, 0, 0, 0, 0, 0, 0, 0, 0] 
        else:
            ub = +np.inf
            lb = -np.inf
        return (lb, ub)
    
    def pretrain(self, xdata, ydata):
        nt = int(len(xdata)/2)
        time = xdata[:nt]
        Sc, Sm = ydata[:nt], ydata[nt:]
        # Determine S0 from data
        t = self.dt*np.arange(len(self.cb))
        BAT = t[np.argmax(self.cb)]
        baseline = time[time <= BAT-20]
        n0 = max([baseline.size,1])
        Scref = dc.signal_srspgre(self.R10c, 1, self.TR, self.FAc, 0, self.TD)
        Smref = dc.signal_srspgre(self.R10m, 1, self.TR, self.FAm, 0, self.TD)
        self.S0c = np.mean(Sc[:n0]) / Scref
        self.S0m = np.mean(Sm[:n0]) / Smref

    def pfree(self, units='standard'):
        pars = [
            ['Fp','Plasma flow',self.pars[0],'mL/sec/mL'], 
            ['Eg','Glomerular extraction fraction',self.pars[1],''], 
            ['fc','Cortical flow fraction',self.pars[2],''], 
            ['Tg','Glomerular mean transit time',self.pars[3],'sec'], 
            ['Tv','Peritubular & venous mean transit time',self.pars[4],'sec'], 
            ['Tpt','Proximal tubuli mean transit time',self.pars[5],'sec'], 
            ['Tlh','Lis of Henle mean transit time',self.pars[6],'sec'], 
            ['Tdt','Distal tubuli mean transit time',self.pars[7],'sec'], 
            ['Tcd','Collecting duct mean transit time',self.pars[8],'sec'],
        ]
        if units == 'custom':
            pars[0][2:] = [pars[0][2]*6000, 'mL/min/100mL']
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*100, '%']
            pars[5][2:] = [pars[5][2]/60, 'min']
            pars[6][2:] = [pars[6][2]/60, 'min']
            pars[7][2:] = [pars[7][2]/60, 'min']
            pars[8][2:] = [pars[8][2]/60, 'min']
        return pars
    
    def pdep(self, units='standard'):
        p = self.pars
        pars = [
            ['FF', 'Filtration fraction', p[1]/(1-p[1]), ''],
            ['Ft', 'Tubular flow', p[1]/(1-p[1])*p[0], 'mL/sec/mL'],
            ['SKGFR','Single-kidney glomerular filtration rate', p[1]/(1-p[1])*p[0]*self.vol,'mL/sec'],
            ['MBF','Medullary blood flow', p[2]*(1-p[1])*p[0]/(1-self.Hct), 'mL/sec/mL'],
            ['SKBF', 'Single-kidney blood flow', self.vol*p[0]/(1-self.Hct), 'mL/sec'],
            ['SKMBF', 'Single-kidney medullary blood flow', self.vol*p[2]*(1-p[1])*p[0]/(1-self.Hct), 'mL/sec'],
            ['fCO', 'Cardiac output fraction', self.vol*p[0]/(1-self.Hct)/self.CO, ''],
            ['CBF', 'Cortical blood flow', p[0]/(1-self.Hct), 'mL/sec/mL'],
        ]
        if units=='custom':
            pars[0][2:] = [pars[0][2]*100, '%']
            pars[1][2:] = [pars[1][2]*100, '%']
            pars[2][2:] = [pars[2][2]*60, 'mL/min']
            pars[3][2:] = [pars[3][2]*6000, 'mL/min/100mL']
            pars[4][2:] = [pars[4][2]*60, 'mL/min']
            pars[5][2:] = [pars[5][2]*60, 'mL/min']
            pars[6][2:] = [pars[6][2]*100, '%']
            pars[7][2:] = [pars[7][2]*6000, 'mL/min/100mL']
        return pars


class LiverSignal5(dc.Model):

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
                dc.sample(t, Ce, tacq, tacq[2]-tacq[1]),
                dc.sample(t, Ch, tacq, tacq[2]-tacq[1]))
        # Return R
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        R1 = self.R10 + rp*Ce + rh*Ch
        if return_rel:
            return dc.sample(t, R1, tacq, tacq[2]-tacq[1])
        signal = dc.signal_spgress(self.TR, self.FA, R1, self.S0)
        return dc.sample(t, signal, tacq, tacq[1]-tacq[0])
    
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
        Sref = dc.signal_spgress(self.TR, self.FA, self.R10, 1)
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
        k_he = dc.interp(t, [k_he_i, k_he_f])
        # Interpolating Kbh here for consistency with original model
        Kbh = dc.interp(t, [1/Th_i, 1/Th_f])
        # Propagate through the gut
        ca = dc.flux_pfcomp(self.cb, Te, De, dt=self.dt, solver='interp')
        # Tissue concentration in the extracellular space
        Ce = ve*ca/(1-self.Hct)
        # Tissue concentration in the hepatocytes
        Ch = dc.conc_nscomp(k_he*ca, 1/Kbh, t)
        if return_conc:
            return (
                dc.sample(t, Ce, xdata, xdata[2]-xdata[1]),
                dc.sample(t, Ch, xdata, xdata[2]-xdata[1]))
        # Return R
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        R1 = self.R10 + rp*Ce + rh*Ch
        if return_rel:
            return dc.sample(t, R1, xdata, xdata[2]-xdata[1])
        signal = dc.signal_spgress(self.TR, self.FA, R1, S01)
        t2 = (t >= self.tR12)
        signal[t2] = dc.signal_spgress(self.TR, self.FA, R1[t2], S02)
        return dc.sample(t, signal, xdata, xdata[1]-xdata[0])
    
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
        Sref = dc.signal_spgress(self.TR, self.FA, self.R10, 1)
        S01 = np.mean(Sdce1[:baseline]) / Sref

        # Estimate S02 from data
        baseline = int(np.floor(60/(tdce2[1]-tdce2[0])))
        Sref = dc.signal_spgress(self.TR, self.FA, self.R12, 1)
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