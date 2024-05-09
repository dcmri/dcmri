import numpy as np
import dcmri as dc




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
    agent = 'gadoterate'
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
    agent = 'gadoterate'
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
        signal = dc.signal_spgress(R1, self.S0, self.TR, self.FA)
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
        Sref = dc.signal_spgress(self.R10, 1, self.TR, self.FA)
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
    agent = 'gadoterate'
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
