import os
import math
import numpy as np
import matplotlib.pyplot as plt

import dcmri
from curvefit import CurveFit


class Aorta(CurveFit):

    xname = 'Time'
    xunit = 'sec'
    yname = 'MRI Signal'
    yunit = 'a.u.'

    def function(self, x, p):

        #R1 = self.R1()
        #self.signal = dcmri.signalSPGRESS(self.TR, p.FA, R1, p.S0)
        R1 = self.R1()
        self.signal = dcmri.signal_genflash_with_sat(self.TI, self.TSAT, self.TR, p.FA, R1, p.S0)
        #self.signal = dcmri.signal_monoExp_aorta(R1, p.S0)
        return dcmri.sample(self.t, self.signal, x, self.tacq)

    def parameters(self):

        return [ 
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA', "Flip angle", self.FA, "deg", 0, 180, False, 4],
            ['BAT', "Bolus arrival time", 60, "sec", 0, np.inf, True, 3],
            ['CO', "Cardiac output", 3000.0/60.0, "mL/sec", 0, np.inf, True, 3],
            ['MTThl', "Heart & lung mean transit time", 6.0, "sec", 0, np.inf, True, 2],
            ['MTTo', "Other organs mean transit time", 20.0, "sec", 0, np.inf, True, 2],
            ['TTDo', "Other organs transit time dispersion", 10.0, "sec", 0, np.inf, True, 2],
            ['MTTe', "Extravascular mean transit time", 120.0, "sec", 0, np.inf, True, 3],
            ['El', "Leakage fraction", 0.5, "", 0, 1, True, 3],
            ['Ee', "Extraction fraction", 0.2,"", 0, 1, True, 3],
        ]

    # Internal time resolution & acquisition time
    dt = 1.0                # sec
    tmax = 40*60.0          # Total acquisition time (sec)

    # Default values for experimental parameters
    tacq = 1.61             # Time to acquire a single datapoint (sec)
    field_strength = 3.0    # Field strength (T)
    weight = 70.0           # Patient weight in kg
    conc = 0.5             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf) #DOTOREM = 0.5mmol/ml, Gadovist = 1.0mmol/ml 
    dose = 0.05            # mL per kg bodyweight (quarter dose)
    rate = 2                # Injection rate (mL/sec)
    TR = 2.2/1000.0        # Repetition time (sec)
    FA = 10.0               # Nominal flip angle (degrees)
    TI = 85/1000
    TSAT = 25.5/1000

    # Physiological parameters
    Hct = 0.45

    @property
    def rp(self):
        field = math.floor(self.field_strength)
        if field == 1.5: return 8.1     # relaxivity of hepatocytes in Hz/mM
        if field == 3.0: return 6.4     # relaxivity of hepatocytes in Hz/mM
        if field == 4.0: return 6.4     # relaxivity of blood in Hz/mM
        if field == 7.0: return 6.2     # relaxivity of blood in Hz/mM
        if field == 9.0: return 6.1     # relaxivity of blood in Hz/mM 

    @property
    def t(self): # internal time
        return np.arange(0, self.tmax+self.dt, self.dt) 

    @property
    def R10lit(self):
        field = math.floor(self.field_strength)
        if field == 1.5: return 1000.0 / 1480.0         # aorta R1 in 1/sec 
        if field == 3.0: return 0.52 * self.Hct + 0.38  # Lu MRM 2004 

    def R1(self):

        p = self.p.value
        Ji = dcmri.injection(self.t, 
            self.weight, self.conc, self.dose, self.rate, p.BAT)
        _, Jb = dcmri.propagate_simple_body(self.t, Ji, 
            p.MTThl, p.El, p.MTTe, p.MTTo, p.TTDo, p.Ee)
        self.cb = Jb*1000/p.CO  # (mM)
        return self.R10 + self.rp * self.cb

    def signal_smooth(self):
        R1 = self.R1()
        #return dcmri.signalSPGRESS(self.TR, self.p.value.FA, R1, self.p.value.S0)
        return dcmri.signal_genflash_with_sat(self.TI, self.TSAT, self.TR, self.p.value.FA, R1, self.p.value.S0)
        #return dcmri.signal_monoExp_aorta(R1, self.p.value.S0)
       
    def set_x(self, x):
        self.x = x
        self.tacq = x[1]-x[0]
        self.tmax = x[-1] + self.tacq

    def set_R10(self, t, R1):
        self.R10 = R1

    def estimate_p(self):

        BAT = self.x[np.argmax(self.y)]
        baseline = np.nonzero(self.x <= BAT-20)[0]
        n0 = baseline.size
        n0 = 10
        if n0 == 0: 
            n0 = 1
        #Sref = dcmri.signalSPGRESS(self.TR, self.p.value.FA, self.R10, 1)
        Sref = dcmri.signal_genflash_with_sat(self.TI, self.TSAT, self.TR, self.p.value.FA, self.R10, 1)
        #Sref = dcmri.signal_monoExp_aorta(self.R10, 1)
        S0 = np.mean(self.y[:n0]) / Sref
        
        self.p.value.S0 = S0
        self.p.value.BAT = BAT

    def plot_fit(self, show=True, save=False, path=None):

        self.plot_with_conc(show=show, save=save, path=path)
        BAT = self.p.value.BAT
        self.plot_with_conc(xrange=[BAT-20, BAT+160], show=show, save=save, path=path)

    def plot_with_conc(self, fit=True, xrange=None, show=True, save=False, path=None, legend=True):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.t[0]
            t1 = self.t[-1]
            win_str = ''
        else:
            t0 = xrange[0]
            t1 = xrange[1]
            win_str = ' [' + str(round(t0)) + ', ' + str(round(t1)) + ']'
        if path is None:
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        name = self.__class__.__name__
        ti = np.nonzero((self.t>=t0) & (self.t<=t1))[0]
        xi = np.nonzero((self.x>=t0) & (self.x<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle(name + " - model fit" + win_str)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(self.x[xi]+self.tacq/2, y[xi], 'ro', label='data')
        ax1.plot(self.t[ti], self.signal_smooth()[ti], 'b-', label='fit')
        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.t[ti], 0*self.t[ti], color='black')
        ax2.plot(self.t[ti], self.cb[ti], 'b-', label=self.plabel())
        if legend:
            ax2.legend()
        if save:          
            plt.savefig(fname=os.path.join(path, name + ' fit ' + win_str + '.png'))
        if show:
            plt.show()
        else:
            plt.close()


class Kidney(CurveFit):

    def function(self, x, p):

        R1 = self.R1()
        #self.signal = dcmri.signalSPGRESS(self.aorta.TR, p.FA, R1, p.S0)
        self.signal = dcmri.signal_genflash_with_sat(self.TI, self.TSAT, self.aorta.TR, p.FA, R1, p.S0)

        return dcmri.sample(self.aorta.t, self.signal, x, self.aorta.tacq)

    def parameters(self):

        return [
            ['S0', "Signal amplitude S0", 1200, "a.u.", 0, np.inf, False, 4],
            ['FA', "Flip angle", self.aorta.FA, "deg", 0, 180, False, 4],
            ['Fp', "Renal Plasma Flow", 0.55*2.5/60, 'mL/sec/mL', 0, np.inf, True, 5],
            ['E', "Glomerular Extraction Fraction", 0.15, '', 0, 1, True, 5],
            ['MTTp', "Plasma mean transit time", 8.0, 'sec', 0, np.inf, True, 5],
            ['MTTt', "Tubular mean transit time", 120, 'sec', 0, np.inf, True, 3],
            ['MTTa', "Arterial transit time", 2.0, 'sec', 0, np.inf, True, 3],
        ]

    def __init__(self, aorta):

        self.aorta = aorta
        self.aorta.signal_smooth()
        super().__init__()

    # Kidney constants parameters
    vp = 0.15         # Kidney plasma volume (mL/mL)
    vt = 0.60         # Kidney tubular volume (mL/mL)
    f = 0.99        # reabsorption fraction
    TR = 2.2/1000.0        # Repetition time (sec)
    FA = 10.0               # Nominal flip angle (degrees)
    TI = 85/1000
    TSAT = 25.5/1000

    @property
    def R10lit(self):
        field = math.floor(self.aorta.field_strength)
        if field == 1.5: return 1000.0/602.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 3.0: return 1000.0/752.0     # liver R1 in 1/sec (Waterton 2021)
        if field == 4.0: return 1.281     # liver R1 in 1/sec (Changed from 1.285 on 06/08/2020)
        if field == 7.0: return 1.109     # liver R1 in 1/sec (Changed from 0.8350 on 06/08/2020)
        if field == 9.0: return 0.920     # per sec - liver R1 (https://doi.org/10.1007/s10334-021-00928-x)

    def R1(self):

        p = self.p.value
        ca = dcmri.propagate_delay(self.aorta.t, self.aorta.cb, p.MTTa)
        
        ca = ca/self.aorta.Hct
        cp = dcmri.propagate_compartment(self.aorta.t, ca, p.MTTp)
        ct = dcmri.propagate_compartment(self.aorta.t, cp, p.MTTt)
        np = cp*p.MTTp*p.Fp
        nt = ct*p.MTTt*p.Fp*p.E

        self.ca = ca
        self.cp = np/self.vp
        self.ct = nt/self.vt
        self.ck = (np+nt)/(self.vp+self.vt)

        return self.R10 + self.aorta.rp*(np + nt)

    def signal_smooth(self):

        R1 = self.R1()
        #return dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, R1, self.p.value.S0)
        return dcmri.signal_genflash_with_sat(self.TI, self.TSAT, self.aorta.TR, self.p.value.FA, R1, self.p.value.S0)


    def set_R10(self, t, R1):
        self.R10 = R1

    def estimate_p(self):

        baseline = np.nonzero(self.x <= self.aorta.p.value.BAT-5)[0]
        n0 = baseline.size
        n0 = 5
        if n0 == 0: 
            n0 = 1
        #Sref = dcmri.signalSPGRESS(self.aorta.TR, self.p.value.FA, self.R10, 1)
        Sref = dcmri.signal_genflash_with_sat(self.TI, self.TSAT, self.aorta.TR, self.p.value.FA, self.R10, 1)
        S0 = np.mean(self.y[:n0]) / Sref
        self.p.value.S0 = S0

    def plot_fit(self, show=True, save=False, path=None):

        self.plot_with_conc(show=show, save=save, path=path)
        BAT = self.aorta.p.value.BAT
        self.plot_with_conc(xrange=[BAT-20, BAT+160], show=show, save=save, path=path)
        self.plot_with_conc(xrange=[BAT-20, BAT+600], show=show, save=save, path=path)
        self.plot_with_conc(xrange=[BAT-20, BAT+1200], show=show, save=save, path=path)

    def plot_with_conc(self, fit=True, xrange=None, show=True, save=False, path=None, legend=True):

        if fit is True:
            y = self.y
        else:
            y = self.yp
        if xrange is None:
            t0 = self.aorta.t[0]
            t1 = self.aorta.t[-1]
            win_str = ''
        else:
            t0 = xrange[0]
            t1 = xrange[1]
            win_str = ' [' + str(round(t0)) + ', ' + str(round(t1)) + ']'
        if path is None:
            path = self.path()
        if not os.path.isdir(path):
            os.makedirs(path)
        name = self.__class__.__name__
        ti = np.nonzero((self.aorta.t>=t0) & (self.aorta.t<=t1))[0]
        xi = np.nonzero((self.x>=t0) & (self.x<=t1))[0]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))
        fig.suptitle(name + " - model fit" + win_str)

        ax1.set_title('Signal')
        ax1.set(xlabel=self.xlabel, ylabel=self.ylabel)
        ax1.plot(self.x[xi]+self.aorta.tacq/2, y[xi], 'ro', label='data')
        ax1.plot(self.aorta.t[ti], self.signal_smooth()[ti], 'b-', label=self.plabel())

        if legend:
            ax1.legend()

        ax2.set_title('Reconstructed concentration')
        ax2.set(xlabel=self.xlabel, ylabel='Concentration (mM)')
        ax2.plot(self.aorta.t[ti], 0*self.aorta.t[ti], color='black')
        ax2.plot(self.aorta.t[ti], self.ck[ti], 'b-', label='kidney')
        ax2.plot(self.aorta.t[ti], self.cp[ti], 'r-', label='plasma')
        ax2.plot(self.aorta.t[ti], self.ct[ti], 'g-', label='tubuli')
        if legend:
            ax2.legend()
        if save:          
            plt.savefig(fname=os.path.join(path, name + ' fit ' + win_str + '.png'))
        if show:
            plt.show()
        else:
            plt.close()
