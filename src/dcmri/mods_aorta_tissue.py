from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import dcmri as dc


class AortaLiver(dc.Model):

    def __init__(self, **attr):

        # Constants
        self.dt = 0.5   
        self.tmax = 120  
        self.tacq = 2.0           
        self.dose_tolerance = 0.1   
        self.weight = 70.0          
        self.agent = 'gadoterate'   
        self.dose = 0.025
        self.rate = 1  
        self.field_strength = 3.0
        self.t0 = 0    
        self.TR = 0.005 
        self.FA = 15.0 
        self.TC = 0.180
        
        # Aorta parameters
        self.R10b = 1.0 
        self.S0b = 1
        self.BAT = 60
        self.CO = 100
        self.Thl = 10
        self.Dhl = 0.2
        self.To = 20
        self.Eb = 0.05
        self.Eo = 0.15
        self.Teb = 120

        # Liver parameters
        self.Hct = 0.45 
        self.R10l = 1 
        self.S0l = 1 
        self.Tel = 30.0
        self.De = 0.85
        self.ve = 0.3 
        self.khe = 20/6000
        self.Th = 30*60
        self.khe_f = 20/6000
        self.Th_f = 30*60
        self.vol = None
        self.signal = 'SS'
        self.organs = '2cxm'
        self.kinetics = 'stationary'
        self.free = ['BAT','CO','Thl','Dhl','To','Eb',
                'Tel','De','ve','khe','Th','Eo','Teb']
        self.bounds = [
            [0, 0, 0, 0.05, 0, 0.01,0.1, 0, 0.01, 0, 10*60, 0, 0],
            [np.inf, 300, 30, 0.95, 60, 0.15, 60, 1, 0.6, 0.1, 10*60*60, 0.5, 800],
        ]
        dc.init(self, **attr)
        self._predict = None

    def _conc_aorta(self) -> np.ndarray:
        if self.organs == 'comp':
            organs = ['comp', self.To]
        else:
            organs = ['2cxm', (self.To, self.Teb, self.Eo)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        Ji = dc.influx_step(self.t, self.weight, 
                conc, self.dose, self.rate, self.BAT)
        Jb = dc.flux_aorta(Ji, E=self.Eb,
            heartlung = ['pfcomp', (self.Thl, self.Dhl)],
            organs = organs,
            dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/self.CO
        self.ca = cb/(1-self.Hct)
        return self.t, cb
    
    def _relax_aorta(self) -> np.ndarray:
        t, cb = self._conc_aorta()
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        return t, self.R10b + rp*cb

    def _predict_aorta(self, xdata:np.ndarray) -> np.ndarray:
        self.tmax = max(xdata)+self.tacq+self.dt
        t, R1b = self._relax_aorta()
        if self.signal == 'SR':
            #signal = dc.signal_src(R1b, self.S0b, self.TC, R10=self.R10b)
            signal = dc.signal_src(R1b, self.S0b, self.TC)
        else:
            #signal = dc.signal_ss(R1b, self.S0b, self.TR, self.FA, R10=self.R10b)
            signal = dc.signal_ss(R1b, self.S0b, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.tacq)

    def _conc_liver(self, sum=True):
        if self.kinetics == 'non-stationary':
            khe = dc.interp([self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], self.t)
            Kbh = dc.interp([1/self.Th, 1/self.Th_f], self.t)
            hepatocytes = ['nscomp', 1/Kbh]
        elif self.kinetics == 'non-stationary uptake':
            self.Th_f = self.Th
            khe = dc.interp([self.khe*(1-self.Hct), self.khe_f*(1-self.Hct)], self.t)
            hepatocytes = ['comp', self.Th]
        else:
            self.khe_f = self.khe
            self.Th_f = self.Th
            khe = self.khe*(1-self.Hct)
            hepatocytes = ['comp', self.Th]
        return self.t, dc.conc_liver_hep(
                self.ca, self.ve, khe, dt=self.dt, sum=sum,
                extracellular = ['pfcomp', (self.Tel, self.De)],
                hepatocytes = hepatocytes)
    
    def _relax_liver(self):
        t, Cl = self._conc_liver(sum=False)
        rp = dc.relaxivity(self.field_strength, 'plasma', self.agent)
        rh = dc.relaxivity(self.field_strength, 'hepatocytes', self.agent)
        return t, self.R10l + rp*Cl[0,:] + rh*Cl[1,:]
    
    def _predict_liver(self, xdata:np.ndarray) -> np.ndarray:
        t, R1l = self._relax_liver()
        if self.signal == 'SR':
            #signal = dc.signal_sr(R1l, self.S0l, self.TR, self.FA, self.TC, R10=R1l[0])
            signal = dc.signal_sr(R1l, self.S0l, self.TR, self.FA, self.TC)
        else:
            #signal = dc.signal_ss(R1l, self.S0l, self.TR, self.FA, R10=R1l[0])
            signal = dc.signal_ss(R1l, self.S0l, self.TR, self.FA)
        return dc.sample(xdata, t, signal, self.tacq)
    
    def conc(self, sum=True):
        t, cb = self._conc_aorta()
        t, C = self._conc_liver(sum=sum)
        return t, cb, C
    
    def relax(self):
        t, R1b = self._relax_aorta()
        t, R1l = self._relax_liver()
        return t, R1b, R1l

    def predict(self, xdata:tuple[np.ndarray, np.ndarray])->tuple[np.ndarray, np.ndarray]:
        # Public interface
        if self._predict is None:
            signala = self._predict_aorta(xdata[0])
            signall = self._predict_liver(xdata[1])
            return signala, signall
        # Private interface with different input & output types
        elif self._predict == 'aorta':
            return self._predict_aorta(xdata)
        elif self._predict == 'liver': 
            return self._predict_liver(xdata)


    def train(self, xdata:tuple[np.ndarray, np.ndarray], 
              ydata:tuple[np.ndarray, np.ndarray], **kwargs):

        # Estimate BAT and S0b from data
        if self.signal == 'SR':
            Srefb = dc.signal_sr(self.R10b, 1, self.TR, self.FA, self.TC)
            Srefl = dc.signal_sr(self.R10l, 1, self.TR, self.FA, self.TC)
        else:
            Srefb = dc.signal_ss(self.R10b, 1, self.TR, self.FA)
            Srefl = dc.signal_ss(self.R10l, 1, self.TR, self.FA)
        n0 = max([np.sum(xdata[0]<self.t0), 1])
        self.S0b = np.mean(ydata[0][:n0]) / Srefb
        self.S0l = np.mean(ydata[1][:n0]) / Srefl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-self.Dhl)*self.Thl
        self.BAT = max([self.BAT, 0])

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        f, b = deepcopy(self.free), deepcopy(self.bounds)
        sel = ['BAT','CO','Thl','Dhl','To','Eb','Eo','Teb']
        isel = [i for i in range(len(f)) if f[i] in sel]
        self.free = [f[i] for i in isel]
        self.bounds = [[b[0][i] for i in isel], [b[1][i] for i in isel]]
        dc.train(self, xdata[0], ydata[0], **kwargs)

        # Train free liver parameters on liver data
        self._predict = 'liver'
        sel = ['Tel','De','ve','khe','Th','khe_f','Th_f']
        isel = [i for i in range(len(f)) if f[i] in sel]
        self.free = [f[i] for i in isel]
        self.bounds = [[b[0][i] for i in isel], [b[1][i] for i in isel]]
        dc.train(self, xdata[1], ydata[1], **kwargs)

        # Train all parameters on all data
        self._predict = None
        self.free = f
        self.bounds = b
        return dc.train(self, xdata, ydata, **kwargs)
    
    def plot(self, 
                xdata:tuple[np.ndarray, np.ndarray], 
                ydata:tuple[np.ndarray, np.ndarray],  
                xlim=None, testdata=None, 
                fname=None, show=True):
        t, cb, C = self.conc(sum=False)
        sig = self.predict((t,t))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data1scan(t, sig[0], xdata[0], ydata[0], 
                ax1, xlim, 
                color=['lightcoral','darkred'], 
                test=None if testdata is None else testdata[0])
        _plot_data1scan(t, sig[1], xdata[1], ydata[1], 
                ax3, xlim, 
                color=['cornflowerblue','darkblue'], 
                test=None if testdata is None else testdata[1])
        _plot_conc_aorta(t, cb, ax2, xlim)
        _plot_conc_liver(t, C, ax4, xlim)
        if fname is None:
            plt.show()
        else:   
            plt.savefig(fname=fname)
            if show:
                plt.show()
            else:
                plt.close()

    def pars(self):
        pars = {}
        # Aorta
        pars['T10b']=['Blood precontrast T1', 1/self.R10b, "sec"]
        pars['BAT']=['Bolus arrival time', self.BAT, "sec"] 
        pars['CO']=['Cardiac output', self.CO, "mL/sec"] 
        pars['Thl']=['Heart-lung mean transit time', self.Thl, "sec"]
        pars['Dhl']=['Heart-lung transit time dispersion', self.Dhl, ""]
        pars['To']=["Organs mean transit time", self.To, "sec"]
        pars['Eb']=["Extraction fraction", self.Eb, ""]
        pars['Tc']=["Mean circulation time", self.Thl+self.To, 'sec'] 
        pars['Eo']=["Organs extraction fraction", self.Eo, ""]
        pars['Teb']=["Organs extracellular mean transit time", self.Teb, "sec"]
        # Liver
        pars['T10l']=['Liver precontrast T1', 1/self.R10l, "sec"]
        pars['Tel']=["Liver extracellular mean transit time", self.Tel, 'sec']
        pars['De']=["Liver extracellular dispersion", self.De, '']
        pars['ve']=["Liver extracellular volume fraction", self.ve, 'mL/mL']
        if self.kinetics=='stationary':
            pars['khe']=["Hepatocellular uptake rate", self.khe, 'mL/sec/mL']
            pars['Th']=["Hepatocellular transit time", self.Th, 'sec']
            pars['kbh']=["Biliary excretion rate", (1-self.ve)/self.Th, 'mL/sec/mL']
            pars['Khe']=["Hepatocellular tissue uptake rate", self.khe/self.ve, 'mL/sec/mL']
            pars['Kbh']=["Biliary tissue excretion rate", 1/self.Th, 'mL/sec/mL']
            if self.vol is not None:
                pars['CL']=['Liver blood clearance', self.khe*self.vol, 'mL/sec']
        else:
            khe = [self.khe, self.khe_f]
            Kbh = [1/self.Th, 1/self.Th_f]
            khe_avr = np.mean(khe)
            Kbh_avr = np.mean(Kbh)
            khe_var = (np.amax(khe)-np.amin(khe))/khe_avr
            Kbh_var = (np.amax(Kbh)-np.amin(Kbh))/Kbh_avr 
            kbh = np.mean((1-self.ve)*Kbh_avr)
            Th = np.mean(1/Kbh_avr)
            pars['khe']=["Hepatocellular uptake rate", khe_avr, 'mL/sec/mL']
            pars['Th']=["Hepatocellular transit time", Th, 'sec']
            pars['kbh']=["Biliary excretion rate", kbh, 'mL/sec/mL']
            pars['Khe']=["Hepatocellular tissue uptake rate", khe_avr/self.ve, 'mL/sec/mL']
            pars['Kbh']=["Biliary tissue excretion rate", Kbh_avr, 'mL/sec/mL']
            pars['khe_i']=["Hepatocellular uptake rate (initial)", self.khe, 'mL/sec/mL']
            pars['khe_f']=["Hepatocellular uptake rate (final)", self.khe_f, 'mL/sec/mL']
            pars['Th_i']=["Hepatocellular transit time (initial)", self.Th, 'sec']
            pars['Th_f']=["Hepatocellular transit time (final)", self.Th_f, 'sec']
            pars['khe_var']=["Hepatocellular uptake rate variance", khe_var, '']
            pars['Kbh_var']=["Biliary tissue excretion rate variance", Kbh_var, '']
            pars['kbh_i']=["Biliary excretion rate (initial)", (1-self.ve)/self.Th, 'mL/sec/mL']
            pars['kbh_f']=["Biliary excretion rate (final)", (1-self.ve)/self.Th_f, 'mL/sec/mL']
            if self.vol is not None:
                pars['CL']=['Liver blood clearance', khe_avr*self.vol, 'mL/sec']
        return self.add_sdev(pars)
    

class AortaLiver2scan(AortaLiver):

    def __init__(self, **attr):
        super().__init__()
        self.tacq2 = 2.0
        self.dose = [dc.ca_std_dose('gadoterate')/2, dc.ca_std_dose('gadoterate')/2] 
        self.S02b = 1
        self.S02l = 1
        self.BAT2 = 1200 
        self.R102b = 1 
        self.R102l = 1 
        self.kinetics = 'non-stationary'
        self.free += ['khe_f','Th_f','BAT2','S02b','S02l']
        self.bounds[0] += [0, 10*60, 0, 0, 0]
        self.bounds[1] += [0.1, 10*60*60, np.inf, np.inf, np.inf]
        dc.init(self, **attr)

    def _conc_aorta(self) -> tuple[np.ndarray, np.ndarray]:
        if self.organs == 'comp':
            organs = ['comp', self.To]
        else:
            organs = ['2cxm', (self.To, self.Teb, self.Eo)]
        self.t = np.arange(0, self.tmax, self.dt)
        conc = dc.ca_conc(self.agent)
        J1 = dc.influx_step(self.t, self.weight, conc, self.dose[0], self.rate, self.BAT)
        J2 = dc.influx_step(self.t, self.weight, conc, self.dose[1], self.rate, self.BAT2)
        Jb = dc.flux_aorta(J1 + J2, E=self.Eb,
            heartlung = ['pfcomp', (self.Thl, self.Dhl)],
            organs = organs,
            dt=self.dt, tol=self.dose_tolerance)
        cb = Jb/self.CO
        self.ca = cb/(1-self.Hct)
        return self.t, cb

    def _predict_aorta(self, 
            xdata:tuple[np.ndarray, np.ndarray],
            )->tuple[np.ndarray, np.ndarray]:
        self.tmax = max(xdata[1])+self.tacq2+self.dt
        t, R1 = self._relax_aorta()
        t1 = t<=xdata[0][-1]
        t2 = t>=xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        if self.signal == 'SR':
            signal1 = dc.signal_sr(R11, self.S0b, self.TR, self.FA, self.TC)
            signal2 = dc.signal_sr(R12, self.S02b, self.TR, self.FA, self.TC)
        else:
            signal1 = dc.signal_ss(R11, self.S0b, self.TR, self.FA)
            signal2 = dc.signal_ss(R12, self.S02b, self.TR, self.FA)
        return (
            dc.sample(xdata[0], t[t1], signal1, self.tacq),
            dc.sample(xdata[1], t[t2], signal2, self.tacq2),
        )
    
    def _predict_liver(self, 
            xdata:tuple[np.ndarray, np.ndarray],
            )->tuple[np.ndarray, np.ndarray]:
        t, R1 = self._relax_liver()
        t1 = t<=xdata[0][-1]
        t2 = t>=xdata[1][0]
        R11 = R1[t1]
        R12 = R1[t2]
        if self.signal == 'SR':
            signal1 = dc.signal_sr(R11, self.S0l, self.TR, self.FA, self.TC)
            signal2 = dc.signal_sr(R12, self.S02l, self.TR, self.FA, self.TC)
        else:
            signal1 = dc.signal_ss(R11, self.S0l, self.TR, self.FA)
            signal2 = dc.signal_ss(R12, self.S02l, self.TR, self.FA)
        return (
            dc.sample(xdata[0], t[t1], signal1, self.tacq),
            dc.sample(xdata[1], t[t2], signal2, self.tacq2),
        )
 
    def predict(self, 
            xdata:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            )->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Public interface
        if self._predict is None:
            signal_a = self._predict_aorta((xdata[0],xdata[1]))
            signal_l = self._predict_liver((xdata[2],xdata[3]))
            return signal_a + signal_l
        # Private interface with different in- and outputs
        elif self._predict == 'aorta':
            return self._predict_aorta(xdata)
        elif self._predict == 'liver': 
            return self._predict_liver(xdata)


    def train(self, 
              xdata:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
              ydata:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              **kwargs):
        # x,y: (aorta scan 1, aorta scan 2, liver scan 1, liver scan 2)

        # Estimate BAT
        T, D = self.Thl, self.Dhl
        self.BAT = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self.BAT2 = xdata[1][np.argmax(ydata[1])] - (1-D)*T

        # Estimate S0
        if self.signal == 'SR':
            Srefb = dc.signal_sr(self.R10b, 1, self.TR, self.FA, self.TC)
            Sref2b = dc.signal_sr(self.R102b, 1, self.TR, self.FA, self.TC)
            Srefl = dc.signal_sr(self.R10l, 1, self.TR, self.FA, self.TC)
            Sref2l = dc.signal_sr(self.R102l, 1, self.TR, self.FA, self.TC)
        else:
            Srefb = dc.signal_ss(self.R10b, 1, self.TR, self.FA)
            Sref2b = dc.signal_ss(self.R102b, 1, self.TR, self.FA)
            Srefl = dc.signal_ss(self.R10l, 1, self.TR, self.FA)
            Sref2l = dc.signal_ss(self.R102l, 1, self.TR, self.FA)

        n0 = max([np.sum(xdata[0]<self.t0), 2])
        self.S0b = np.mean(ydata[0][1:n0]) / Srefb
        self.S02b = np.mean(ydata[1][1:n0]) / Sref2b
        self.S0l = np.mean(ydata[2][1:n0]) / Srefl
        self.S02l = np.mean(ydata[3][1:n0]) / Sref2l

        f, b = deepcopy(self.free), deepcopy(self.bounds)

        # Train free aorta parameters on aorta data
        self._predict = 'aorta'
        sel = ['BAT','CO','Thl','Dhl','To','Eb','Eo','Teb','BAT2','S02b']
        isel = [i for i in range(len(f)) if f[i] in sel]
        self.free = [f[i] for i in isel]
        self.bounds = [[b[0][i] for i in isel], [b[1][i] for i in isel]]
        dc.train(self, (xdata[0],xdata[1]), (ydata[0],ydata[1]), **kwargs)       

        # Train free liver parameters on liver data
        self._predict = 'liver'
        sel = ['Tel','De','ve','khe','Th','khe_f','Th_f','S02l']
        isel = [i for i in range(len(f)) if f[i] in sel]
        self.free = [f[i] for i in isel]
        self.bounds = [[b[0][i] for i in isel], [b[1][i] for i in isel]]
        dc.train(self, (xdata[2],xdata[3]), (ydata[2],ydata[3]), **kwargs) 

        # Train all parameters on all data
        self._predict = None
        self.free = f
        self.bounds = b
        return dc.train(self, xdata, ydata, **kwargs)
    
    def plot(self,
             xdata:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
             ydata:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             testdata=None, xlim=None, fname=None, show=True):
        t, cb, C = self.conc(sum=False)
        ta1 = t[t<=xdata[1][0]]
        ta2 = t[(t>xdata[1][0]) & (t<=xdata[1][-1])]
        tl1 = t[t<=xdata[3][0]]
        tl2 = t[(t>xdata[3][0]) & (t<=xdata[3][-1])]
        sig = self.predict((ta1,ta2,tl1,tl2))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data2scan((ta1,ta2), sig[:2], xdata[:2], ydata[:2], 
                ax1, xlim, 
                color=['lightcoral','darkred'], 
                test=None if testdata is None else testdata[0])
        _plot_data2scan((tl1,tl2), sig[2:], xdata[2:], ydata[2:], 
                ax3, xlim, 
                color=['cornflowerblue','darkblue'], 
                test=None if testdata is None else testdata[1])
        _plot_conc_aorta(t, cb, ax2, xlim)
        _plot_conc_liver(t, C, ax4, xlim)
        if fname is None:
            plt.show()
        else:   
            plt.savefig(fname=fname)
            if show:
                plt.show()
            else:
                plt.close()
    

# Helper functions for plotting

def _plot_conc_aorta(t:np.ndarray, cb:np.ndarray, ax, xlim=None):
    if xlim is None:
        xlim = [t[0],t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-', color='darkred', linewidth=3.0, label='Aorta')
    ax.legend()

def _plot_conc_liver(t:np.ndarray, C:np.ndarray, ax, xlim=None):
    color = 'darkblue'
    if xlim is None:
        xlim = [t[0],t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*C[0,:], linestyle='-.', color=color, linewidth=2.0, label='Extracellular')
    ax.plot(t/60, 1000*C[1,:], linestyle='--', color=color, linewidth=2.0, label='Hepatocytes')
    ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', color=color, linewidth=3.0, label='Tissue')
    ax.legend()

def _plot_data2scan(t:tuple[np.ndarray, np.ndarray], sig:tuple[np.ndarray, np.ndarray], 
        xdata:tuple[np.ndarray, np.ndarray], ydata:tuple[np.ndarray, np.ndarray], 
        ax, xlim, color=['black', 'black'], test=None):
    if xlim is None:
        xlim = [0,t[1][-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
    ax.plot(np.concatenate(xdata)/60, np.concatenate(ydata), 
            marker='o', color=color[0], label='fitted data', linestyle = 'None')
    ax.plot(np.concatenate(t)/60, np.concatenate(sig), 
            linestyle='-', color=color[1], linewidth=3.0, label='fit' )
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black', marker='D', linestyle='None', label='Test data')
    ax.legend()

def _plot_data1scan(t:np.ndarray, sig:np.ndarray,
        xdata:np.ndarray, ydata:np.ndarray, 
        ax, xlim, color=['black', 'black'], 
        test=None):
    if xlim is None:
        xlim = [t[0],t[-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
    ax.plot(xdata/60, ydata, marker='o', color=color[0], label='fitted data', linestyle = 'None')
    ax.plot(t/60, sig, linestyle='-', color=color[1], linewidth=3.0, label='fit' )
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black', marker='D', linestyle='None', label='Test data')
    ax.legend()


