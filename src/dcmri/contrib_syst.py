import os
import time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import contrib_rt as rt
import contrib_sig as sig
import contrib_aux as aux

#jmax = 6mmol/L * 0.6L/sec = 3.6 mmol/sec
class Conv1D1C():

    def __init__(self, 
            dim = [5, 25],       # system dimensions (sec, cm)
            mat = [60, 20],      # sampling matrix (time, space)  
            Jn = [0,0,0,0,0],    # low-res influx right (mmol/sec)
            Jp = [0,1,0,0,0],    # low-res influx left (mmol/sec)
            u = [10,10,10],      # low-res velocities (cm/sec)  
            nx = None,           # high-res nr of voxels
            umax = None,         # maximal velocity (cm/sec) 
            jmax = None,         # maximum influx (cm/sec)
            ):

        # Defaults
        if nx is None: # default = no upsampling
            nx = mat[1]
        if umax is None:
            umax = 2*np.amax(u)
        if jmax is None:
            j = np.concatenate([Jp,Jn])
            jmax = 2*np.amax(j)

        if nx < mat[1]:
            msg = 'Numerical voxel size must be smaller than measured voxel size.'
            raise ValueError(msg)

        # Variables
        self.u = np.array(u)     # low-res velocities (cm/sec)
        self.Jp = np.array(Jp)  # low-res influx left (mmol/sec)
        self.Jn = np.array(Jn)  # low-res influx right (mmol/sec)

        # Constants
        self.dim = np.array(dim)      # (acquisition duration, length)
        self.mat = np.array(mat)     # measured matrix (time, space)
        self.nx = nx       # number of voxels at high-res   
        self.umax = umax   # maximum velocity   
        self.jmax = jmax    # maximum influx

        # Generate concentrations
        self.precompute()
        self.calc_conc()


    def precompute(self):
        # Locations of low and high-res velocities  
        self.xl = np.linspace(0, self.dim[1], len(self.u))  # locations of low-res velocities  
        self.xh = np.linspace(0, self.dim[1], self.nx+1)  # locations of high-res velocities
    
        # Locations of low and high-res influx
        dth = self.xh[1]/self.umax                    # time interval between high-res fluxes
        nth = np.ceil(1 + self.dim[0]/dth).astype(np.int32)
        self.tl = np.linspace(0, self.dim[0], len(self.Jp))
        self.th = np.linspace(0, self.dim[0], nth)           # time points of high-res fluxes    

        # Locations of sample points
        self.tx = sig.sample_loc_1d((nth,self.nx), self.mat)

        # Parameter bounds (normalized parameters)
        u1 = np.ones(len(self.u))
        J0 = np.zeros(2*len(self.Jp))
        J1 = np.ones(2*len(self.Jp))
        upper = np.concatenate([u1, J1])
        lower = np.concatenate([-u1, J0])
        self.bounds = (lower, upper)


    def calc_conc(self):
        # Upsample parameter fields
        u = np.interp(self.xh, self.xl, self.u)          # high-res velocities  
        Jp = np.interp(self.th, self.tl, self.Jp)       # high-res influx left
        Jn = np.interp(self.th, self.tl, self.Jn)       # high-res influx right 
        # Calculate concentrations
        Kp, Kn = rt.K_flow_1d(self.xh[1], u)           # high-res rate constants
        C = rt.conc_1d1c(self.th, Jp, Jn, Kp, Kn)      # high-res concentrations
        # Downsample concentrations
        self.C = sig.sample_1d(C, self.mat, loc=self.tx)

    def _set_pars(self, p):
        nu = len(self.u)
        nj = len(self.Jp)
        self.u = p[:nu]*self.umax
        self.Jp = p[nu:nu+nj]*self.jmax
        self.Jn = p[nu+nj:]*self.jmax

    def _fit_func(self, _, *p):
        p = np.array(p)
        self._set_pars(p)
        self.calc_conc()
        return self.C.ravel()

    def fit_to(self, Cmeas, umax=None, jmax=None, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        if umax is not None:
            self.umax=umax
            self.precompute()
        if jmax is not None:
            self.jmax=jmax
        p0 = np.concatenate([self.u/self.umax, self.Jp/self.jmax, self.Jn/self.jmax])
        p, pcov = curve_fit(self._fit_func, None, Cmeas.ravel(), p0=p0, bounds=self.bounds, xtol=xtol, ftol=ftol, gtol=gtol) 
        self._set_pars(p) 
        self.calc_conc() 

    def plot_conc(self, time=False, data=None):
        # Display measured concentrations
        Dx = self.dim[1]/self.mat[1]                     # voxel size
        Dt = self.dim[0]/self.mat[0]                     # sampling interval
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx)      # voxel centers
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt)       # sample interval centers
        if time:
            rt.plot_Ct_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)
        else:
            rt.plot_Cx_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)

    def plot_pars(self, truth=None):
        if truth is None:
            rt.plot_flow_1d1c_pars(self.tl, self.xl, self.Jp, self.Jn, self.u)
        else:
            rt.plot_flow_1d1c_pars(self.tl, self.xl, self.Jp, self.Jn, self.u, truth.tl, truth.xl, truth.Jp, truth.Jn, truth.u)



class Perf1D():

    def __init__(self, 
            dim = [5, 25],       # system dimensions (sec, cm)
            mat = [60, 20],      # sampling matrix (time, space)  
            Jna = [1,1],    # low-res influx right (mmol/sec/cm^2)
            Jpa = [1,1],    # low-res influx left (mmol/sec/cm^2)
            ua = [10,-10],      # low-res velocities (cm/sec)  
            uv = [-5,+5],      # low-res velocities (cm/sec) 
            Kva = [0,0],     # perfusion
            nx = None,           # high-res nr of voxels
            umax = None,         # maximal velocity (cm/sec) 
            Jmax = None,
            Kmax = None,
            ):
        # Defaults
        if nx is None: 
            nx = mat[1]
        if umax is None:
            umax = 2*np.amax(ua)
        if Jmax is None:
            Jmax = 2*np.amax(np.concatenate([Jpa, Jna]))
        if Kmax is None:
            Kmax = 2*np.amax(Kva)
        if nx < mat[1]:
            msg = 'Numerical voxel size must be smaller than measured voxel size.'
            raise ValueError(msg)
        # Variables
        self.ua = np.array(ua) 
        self.uv = np.array(uv)
        self.Kva = np.array(Kva)
        self.Jpa = np.array(Jpa) 
        self.Jna = np.array(Jna) 
        # Constants
        self.dim = np.array(dim)  
        self.mat = np.array(mat) 
        self.nx = nx    
        self.umax = umax  
        self.Jmax = Jmax  
        self.Kmax = Kmax
        # Concentrations
        self.precompute()
        self.calc_conc()

    def plot_pars(self, truth=None, file=None):
        xl = np.linspace(0, self.dim[1], len(self.ua))
        tl = np.linspace(0, self.dim[0], len(self.Jpa))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(8.27,11.69))
        fig.suptitle('1D 2C flow parameters', fontsize=12)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8, wspace=0.2)
        # Plot left influx
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Left arterial influx (mmol/sec/cm^2)')
        ax1.plot(tl, self.Jpa, 'r--')
        ax1.set_xlim(tl[0], tl[-1])
        # Plot right influx
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Right arterial influx (mmol/sec/cm^2)')
        ax2.plot(tl, self.Jna, 'r--')  
        ax2.set_xlim(tl[0], tl[-1])
        # Plot velocity
        ax3.set_xlabel('Position (cm)')
        ax3.set_ylabel('Velocity (cm/sec)')
        ax3.plot(xl, self.ua, 'r--', label='Arterial')
        ax3.plot(xl, self.uv, 'b--', label='Venous')
        ax3.set_xlim(xl[0], xl[-1])
        # Plot perfusion
        ax4.set_xlabel('Position (cm)')
        ax4.set_ylabel('Transfer constant (1/sec)')
        ax4.plot(xl, self.Kva, 'm--')
        ax4.set_xlim(xl[0], xl[-1])
        # Plot ground truths
        if truth is not None:
            xl = np.linspace(0, truth.dim[1], len(truth.ua))
            tl = np.linspace(0, truth.dim[0], len(truth.Jpa))
            ax1.plot(tl, truth.Jpa, 'r-')
            ax2.plot(tl, truth.Jna, 'r-')
            ax3.plot(xl, truth.ua, 'r-')
            ax3.plot(xl, truth.uv, 'b-')
            ax4.plot(xl, truth.Kva, 'm-')
        ax3.legend()
        if file is None:   
            plt.show()
        else:
            path = os.path.dirname(file)
            if not os.path.exists(path):
                os.makedirs(path)       
            plt.savefig(fname=file)
        plt.close()

    def precompute(self):
        self.xl = np.linspace(0, self.dim[1], len(self.ua))
        self.tl = np.linspace(0, self.dim[0], len(self.Jpa))
        # Locations of low and high-res velocities  
        self.xh = np.linspace(0, self.dim[1], self.nx+1)  
        dxh = self.xh[1]
        self.xv = np.linspace(dxh/2, self.dim[1]-dxh/2, self.nx)  
        # Locations of low and high-res influx
        # dth = dxh/self.umax      
        dth = rt.dt_1d2cf(dxh, self.umax, self.Kmax)    
        nth = np.ceil(1 + self.dim[0]/dth).astype(np.int32)
        self.th = np.linspace(0, self.dim[0], nth)           
        # Locations of sample points
        self.tx = sig.sample_loc_1d((nth,self.nx), self.mat)

    def calc_conc(self, split=False):
        dx = self.xh[1]
        # Upsample parameter fields
        ua = np.interp(self.xh, self.xl, self.ua)  
        uv = np.interp(self.xh, self.xl, self.uv)
        Kva = np.interp(self.xv, self.xl, self.Kva)
        Jpa = np.interp(self.th, self.tl, self.Jpa)
        Jna = np.interp(self.th, self.tl, self.Jna)  
        # Calculate concentrations
        Kpa, Kna = rt.K_flow_1d(dx, ua) 
        Kpv, Knv = rt.K_flow_1d(dx, uv) 
        jpa, jna = Jpa/dx, Jna/dx
        Ca, Cv = rt.conc_1d2cf(self.th, jpa, jna, Kpa, Kna, Kpv, Knv, Kva)
        # Downsample concentrations
        if split:
            self.Ca = sig.sample_1d(Ca, self.mat, loc=self.tx)
            self.Cv = sig.sample_1d(Cv, self.mat, loc=self.tx)
            self.C = self.Ca + self.Cv
        else:
            self.C = sig.sample_1d(Ca+Cv, self.mat, loc=self.tx)

    def plot_conc(self, time=False, data=None):
        Dx = self.dim[1]/self.mat[1]                    
        Dt = self.dim[0]/self.mat[0]         
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx) 
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt) 
        if time:
            rt.plot_Ct_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)
        else:
            rt.plot_Cx_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)

    def plot_split_conc(self, time=False):
        Dx = self.dim[1]/self.mat[1]  
        Dt = self.dim[0]/self.mat[0]  
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx) 
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt) 
        if time:
            rt.plot_Ct_1d2c(tv, xv, self.Ca, self.Cv, rows=6, cols=6)
        else:
            rt.plot_Cx_1d2c(tv, xv, self.Ca, self.Cv, rows=6, cols=6)

    def fitpars(self):
        p = [
            self.ua/self.umax, 
            self.uv/self.umax,
            self.Jpa/self.Jmax, 
            self.Jna/self.Jmax,
            self.Kva/self.Kmax,
            #self.Kva[1:-1]/self.Kmax,
        ]
        return np.concatenate(p)

    def set_fitpars(self, p):
        nu = len(self.ua)
        nj = len(self.Jpa)
        self.ua = p[:nu]*self.umax
        self.uv = p[nu:2*nu]*self.umax
        self.Jpa = p[2*nu:2*nu+nj]*self.Jmax
        self.Jna = p[2*nu+nj:2*nu+2*nj]*self.Jmax
        self.Kva = p[2*nu+2*nj:]*self.Kmax
        #self.Kva[1:-1] = p[2*nu+2*nj:]*self.Kmax

    def fitbounds(self):
        # Parameter bounds (normalized parameters)
        u1 = np.ones(2*len(self.ua))
        u0 = -u1
        J1 = np.ones(2*len(self.Jpa))
        J0 = np.zeros(2*len(self.Jpa))
        # K1 = np.ones(len(self.Kva)-2)
        # K0 = np.zeros(len(self.Kva)-2)
        K1 = np.ones(len(self.Kva))
        K0 = np.zeros(len(self.Kva))
        upper = np.concatenate([u1, J1, K1])
        lower = np.concatenate([u0, J0, K0])
        return (lower, upper)

    def fit_func(self, _, *p):
        p = np.array(p)
        self.set_fitpars(p)
        self.calc_conc()
        return self.C.ravel()
    
    def fit_to(self, Cmeas, umax=None, Jmax=None, Kmax=None, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        if umax is not None:
            self.umax=umax
        if Jmax is not None:
            self.Jmax=Jmax
        if Kmax is not None:
            self.Kmax=Kmax
        # get parameters and initial values
        self.precompute()
        p0 = self.fitpars()
        bounds = self.fitbounds()
        # Fit parameters
        p, pcov = curve_fit(self.fit_func, None, Cmeas.ravel(), p0=p0, bounds=bounds, xtol=xtol, ftol=ftol, gtol=gtol) 
        pcorr = np.linalg.norm(p-p0)/np.linalg.norm(p0)
        # Create predictions
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov, pcorr
    
    def resample(self, scl):
        nt = np.ceil(scl[0]*len(self.tl)).astype(np.uint32)
        nx = np.ceil(scl[1]*len(self.xl)).astype(np.uint32)
        nt = np.amax([nt,2])
        nx = np.amax([nx,2])
        tl = np.linspace(0, self.dim[0], nt)
        xl = np.linspace(0, self.dim[1], nx)
        self.ua = np.interp(xl, self.xl, self.ua)          
        self.uv = np.interp(xl, self.xl, self.uv)
        self.Kva = np.interp(xl, self.xl, self.Kva)
        self.Jpa = np.interp(tl, self.tl, self.Jpa)       
        self.Jna = np.interp(tl, self.tl, self.Jna)   
        self.precompute()

    def mres_fit_to(self, Cmeas, umax=None, Jmax=None, Kmax=None, xtol=1e-3, ftol=1e-3, gtol=1e-3, mxtol=1e-2, export_path=None, filename='mres'):
        it, start, start_mres = 0, time.time(), time.time()
        print('mres level: ', it)
        p, pcov, pcorr = self.fit_fast_to(Cmeas, umax, Jmax, Kmax, xtol, ftol, gtol)
        print('>> calculation time (mins): ', (time.time()-start)/60)
        print('>> parameter correction (%): ', 100*pcorr)
        if export_path is not None:
            file = os.path.join(export_path, filename+'_'+str(it)+'.png')
            self.plot_pars(file=file)
        while pcorr > mxtol:
            self.resample((2,2))
            it, start = it+1, time.time()
            print('mres level: ', it)
            p, pcov, pcorr = self.fit_fast_to(Cmeas, umax, Jmax, Kmax, xtol, ftol, gtol)
            print('>> calculation time (mins): ', (time.time()-start)/60)
            print('>> parameter correction (%): ', 100*pcorr)
            if export_path is not None:
                file = os.path.join(export_path, filename+'_'+str(it)+'.png')
                self.plot_pars(file=file)
        print('Multi-resolution calculation time (mins): ', (time.time()-start_mres)/60)
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov
    
    def _fit_fast_to(self, Cmeas, umax=None, Jmax=None, Kmax=None, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        #fast but does not seem to work very well. Keep to original resolution
        if umax is not None:
            self.umax=umax
        if Jmax is not None:
            self.Jmax=Jmax
        if Kmax is not None:
            self.Kmax=Kmax
        # Downsample measurement concentrations to parameter resolution
        scale = 2
        self.mat = [
            np.amin([scale*len(self.Jpa), Cmeas.shape[0]]), 
            np.amin([scale*len(self.ua), Cmeas.shape[1]]), 
        ]
        Cmeas_lowres = sig.sample_1d(Cmeas, self.mat)
        nx_orig = self.nx
        self.nx = self.mat[1]
        # Perform the fit
        self.precompute()
        p0 = self.fitpars()
        bounds = self.fitbounds()
        p, pcov = curve_fit(self.fit_func, None, Cmeas_lowres.ravel(), p0=p0, bounds=bounds, xtol=xtol, ftol=ftol, gtol=gtol) 
        pcorr = np.linalg.norm(p-p0)/np.linalg.norm(p0)
        # Return to origonal measurement resolution and create predictions
        self.nx = nx_orig
        self.mat = Cmeas.shape
        self.precompute()
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov, pcorr

    def _calc_flow(self):
        pass
    # Kva Ca = Kva va ca = F ca
        # fa = ua * va
        # fv = uv * vv
        # fa + fv = f (const)
        # F = Kva * va
        # 0 = ua * va + uv * vv
        # -fa' = Kva * va = -ua' * va - ua * va'
        # 0 = (Kva+ua')va + ua va'


# variable dth - does not seem to work as well but needs more testing
class _Perf1D():

    def __init__(self, 
            dim = [5, 25],       # system dimensions (sec, cm)
            mat = [60, 20],      # sampling matrix (time, space)  
            Jna = [1,1],    # low-res influx right (mmol/sec/cm^2)
            Jpa = [1,1],    # low-res influx left (mmol/sec/cm^2)
            ua = [10,-10],      # low-res velocities (cm/sec)  
            uv = [-5,+5],      # low-res velocities (cm/sec) 
            Kva = [0,0],     # perfusion
            nx = None,           # high-res nr of voxels
            umax = None,         # maximal velocity (cm/sec) 
            Jmax = None,
            Kmax = None,
            ):
        # Defaults
        if nx is None: 
            nx = mat[1]
        if umax is None:
            umax = 2*np.amax(ua)
        if Jmax is None:
            Jmax = 2*np.amax(np.concatenate([Jpa, Jna]))
        if Kmax is None:
            Kmax = 2*np.amax(Kva)
        if nx < mat[1]:
            msg = 'Numerical voxel size must be smaller than measured voxel size.'
            raise ValueError(msg)
        # Variables
        self.ua = np.array(ua) 
        self.uv = np.array(uv)
        self.Kva = np.array(Kva)
        self.Jpa = np.array(Jpa) 
        self.Jna = np.array(Jna) 
        # Constants
        self.dim = np.array(dim)  
        self.mat = np.array(mat) 
        self.nx = nx    
        self.umax = umax  
        self.Jmax = Jmax  
        self.Kmax = Kmax
        # Concentrations
        self.precompute()
        self.calc_conc()

    def plot_pars(self, truth=None, file=None):
        xl = np.linspace(0, self.dim[1], len(self.ua))
        tl = np.linspace(0, self.dim[0], len(self.Jpa))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(8.27,11.69))
        fig.suptitle('1D 2C flow parameters', fontsize=12)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8, wspace=0.2)
        # Plot left influx
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Left arterial influx (mmol/sec/cm^2)')
        ax1.plot(tl, self.Jpa, 'r--')
        ax1.set_xlim(tl[0], tl[-1])
        # Plot right influx
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Right arterial influx (mmol/sec/cm^2)')
        ax2.plot(tl, self.Jna, 'r--')  
        ax2.set_xlim(tl[0], tl[-1])
        # Plot velocity
        ax3.set_xlabel('Position (cm)')
        ax3.set_ylabel('Velocity (cm/sec)')
        ax3.plot(xl, self.ua, 'r--', label='Arterial')
        ax3.plot(xl, self.uv, 'b--', label='Venous')
        ax3.set_xlim(xl[0], xl[-1])
        # Plot perfusion
        ax4.set_xlabel('Position (cm)')
        ax4.set_ylabel('Transfer constant (1/sec)')
        ax4.plot(xl, self.Kva, 'm--')
        ax4.set_xlim(xl[0], xl[-1])
        # Plot ground truths
        if truth is not None:
            xl = np.linspace(0, truth.dim[1], len(truth.ua))
            tl = np.linspace(0, truth.dim[0], len(truth.Jpa))
            ax1.plot(tl, truth.Jpa, 'r-')
            ax2.plot(tl, truth.Jna, 'r-')
            ax3.plot(xl, truth.ua, 'r-')
            ax3.plot(xl, truth.uv, 'b-')
            ax4.plot(xl, truth.Kva, 'm-')
        ax3.legend()
        if file is None:   
            plt.show()
        else:
            path = os.path.dirname(file)
            if not os.path.exists(path):
                os.makedirs(path)       
            plt.savefig(fname=file)
        plt.close()


    def precompute(self):
        self.xl = np.linspace(0, self.dim[1], len(self.ua))
        self.tl = np.linspace(0, self.dim[0], len(self.Jpa))
        # Locations of low and high-res velocities  
        self.xh = np.linspace(0, self.dim[1], self.nx+1)  
        dxh = self.xh[1]
        self.xv = np.linspace(dxh/2, self.dim[1]-dxh/2, self.nx)     


    def calc_conc(self, split=False):
        # Locations of high-res influx
        dxh = self.xh[1]
        dth = rt.dt_1d2cf_pix(dxh, self.ua, self.uv, self.Kva)
        nth = np.ceil(1 + self.dim[0]/dth).astype(np.int32)
        self.th = np.linspace(0, self.dim[0], nth)           
        self.tx = sig.sample_loc_1d((nth,self.nx), self.mat)
        # Upsample parameter fields
        ua = np.interp(self.xh, self.xl, self.ua)  
        uv = np.interp(self.xh, self.xl, self.uv)
        Kva = np.interp(self.xv, self.xl, self.Kva)
        Jpa = np.interp(self.th, self.tl, self.Jpa)
        Jna = np.interp(self.th, self.tl, self.Jna)  
        # Calculate concentrations
        Kpa, Kna = rt.K_flow_1d(dxh, ua) 
        Kpv, Knv = rt.K_flow_1d(dxh, uv) 
        jpa, jna = Jpa/dxh, Jna/dxh
        Ca, Cv = rt.conc_1d2cf(self.th, jpa, jna, Kpa, Kna, Kpv, Knv, Kva)
        # Downsample concentrations
        if split:
            self.Ca = sig.sample_1d(Ca, self.mat, loc=self.tx)
            self.Cv = sig.sample_1d(Cv, self.mat, loc=self.tx)
            self.C = self.Ca + self.Cv
        else:
            self.C = sig.sample_1d(Ca+Cv, self.mat, loc=self.tx)


    def plot_conc(self, time=False, data=None):
        Dx = self.dim[1]/self.mat[1]                    
        Dt = self.dim[0]/self.mat[0]         
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx) 
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt) 
        if time:
            rt.plot_Ct_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)
        else:
            rt.plot_Cx_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)


    def plot_split_conc(self, time=False):
        Dx = self.dim[1]/self.mat[1]  
        Dt = self.dim[0]/self.mat[0]  
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx) 
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt) 
        if time:
            rt.plot_Ct_1d2c(tv, xv, self.Ca, self.Cv, rows=6, cols=6)
        else:
            rt.plot_Cx_1d2c(tv, xv, self.Ca, self.Cv, rows=6, cols=6)

    def fitpars(self):
        p = [
            self.ua/self.umax, 
            self.uv/self.umax,
            self.Jpa/self.Jmax, 
            self.Jna/self.Jmax,
            self.Kva/self.Kmax,
        ]
        return np.concatenate(p)

    def set_fitpars(self, p):
        nu = len(self.ua)
        nj = len(self.Jpa)
        self.ua = p[:nu]*self.umax
        self.uv = p[nu:2*nu]*self.umax
        self.Jpa = p[2*nu:2*nu+nj]*self.Jmax
        self.Jna = p[2*nu+nj:2*nu+2*nj]*self.Jmax
        self.Kva = p[2*nu+2*nj:]*self.Kmax

    def fitbounds(self):
        # Parameter bounds (normalized parameters)
        u1 = np.ones(2*len(self.ua))
        u0 = -u1
        J1 = np.ones(2*len(self.Jpa))
        J0 = np.zeros(2*len(self.Jpa))
        # K1 = np.ones(len(self.Kva)-2)
        # K0 = np.zeros(len(self.Kva)-2)
        K1 = np.ones(len(self.Kva))
        K0 = np.zeros(len(self.Kva))
        upper = np.concatenate([u1, J1, K1])
        lower = np.concatenate([u0, J0, K0])
        return (lower, upper)

    def fit_func(self, _, *p):
        p = np.array(p)
        self.set_fitpars(p)
        self.calc_conc()
        return self.C.ravel()
    
    def fit_to(self, Cmeas, umax=None, Jmax=None, Kmax=None, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        if umax is not None:
            self.umax=umax
        if Jmax is not None:
            self.Jmax=Jmax
        if Kmax is not None:
            self.Kmax=Kmax
        # get parameters and initial values
        self.precompute()
        p0 = self.fitpars()
        bounds = self.fitbounds()
        # Fit parameters
        p, pcov = curve_fit(self.fit_func, None, Cmeas.ravel(), p0=p0, bounds=bounds, xtol=xtol, ftol=ftol, gtol=gtol) 
        pcorr = np.linalg.norm(p-p0)/np.linalg.norm(p0)
        # Create predictions
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov, pcorr
    
    def resample(self, scl):
        nt = np.ceil(scl[0]*len(self.tl)).astype(np.uint32)
        nx = np.ceil(scl[1]*len(self.xl)).astype(np.uint32)
        nt = np.amax([nt,2])
        nx = np.amax([nx,2])
        tl = np.linspace(0, self.dim[0], nt)
        xl = np.linspace(0, self.dim[1], nx)
        self.ua = np.interp(xl, self.xl, self.ua)          
        self.uv = np.interp(xl, self.xl, self.uv)
        self.Kva = np.interp(xl, self.xl, self.Kva)
        self.Jpa = np.interp(tl, self.tl, self.Jpa)       
        self.Jna = np.interp(tl, self.tl, self.Jna)   
        self.precompute()

    def mres_fit_to(self, Cmeas, umax=None, Jmax=None, Kmax=None, xtol=1e-3, ftol=1e-3, gtol=1e-3, mxtol=1e-2, export_path=None, filename='mres'):
        it, start, start_mres = 0, time.time(), time.time()
        print('mres level: ', it)
        p, pcov, pcorr = self.fit_to(Cmeas, umax, Jmax, Kmax, xtol, ftol, gtol)
        print('>> calculation time (mins): ', (time.time()-start)/60)
        print('>> parameter correction (%): ', 100*pcorr)
        if export_path is not None:
            file = os.path.join(export_path, filename+'_'+str(it)+'.png')
            self.plot_pars(file=file)
        while pcorr > mxtol:
            self.resample((2,2))
            it, start = it+1, time.time()
            print('mres level: ', it)
            p, pcov, pcorr = self.fit_to(Cmeas, umax, Jmax, Kmax, xtol, ftol, gtol)
            print('>> calculation time (mins): ', (time.time()-start)/60)
            print('>> parameter correction (%): ', 100*pcorr)
            if export_path is not None:
                file = os.path.join(export_path, filename+'_'+str(it)+'.png')
                self.plot_pars(file=file)
        print('Multi-resolution calculation time (mins): ', (time.time()-start_mres)/60)
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov
    
    def _fit_fast_to(self, Cmeas, umax=None, Jmax=None, Kmax=None, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        #fast but does not seem to work very well. Keep to original resolution
        if umax is not None:
            self.umax=umax
        if Jmax is not None:
            self.Jmax=Jmax
        if Kmax is not None:
            self.Kmax=Kmax
        # Downsample measurement concentrations to parameter resolution
        scale = 2
        self.mat = [
            np.amin([scale*len(self.Jpa), Cmeas.shape[0]]), 
            np.amin([scale*len(self.ua), Cmeas.shape[1]]), 
        ]
        Cmeas_lowres = sig.sample_1d(Cmeas, self.mat)
        nx_orig = self.nx
        self.nx = self.mat[1]
        # Perform the fit
        self.precompute()
        p0 = self.fitpars()
        bounds = self.fitbounds()
        p, pcov = curve_fit(self.fit_func, None, Cmeas_lowres.ravel(), p0=p0, bounds=bounds, xtol=xtol, ftol=ftol, gtol=gtol) 
        pcorr = np.linalg.norm(p-p0)/np.linalg.norm(p0)
        # Return to origonal measurement resolution and create predictions
        self.nx = nx_orig
        self.mat = Cmeas.shape
        self.precompute()
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov, pcorr
    

    def _calc_flow(self):
        pass
    # Kva Ca = Kva va ca = F ca
        # fa = ua * va
        # fv = uv * vv
        # fa + fv = f (const)
        # F = Kva * va
        # 0 = ua * va + uv * vv
        # -fa' = Kva * va = -ua' * va - ua * va'
        # 0 = (Kva+ua')va + ua va'



class Perf1D_fpic():

    def __init__(self, 
            dim = [60, 30],       # system dimensions (sec, cm)
            mat = [90, 128],      # sampling matrix (time, space)  
            Jna = [0.1,0.1],    # low-res influx right (mmol/sec/cm^2)
            Jpa = [0.1,0.1],    # low-res influx left (mmol/sec/cm^2)
            af = [0.5, 0.5],      # arterial volume fraction 
            v = [1,1],      # total volume fraction
            F = [0.1,0.1],     # perfusion (mL/sec/mL)
            faL = 0,        # left arterial flow (ml/sec/cm^2)
            f = 0,              # total flow (mL/sec/cm^2) 
            nx = 128,           # high-res nr of voxels
            fmax = 30,          # maximum flow (mL/sec/cm^2) 
            Jmax = 10,          # maximum influx (mmol/sec/cm^2)
            Fmax = 1,           # maximum perfusion (mL/sec/mL)
            vmin = 0.01,         # minimum volume fraction
            vmax = 1.0,         # maximum volume fraction
            afmin = 0.05,         # minimum afterial volume fraction
            afmax = 0.95,         # minimum afterial volume fraction
            ):
        if nx < mat[1]:
            msg = 'Numerical voxel size must be smaller than measured voxel size.'
            raise ValueError(msg)
        # Variables
        self.faL = faL
        self.f = f
        self.af = np.array(af) 
        self.v = np.array(v)
        self.F = np.array(F)
        self.Jpa = np.array(Jpa) 
        self.Jna = np.array(Jna) 
        # Constants
        self.dim = np.array(dim)  
        self.mat = np.array(mat) 
        self.nx = nx     
        self.Jmax = Jmax  
        self.Fmax = Fmax
        self.fmax = fmax
        self.vmin = vmin
        self.vmax = vmax
        self.afmin = afmin
        self.afmax = afmax
        # Concentrations
        self.calc_conc()

    def plot_pars(self, truth=None, file=None):
        xl = np.linspace(0, self.dim[1], len(self.v))
        tl = np.linspace(0, self.dim[0], len(self.Jpa))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(8.27,11.69))
        fig.suptitle('1D 2C flow parameters', fontsize=12)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8, wspace=0.2)
        # Plot left influx
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Left arterial influx (mmol/sec/cm^2)')
        ax1.plot(tl, self.Jpa, 'r--')
        ax1.set_xlim(tl[0], tl[-1])
        # Plot right influx
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Right arterial influx (mmol/sec/cm^2)')
        ax2.plot(tl, self.Jna, 'r--')  
        ax2.set_xlim(tl[0], tl[-1])
        # Plot velocity
        ax3.set_xlabel('Position (cm)')
        ax3.set_ylabel('Volume fraction')
        ax3.plot(xl, self.af*self.v, 'r--', label='Arterial')
        ax3.plot(xl, (1-self.af)*self.v, 'b--', label='Venous')
        ax3.set_xlim(xl[0], xl[-1])
        # Plot perfusion
        ax4.set_xlabel('Position (cm)')
        ax4.set_ylabel('Perfusion (mL/sec/mL)')
        ax4.plot(xl, self.F, 'm--')
        ax4.set_xlim(xl[0], xl[-1])
        # Plot ground truths
        if truth is not None:
            xl = np.linspace(0, truth.dim[1], len(truth.af))
            tl = np.linspace(0, truth.dim[0], len(truth.Jpa))
            ax1.plot(tl, truth.Jpa, 'r-')
            ax2.plot(tl, truth.Jna, 'r-')
            ax3.plot(xl, truth.af*truth.v, 'r-')
            ax3.plot(xl, (1-truth.af)*truth.v, 'b-')
            ax4.plot(xl, truth.F, 'm-')
        ax3.legend()
        if file is None:   
            plt.show()
        else:
            path = os.path.dirname(file)
            if not os.path.exists(path):
                os.makedirs(path)       
            plt.savefig(fname=file)
        plt.close()   
         
    def calc_conc(self, split=False):
        # Upsample spatial fields
        self.xl = np.linspace(0, self.dim[1], len(self.af)) 
        self.xh = np.linspace(0, self.dim[1], self.nx+1)
        dxh = self.xh[1]
        self.xv = np.linspace(dxh/2, self.dim[1]-dxh/2, self.nx)
        af = np.interp(self.xh, self.xl, self.af)  
        v = np.interp(self.xh, self.xl, self.v)
        F = np.interp(self.xh, self.xl, self.F)
        # Calculate derived spatial fields
        va = af*v
        vv = v-va
        fa = self.faL - aux.trapz(self.xh, F)
        fv = self.f - fa
        ua = fa/va
        uv = fv/vv
        Kva = F/va
        Kpa, Kna = rt.K_flow_1d(dxh, ua) 
        Kpv, Knv = rt.K_flow_1d(dxh, uv)
        Kva = (Kva[1:]+Kva[:-1])/2 
        # Find time resolution
        uamax = np.amax(np.abs(ua))
        uvmax = np.amax(np.abs(uv))
        Kmax = np.amax(Kva)
        dth = np.amin([dxh/(uamax + dxh*Kmax), dxh/uvmax])
        # Upsample temporal fields
        nth = np.ceil(1 + self.dim[0]/dth).astype(np.int32)
        self.th = np.linspace(0, self.dim[0], nth)  
        self.tl = np.linspace(0, self.dim[0], len(self.Jpa))
        Jpa = np.interp(self.th, self.tl, self.Jpa)
        Jna = np.interp(self.th, self.tl, self.Jna)
        # Calculate concentrations
        Ca, Cv = rt.conc_1d2cf(self.th, Jpa/dxh, Jna/dxh, Kpa, Kna, Kpv, Knv, Kva)
        # Downsample concentrations
        self.tx = sig.sample_loc_1d((nth,self.nx), self.mat)
        if split:
            self.Ca = sig.sample_1d(Ca, self.mat, loc=self.tx)
            self.Cv = sig.sample_1d(Cv, self.mat, loc=self.tx)
            self.C = self.Ca + self.Cv
        else:
            self.C = sig.sample_1d(Ca+Cv, self.mat, loc=self.tx)

    def plot_conc(self, time=False, data=None):
        Dx = self.dim[1]/self.mat[1]                    
        Dt = self.dim[0]/self.mat[0]         
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx) 
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt) 
        if time:
            rt.plot_Ct_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)
        else:
            rt.plot_Cx_1d(tv, xv, self.C, Cmeas=data, rows=6, cols=6)

    def plot_split_conc(self, time=False):
        Dx = self.dim[1]/self.mat[1]  
        Dt = self.dim[0]/self.mat[0]  
        xv = np.arange(Dx/2, self.dim[1]+Dx/2, Dx) 
        tv = np.arange(Dt/2, self.dim[0]+Dt/2, Dt) 
        if time:
            rt.plot_Ct_1d2c(tv, xv, self.Ca, self.Cv, rows=6, cols=6)
        else:
            rt.plot_Cx_1d2c(tv, xv, self.Ca, self.Cv, rows=6, cols=6)

    def fitpars(self):
        p = [
            (self.af-self.afmin)/(self.afmax-self.afmin), 
            (self.v-self.vmin)/(self.vmax-self.vmin),
            self.F/self.Fmax,
            self.Jpa/self.Jmax, 
            self.Jna/self.Jmax,
            [self.faL/self.fmax],
            [self.f/self.fmax],
        ]
        return np.concatenate(p)

    def set_fitpars(self, p):
        nx = len(self.af)
        nt = len(self.Jpa)
        self.af = p[:nx]*(self.afmax-self.afmin) + self.afmin
        self.v = p[nx:2*nx]*(self.vmax-self.vmin) + self.vmin
        self.F = p[2*nx:3*nx]*self.Fmax
        self.Jpa = p[3*nx:3*nx+nt]*self.Jmax
        self.Jna = p[3*nx+nt:3*nx+2*nt]*self.Jmax
        self.faL = p[-2]*self.fmax
        self.f = p[-1]*self.fmax

    def fitbounds(self):
        x1 = np.ones(3*len(self.v))
        x0 = np.zeros(3*len(self.v))
        t1 = np.ones(2*len(self.Jpa))
        t0 = np.zeros(2*len(self.Jpa))
        f1 = [1,1]
        f0 = [-1,-1]
        upper = np.concatenate([x1, t1, f1])
        lower = np.concatenate([x0, t0, f0])
        return (lower, upper)

    def fit_func(self, _, *p):
        p = np.array(p)
        self.set_fitpars(p)
        self.calc_conc()
        return self.C.ravel()
    
    def fit_to(self, Cmeas, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        # for speed consider downsampling Conc for lower resolutions
        p0 = self.fitpars()
        bounds = self.fitbounds()
        p, pcov = curve_fit(self.fit_func, None, Cmeas.ravel(), p0=p0, bounds=bounds, xtol=xtol, ftol=ftol, gtol=gtol) 
        pcorr = np.linalg.norm(p-p0)/np.linalg.norm(p0)
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov, pcorr
    
    def resample(self, scl):
        nt = np.ceil(scl[0]*len(self.tl)).astype(np.uint32)
        nx = np.ceil(scl[1]*len(self.xl)).astype(np.uint32)
        nt = np.amax([nt,2])
        nx = np.amax([nx,2])
        tl = np.linspace(0, self.dim[0], nt)
        xl = np.linspace(0, self.dim[1], nx)
        self.af = np.interp(xl, self.xl, self.af)          
        self.v = np.interp(xl, self.xl, self.v)
        self.F = np.interp(xl, self.xl, self.F)
        self.Jpa = np.interp(tl, self.tl, self.Jpa)       
        self.Jna = np.interp(tl, self.tl, self.Jna)   

    def mres_fit_to(self, Cmeas, xtol=1e-3, ftol=1e-3, gtol=1e-3, mxtol=1e-2, export_path=None, filename='mres'):
        it, start, start_mres = 0, time.time(), time.time()
        print('mres level: ', it)
        p, pcov, pcorr = self.fit_to(Cmeas, xtol, ftol, gtol)
        print('>> calculation time (mins): ', (time.time()-start)/60)
        print('>> parameter correction (%): ', 100*pcorr)
        if export_path is not None:
            file = os.path.join(export_path, filename+'_'+str(it)+'.png')
            self.plot_pars(file=file)
        while pcorr > mxtol:
            self.resample((2,2))
            it, start = it+1, time.time()
            print('mres level: ', it)
            p, pcov, pcorr = self.fit_to(Cmeas, xtol, ftol, gtol)
            print('>> calculation time (mins): ', (time.time()-start)/60)
            print('>> parameter correction (%): ', 100*pcorr)
            if export_path is not None:
                file = os.path.join(export_path, filename+'_'+str(it)+'.png')
                self.plot_pars(file=file)
        print('Multi-resolution calculation time (mins): ', (time.time()-start_mres)/60)
        self.set_fitpars(p) 
        self.calc_conc() 
        return p, pcov


