import time, os
import numpy as np
import matplotlib.pyplot as plt

import contrib_dist as dist
import contrib_dro as dro
import contrib_rt as rt
import contrib_syst as syst
import contrib_pk as pk
import aif

def fit_perf_system_1d_fpic():

    truth = dro.organ_perf_1d_fpic(step=True)
    truth.calc_conc(split=True)
    #truth.plot_pars()
    #truth.plot_split_conc()

    # Reconstruct
    rec = syst.Perf1D_fpic(
        dim=truth.dim, 
        mat=truth.mat,
        nx=truth.mat[1])
    #rec.plot_pars()
    #rec.calc_conc(split=True)
    #rec.plot_split_conc()

    rec.mres_fit_to(truth.C, xtol=1e-4, mxtol=1e-2, export_path=os.path.dirname(__file__))


def plot_perf_system_1d_fpic():

    truth = dro.organ_perf_1d_fpic(step=True)
    truth.calc_conc(split=True)
    truth.plot_pars()
    truth.plot_split_conc()


def fit_perf_system_1d():

    truth = dro.organ_perf_1d(step=False, tmax=45)
    truth.calc_conc(split=True)
    truth.plot_pars()
    truth.plot_split_conc()

    # Reconstruct
    rec = syst.Perf1D(
        dim=truth.dim, mat=truth.mat,
        Jpa=[1,1], Jna=[1,1], 
        ua=[1,1], uv=[1,1], 
        Kva = [1,1],
        nx=truth.nx, umax=30, Jmax=5, Kmax=10, 
        )
    
    # start = time.time()
    # p, pcov, pcorr = rec.fit_to(truth.C, xtol=1e-1)
    # print('Calculation time (mins): ', (time.time()-start)/60)
    # print('Parameter correction (%): ', 100*pcorr)
    # rec.plot_conc(data=truth.C)
    # rec.plot_pars(truth=truth)


    # start = time.time()
    # rec.resample((2,2))
    # p, pcov, pcorr = rec.fit_to(truth.C, xtol=1e-1)
    # print('Calculation time (mins): ', (time.time()-start)/60)
    # print('Parameter correction (%): ', 100*pcorr)
    # rec.plot_conc(data=truth.C)
    # rec.plot_pars(truth=truth)

    path = os.path.join(os.path.dirname(__file__), 'results')
    rec.mres_fit_to(truth.C, xtol=1e-4, mxtol=1e-2, export_path=path)
    rec.plot_conc(data=truth.C)
    rec.plot_pars(truth=truth)


def plot_perf_system_1d():

    truth = dro.organ_perf_1d()
    truth.precompute()
    truth.calc_conc(split=True)
    #truth.plot_pars()
    truth.plot_split_conc()


def fit_flow_system_1d():

    # Generate a ground truth system
    tmax = 40 # sec
    length = 30 # cm
    nr = 30
    dt = 0.1 # sec
    umin, umax = 1, 2 # cm/sec

    # Create system explicitly
    t = np.arange(0, tmax, dt)
    Jp = dro.step_inject(t, start=2)
    Jn = 0*Jp
    xc, xb, u, v = dro.flow_organ_1d(length=length, nr=nr, umax=umax, umin=umin, flow=0.5)

    # Calculate concentrations
    truth = syst.Conv1D1C(
        dim=[tmax,length], mat=[len(t),nr],
        Jp=Jp, Jn=Jn, u=u, nx=100,
        )
#    truth.plot_conc()

    # Reconstruct
    res = (20, 10)
    rec = syst.Conv1D1C(
        dim=[tmax,length], mat=[len(t),nr],
        Jp=np.ones(res[0]), Jn=np.ones(res[0]), u=np.ones(res[1]), nx=100,
        )
    #rec.plot_conc()
    start = time.time()
    rec.fit_to(truth.C, xtol=1e-2)
    print('Calculation time (mins): ', (time.time()-start)/60)
    rec.plot_conc(data=truth.C)
    rec.plot_pars(truth=truth)




    
def plot_flow_system_1d():

    # Generate a ground truth system
    tmax = 30 # sec
    length = 30 # cm
    nr = 30
    dt = 0.1 # sec
    umin, umax = 1, 2 # cm/sec

    # Create system explicitly
    t = np.arange(0, tmax, dt)
    Jp = dro.step_inject(t, start=1)
    Jn = 0*Jp
    xc, xb, u, v = dro.flow_organ_1d(length=length, nr=nr, umax=umax, umin=umin, flow=0.5)

    # Calculate with exact u
    syst.Conv1D1C(
        dim=[tmax,length], mat=[len(t),nr],
        Jp=Jp, Jn=Jn, u=u, nx=500,
        ).plot_conc()
    
    # Calculate with u defined at nodes
    u = [umax,umin,umax]
    syst.Conv1D1C(
        dim=[tmax,length], mat=[len(t),nr],
        Jp=Jp, Jn=Jn, u=u, nx=500,
        ).plot_conc()
    

def plot_flow_organ_conc_1d():
    # Define system
    xc, xb, u, v = dro.flow_organ_1d(length=25, nr=30, flow=0.5, umin=1,umax=2)
    dx = xb[1]
    # Define input
    tmax = 45 # sec
    dt = 0.5*np.amin(dx/u) # sec
    t = np.arange(0, tmax, dt)
    Jp = dro.step_inject(t, start=10)
    Jn = np.zeros(len(t))
    # Calculate concentrations
    Kp, Kn = rt.K_flow_1d(dx, u)
    C = rt.conc_1d1c(t, Jp, Jn, Kp, Kn)
    # Plot concentrations
    rt.plot_Cx_1d(t, xc, C, rows=8, cols=8)
    rt.plot_Ct_1d(t, xc, C, rows=8, cols=8)
    

def plot_flow_organ_1d():
    c, b, u, v = dro.flow_organ_1d(flow=0.5, umin=1,umax=2)
    rt.plot_flow_system_1d(c, b, u, v)


def plot_step_inject():
    tmax = 60 # sec
    dt = 0.01 # sec
    t = np.arange(0, tmax, dt)
    J = dro.step_inject(t, start=20)
    fig, ax = plt.subplots(1,1,figsize=(8.27,11.69))
    fig.suptitle('Step injection', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.7, wspace=0.3)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Influx (mmol/sec)')
    ax.plot(t,J)
    plt.show()

def plot_nephc():

    # Tissue constants
    vt = 0.7 # tubular volume fraction (mL/mL)
    Ft = 30/6000 # plasma flow (mL/sec/mL)
    r = 0.90 # reabsorption fraction
    dr = 1e-5 # diffusion rate (1/sec)
    Tp = 5.0 # plasma MTT (sec)

    # Input function
    dt = 1.0 # sec
    tmax = 40*60 # sec
    start = 5 # sec
    nt = 1 + np.ceil(tmax/dt).astype(np.int32)
    t = np.linspace(0,tmax,nt)
    cp = aif.aif_parker(t, start)
    Ct = dist.conc_nephc(t, Ft*cp, r, Ft/vt, dr/vt, n=30)
    vp = 1-vt
    Fp = vp/Tp
    Cp = pk.conc_comp(t, Fp*cp, 1/Tp)
    plt.plot(t, Cp+Ct)
    plt.show()


if __name__ == "__main__":
    #plot_flow_organ_1d()
    #plot_step_inject()
    #plot_flow_organ_conc_1d()
    #plot_flow_system_1d()
    #fit_flow_system_1d()
    plot_perf_system_1d()
    #fit_perf_system_1d()
    #plot_perf_system_1d_fpic()
    #fit_perf_system_1d_fpic()
    #plot_nephc()
    