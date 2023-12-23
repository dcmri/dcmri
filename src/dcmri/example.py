import numpy as np
import matplotlib.pyplot as plt

import contrib_dro as dro
import contrib_rt as rt

def plot_Ct(t: np,ndarray,
            x: np.ndarray,
            C: np.ndarray,
            nr=5,
            nc=4
            ) -> plt:
    # Plot concentrations vs time
    fig, ax = plt.subplots(nr,nc,figsize=(8.27,11.69))
    xi = np.linspace(x[0], x[-1], nr*nc)
    nt = len(t)
    Cmax = np.amax(C)
    i=0
    for r in range(nr):
        for c in range(nc):
            Cx = [np.interp(xi[i], x, C[k,:]) for k in range(nt)]
            ax[r][c].plot(t, Cx)
            ax[r][c].set_ylim(0, Cmax)
            i+=1
    plt.show()
    plt.close()

def plot_Cx(t: np,ndarray,
            x: np.ndarray,
            C: np.ndarray,
            nr=5,
            nc=4
            ) -> plt:
    # Plot concentrations vs time
    fig, ax = plt.subplots(nr,nc,figsize=(8.27,11.69))
    ti = np.linspace(t[0], t[-1], nr*nc)
    nx = len(x)
    Cmax = np.amax(C)
    i=0
    for r in range(nr):
        for c in range(nc):
            Ct = [np.interp(ti[i], t, C[:,k]) for k in range(nx)]
            ax[r][c].plot(x, Ct)
            ax[r][c].set_ylim(0, Cmax)
            i+=1
    plt.show()
    plt.close()


def plot_flow_conc_1d() -> plt:
    # Define system
    organ = dro.flow_organ_1d(flow=0.5, vmin=1,vmax=2)
    dx = organ['voxel boundaries (cm)'][1]
    u = organ['velocity (cm/sec)']
    # Define input
    tmax = 45 # sec
    dt = 0.5*np.amin(dx/u) # sec
    t = np.arange(0, tmax, dt)
    nt = len(t)
    Jp = dro.step_inject(t, start=10)
    Jn = np.zeros(len(t))
    # Calculate concentrations
    Kp, Kn = rt.K_flow_1d(dx, u)
    C = rt.conc_space_1d1c(t, Jp, Jn, Kp, Kn)
    # Plot concentrations vs time
    xc = organ['voxel centers (cm)']
    plot_Ct(t, xc, C)
    plot_Cx(t, xc, C)


def plot_flow_organ_1d() -> plt:
    organ = dro.flow_organ_1d(flow=0.5, vmin=1,vmax=2)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8.27,11.69))
    fig.suptitle('1D flow organ', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.7, wspace=0.3)
    #ax1.set_title('Volume fraction', fontsize=12, pad=10)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Volume fraction')
    ax1.plot(
        organ['voxel centers (cm)'], 
        organ['volume fraction'],
        'k-', marker='o')
    #ax2.set_title('Velocity', fontsize=12, pad=10)
    ax2.set_ylim(0, 25)
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Velocity (cm/sec)')
    ax2.plot(
        organ['voxel boundaries (cm)'], 
        organ['velocity (cm/sec)'],
        'k-', marker='o')
    plt.show()
    plt.close()

def plot_step_inject() -> plt:
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


if __name__ == "__main__":
    #plot_flow_organ_1d()
    #plot_step_inject()
    plot_flow_conc_1d()