"""Spatial compartment models"""

import numpy as np
import matplotlib.pyplot as plt
import contrib_sig as sig

# report warnings as errors
import warnings
warnings.filterwarnings("error")


def plot_Ct_1d(t, x, C, Cmeas=None, rows=5, cols=4):
    fontsize=8
    # Plot concentrations vs time
    fig, ax = plt.subplots(rows,cols,figsize=(8.27,11.69))
    fig.suptitle('Tissue concentration vs time at different positions', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)
    xi = np.linspace(x[0], x[-1], rows*cols)
    nt = len(t)
    if Cmeas is None:
        Cmax = np.amax(C)
    else:
        Cmax = np.amax(np.concatenate([C,Cmeas]))
    i=0
    for r in range(rows):
        for c in range(cols):
            if Cmeas is not None:
                Cx = [np.interp(xi[i], x, Cmeas[k,:]) for k in range(nt)]
                ax[r][c].plot(t, Cx, 'g-')
            Cx = [np.interp(xi[i], x, C[k,:]) for k in range(nt)]
            ax[r][c].plot(t, Cx, 'g--')
            ax[r][c].set_ylim(0, 1.2*Cmax)
            ax[r][c].set_xlim(t[0], t[-1])
            tstr = str(round(xi[i], 1)) + ' cm'
            ax[r][c].annotate(tstr, xy=(0.05,0.85), xycoords='axes fraction', fontsize=fontsize)
            if r==rows-1:
                ax[r][c].set_xlabel('Time (sec)', fontsize=fontsize)
                ax[r][c].tick_params(axis='x', labelsize=fontsize)
            else:
                ax[r][c].set_xticks([])
            if c==0:
                ax[r][c].set_ylabel('Conc (mM)', fontsize=fontsize)
                ax[r][c].tick_params(axis='y', labelsize=fontsize)
            else:
                ax[r][c].set_yticks([])
            i+=1
    plt.show()
    plt.close()


def plot_Cx_1d(t, x, C, Cmeas=None, rows=5, cols=4):
    # Plot concentrations vs position
    fontsize=8
    fig, ax = plt.subplots(rows,cols,figsize=(8.27,11.69))
    fig.suptitle('Tissue concentration vs position at different times', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    ti = np.linspace(t[0], t[-1], rows*cols)
    nx = len(x)
    if Cmeas is None:
        Cmax = np.amax(C)
    else:
        Cmax = np.amax(np.concatenate([C,Cmeas]))
    i=0
    for r in range(rows):
        for c in range(cols):
            if Cmeas is not None:
                Ct = [np.interp(ti[i], t, Cmeas[:,k]) for k in range(nx)]
                ax[r][c].plot(x, Ct, 'g-')
            Ct = [np.interp(ti[i], t, C[:,k]) for k in range(nx)]
            ax[r][c].plot(x, Ct, 'g--')
            ax[r][c].set_ylim(0, 1.2*Cmax)
            ax[r][c].set_xlim(x[0], x[-1])
            tstr = str(round(ti[i], 1)) + ' sec'
            ax[r][c].annotate(tstr, xy=(0.05,0.85), xycoords='axes fraction', fontsize=fontsize)
            if r==rows-1:
                ax[r][c].set_xlabel('Position (cm)', fontsize=fontsize)
                ax[r][c].tick_params(axis='x', labelsize=fontsize)
            else:
                ax[r][c].set_xticks([])
            if c==0:
                ax[r][c].set_ylabel('Conc (mM)', fontsize=fontsize)
                ax[r][c].tick_params(axis='y', labelsize=fontsize)
            else:
                ax[r][c].set_yticks([])
            i+=1
    plt.show()
    plt.close()

def plot_Ct_1d2c(t, x, C1, C2, rows=5, cols=4):
    fontsize=8
    # Plot concentrations vs time
    fig, ax = plt.subplots(rows,cols,figsize=(8.27,11.69))
    fig.suptitle('Tissue concentration vs time at different positions', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)
    xi = np.linspace(x[0], x[-1], rows*cols)
    nt = len(t)
    Cmax = np.amax(C1+C2)
    if Cmax==0:
        Cmax=1
    i=0
    for r in range(rows):
        for c in range(cols):
            Cx1 = [np.interp(xi[i], x, C1[k,:]) for k in range(nt)]
            ax[r][c].plot(t, Cx1, 'r-')
            Cx2 = [np.interp(xi[i], x, C2[k,:]) for k in range(nt)]
            ax[r][c].plot(t, Cx2, 'b-')
            ax[r][c].plot(t, np.add(Cx1,Cx2), 'm-')
            ax[r][c].set_ylim(0, 1.2*Cmax)
            ax[r][c].set_xlim(t[0], t[-1])
            tstr = str(round(xi[i], 1)) + ' cm'
            ax[r][c].annotate(tstr, xy=(0.05,0.85), xycoords='axes fraction', fontsize=fontsize)
            if r==rows-1:
                ax[r][c].set_xlabel('Time (sec)', fontsize=fontsize)
                ax[r][c].tick_params(axis='x', labelsize=fontsize)
            else:
                ax[r][c].set_xticks([])
            if c==0:
                ax[r][c].set_ylabel('Conc (mM)', fontsize=fontsize)
                ax[r][c].tick_params(axis='y', labelsize=fontsize)
            else:
                ax[r][c].set_yticks([])
            i+=1
    plt.show()
    plt.close()


def plot_Cx_1d2c(t, x, C1, C2, rows=5, cols=4):
    # Plot concentrations vs position
    fontsize=8
    fig, ax = plt.subplots(rows,cols,figsize=(8.27,11.69))
    fig.suptitle('Tissue concentration vs position at different times', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    ti = np.linspace(t[0], t[-1], rows*cols)
    nx = len(x)
    Cmax = np.amax(C1+C2)
    if Cmax==0:
        Cmax=1
    i=0
    for r in range(rows):
        for c in range(cols):
            Ct1 = [np.interp(ti[i], t, C1[:,k]) for k in range(nx)]
            ax[r][c].plot(x, Ct1, 'r-')
            Ct2 = [np.interp(ti[i], t, C2[:,k]) for k in range(nx)]
            ax[r][c].plot(x, Ct2, 'b-')
            ax[r][c].plot(x, np.add(Ct1, Ct2), 'm-')
            ax[r][c].set_ylim(0, 1.2*Cmax)
            ax[r][c].set_xlim(x[0], x[-1])
            tstr = str(round(ti[i], 1)) + ' sec'
            ax[r][c].annotate(tstr, xy=(0.05,0.85), xycoords='axes fraction', fontsize=fontsize)
            if r==rows-1:
                ax[r][c].set_xlabel('Position (cm)', fontsize=fontsize)
                ax[r][c].tick_params(axis='x', labelsize=fontsize)
            else:
                ax[r][c].set_xticks([])
            if c==0:
                ax[r][c].set_ylabel('Conc (mM)', fontsize=fontsize)
                ax[r][c].tick_params(axis='y', labelsize=fontsize)
            else:
                ax[r][c].set_yticks([])
            i+=1
    plt.show()
    plt.close()



def plot_flow_1d1c_pars(t, x, Jp, Jn, u, t_truth=None, x_truth=None, Jp_truth=None, Jn_truth=None, u_truth=None): 
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(8.27,11.69))
    fig.suptitle('1D 1C flow parameters', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.7, wspace=0.3)
    # Plot left influx
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Left influx (mmol/sec)')
    ax1.plot(t, Jp, 'go')
    if Jp_truth is not None:
        ax1.plot(t_truth, Jp_truth, 'b-')
    # Plot velocity
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Velocity (cm/sec)')
    ax2.plot(x, u, 'ro')
    if u_truth is not None:
        ax2.plot(x_truth, u_truth, 'b-')
    # Plot right influx
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Right influx (mmol/sec)')
    ax3.plot(t, Jn, 'go')
    if Jn_truth is not None:
        ax3.plot(t_truth, Jn_truth, 'b-')
    plt.show()
    plt.close()
    


def plot_flow_system_1d(c, b, u, v):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8.27,11.69))
    fig.suptitle('1D flow organ', fontsize=12)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.7, wspace=0.3)
    #ax1.set_title('Volume fraction', fontsize=12, pad=10)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Volume fraction')
    ax1.plot(c, v, 'k-', marker='o')
    #ax2.set_title('Velocity', fontsize=12, pad=10)
    ax2.set_ylim(0, 25)
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Velocity (cm/sec)')
    ax2.plot(b, u,'k-', marker='o')
    plt.show()
    plt.close()



def K_flow_1d(dx, u):
    nc = len(u)-1
    # Calculate Kn
    Kn = np.zeros(nc)
    un = u[:-1]
    neg = np.where(un < 0)
    Kn[neg] = -un[neg]/dx
    # Calculate Kp
    Kp = np.zeros(nc)
    up = u[1:]
    pos = np.where(up > 0)
    Kp[pos] = up[pos]/dx     
    return Kp, Kn

def K_diff_1d(dx, D):
    Kn = D[:-1]/dx**2
    Kp = D[1:]/dx**2   
    return Kp, Kn

def K_flowdiff_1d(dx, u, D):
    Ku = K_flow_1d(dx, u)
    Kd = K_diff_1d(dx, D)
    Kp = Ku[0] + Kd[0]
    Kn = Ku[1] + Kd[1]
    return Kp, Kn

def F_flow_1d(U, u): 
    # U=dx/dt with dx=voxel width and dt=time step
    nc = len(u)-1
    # Calculate Kn
    Fn = np.zeros(nc)
    un = u[:-1]
    neg = np.where(un < 0)
    Fn[neg] = -un[neg]/U
    # Calculate Kp
    Fp = np.zeros(nc)
    up = u[1:]
    pos = np.where(up > 0)
    Fp[pos] = up[pos]/U     
    return Fp, Fn



def conc_1d1c(t, Jp, Jn, Kp, Kn):
    """Concentration in a spatial 1-compartment model in 1D"""
    # t in sec
    # J in mmol/sec/mL
    # K in 1/sec
    # returns C in mmol/mL
    nt = len(Jp)
    nc = len(Kp)
    K = Kp + Kn
    C = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C[k+1,:] = C[k,:]
        # Add influxes at the boundaries:
        C[k+1,0] += dt*(Jp[k+1]+Jp[k])/2
        C[k+1,-1] += dt*(Jn[k+1]+Jn[k])/2
        # Remove outflux to the neigbours:
        C[k+1,:] -= dt*K*C[k,:]
        # Add influx from the neighbours:
        C[k+1,:-1] += dt*Kn[1:]*C[k,1:]
        C[k+1,1:] += dt*Kp[:-1]*C[k,:-1]
    return C


def dt_1d2cf_pix(dx, u1, u2, K21):
    K1 = u1/dx + K21
    K2 = u2/dx
    Kmax = np.amax(np.concatenate([K1,K2]))
    if Kmax == 0:
        return 1
    MTTmin = 1/Kmax
    return 0.99*MTTmin


def dt_1d2cf(dx, umax, K21max):
    Kmax = umax/dx + K21max
    if Kmax == 0:
        return 1
    MTTmin = 1/Kmax
    return 0.9*MTTmin


def conc_1d2cf(t, Jp1, Jn1, Kp1, Kn1, Kp2, Kn2, K21):
    """Concentration in a spatial 2-compartment filtration model in 1D"""
    # J = flux per unit volume = mmol/sec/mL
    nt = len(Jp1)
    nc = len(Kp1)
    K1 = Kp1 + Kn1 + K21
    K2 = Kp2 + Kn2
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*(Jp1[k+1]+Jp1[k])/2
        C1[k+1,-1] += dt*(Jn1[k+1]+Jn1[k])/2
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        # Add influx from the neighbours:
        C1[k+1,:-1] += dt*Kn1[1:]*C1[k,1:]
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        C2[k+1,:-1] += dt*Kn2[1:]*C2[k,1:]
        C2[k+1,1:] += dt*Kp2[:-1]*C2[k,:-1]
        # Add exchange at the same location
        C2[k+1,:] += dt*K21*C1[k,:]
    return C1, C2

def conc_1d2cx(t, Jp1, Jn1, Kp1, Kn1, K12, K21):
    """Concentration in a spatial 2-compartment exchange model in 1D"""
    nt = len(Jp1)
    nc = len(Kp1)
    K1 = Kp1 + Kn1 + K21
    K2 = K12
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*Jp1[k]
        C1[k+1,-1] += dt*Jn1[k]
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        # Add influx from the neighbours:
        C1[k+1,:-1] += dt*Kn1[1:]*C1[k,1:]
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        # Add influx at the same location
        C2[k+1,:] += dt*K21*C1[k,:]
        C1[k+1,:] += dt*K12*C2[k,:]
    return C1, C2


def conc_2d1c(t, Jpx, Jnx, Jpy, Jny, Kpx, Knx, Kpy, Kny):
    """Concentration in a spatial 1-compartment model in 2D"""
    nt = len(Jpx)
    nx, ny = Kpx.shape
    C = np.zeros((nt,nx,ny))
    K = Kpx + Knx + Kpy + Kny
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C[k+1,:,:] = C[k,:,:]
        # Add influxes at boundaries
        C[k+1,0,:] += dt*Jpx[k,:]
        C[k+1,-1,:] += dt*Jnx[k,:]
        C[k+1,:,0] += dt*Jpy[k,:]
        C[k+1,:,-1] += dt*Jny[k,:]
        # Remove outflux to the neigbours
        C[k+1,:,:] -= dt*K*C[k,:,:]
        # Add influx from the neighbours
        C[k+1,:-1,:] += dt*Knx[1:,:]*C[k,1:,:]
        C[k+1,1:,:] += dt*Kpx[:-1,:]*C[k,:-1,:]
        C[k+1,:,:-1] += dt*Kny[:,1:]*C[k,:,1:]
        C[k+1,:,1:] += dt*Kpy[:,:-1]*C[k,:,:-1]
    return C


def conc_1d2c(t, Jp1, Jn1, Jp2, Jn2, Kp1, Kn1, Kp2, Kn2, K12, K21):
    """Concentration in a spatial 2-compartment model in 1D"""
    nt = len(Jp1)
    nc = len(Kp1)
    K1 = Kp1 + Kn1 + K21
    K2 = Kp2 + Kn2 + K12
    C1 = np.zeros((nt,nc))
    C2 = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:] = C1[k,:]
        C2[k+1,:] = C2[k,:]
        # Add influxes at the boundaries:
        C1[k+1,0] += dt*Jp1[k]
        C1[k+1,-1] += dt*Jn1[k]
        C2[k+1,0] += dt*Jp2[k]
        C2[k+1,-1] += dt*Jn2[k]
        # Remove outflux to the neigbours:
        C1[k+1,:] -= dt*K1*C1[k,:]
        C2[k+1,:] -= dt*K2*C2[k,:]
        # Add influx from the neighbours:
        C1[k+1,:-1] += dt*Kn1[1:]*C1[k,1:]
        C1[k+1,1:] += dt*Kp1[:-1]*C1[k,:-1]
        C2[k+1,:-1] += dt*Kn2[1:]*C2[k,1:]
        C2[k+1,1:] += dt*Kp2[:-1]*C2[k,:-1]
        # Add influx at the same location
        C2[k+1,:] += dt*K21*C1[k,:]
        C1[k+1,:] += dt*K12*C2[k,:]
    return C1, C2

def conc_2d2c(t, 
            Jp1x, Jn1x, Jp2x, Jn2x, 
            Jp1y, Jn1y, Jp2y, Jn2y, 
            Kp1x, Kn1x, Kp2x, Kn2x, 
            Kp1y, Kn1y, Kp2y, Kn2y, 
            K12, K21):
    """Concentration in a spatial 2-compartment model in 2D"""
    nt = len(Jp1x)
    nx, ny = Kp1x.shape
    C = np.zeros((nt,nx,ny))
    K1 = Kp1x + Kn1x + Kp1y + Kn1y + K21
    K2 = Kp2x + Kn2x + Kp2y + Kn2y + K12
    C1 = np.zeros((nt,nx,ny))
    C2 = np.zeros((nt,nx,ny))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C1[k+1,:,:] = C1[k,:,:]
        C2[k+1,:,:] = C2[k,:,:]
        # Add influxes at the boundaries:
        C1[k+1,0,:] += dt*Jp1x[k,:]
        C1[k+1,-1,:] += dt*Jn1x[k,:]
        C1[k+1,:,0] += dt*Jp1y[k,:]
        C1[k+1,:,-1] += dt*Jn1y[k,:]
        C2[k+1,0,:] += dt*Jp2x[k,:]
        C2[k+1,-1,:] += dt*Jn2x[k,:]
        C2[k+1,:,0] += dt*Jp2y[k,:]
        C2[k+1,:,-1] += dt*Jn2y[k,:]
        # Remove outflux to the neigbours:
        C1[k+1,:,:] -= dt*K1*C1[k,:,:]
        C2[k+1,:,:] -= dt*K2*C2[k,:,:]
        # Add influx from the neighbours
        C1[k+1,:-1,:] += dt*Kn1x[1:,:]*C1[k,1:,:]
        C1[k+1,1:,:] += dt*Kp1x[:-1,:]*C1[k,:-1,:]
        C1[k+1,:,:-1] += dt*Kn1y[:,1:]*C1[k,:,1:]
        C1[k+1,:,1:] += dt*Kp1y[:,:-1]*C1[k,:,:-1]
        C2[k+1,:-1,:] += dt*Kn2x[1:,:]*C2[k,1:,:]
        C2[k+1,1:,:] += dt*Kp2x[:-1,:]*C2[k,:-1,:]
        C2[k+1,:,:-1] += dt*Kn2y[:,1:]*C2[k,:,1:]
        C2[k+1,:,1:] += dt*Kp2y[:,:-1]*C2[k,:,:-1]
        # Add influx at the same location
        C2[k+1,:,:] += dt*K21*C1[k,:,:]
        C1[k+1,:,:] += dt*K12*C2[k,:,:]
    return C1, C2
