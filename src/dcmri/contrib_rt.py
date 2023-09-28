"""Spatial compartment models"""

import numpy as np

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


def conc_space_1d1c(t, Jp, Jn, Kp, Kn):
    """Concentration in a spatial 1-compartment model in 1D"""
    nt = len(Jp)
    nc = len(Kp)
    K = Kp + Kn
    C = np.zeros((nt,nc))
    for k in range(nt-1):
        dt = t[k+1]-t[k]
        # Initialise at current concentration
        C[k+1,:] = C[k,:]
        # Add influxes at the boundaries:
        C[k+1,0] += dt*Jp[k]
        C[k+1,-1] += dt*Jn[k]
        # Remove outflux to the neigbours:
        C[k+1,:] -= dt*K*C[k,:]
        # Add influx from the neighbours:
        C[k+1,:-1] += dt*Kn[1:]*C[k,1:]
        C[k+1,1:] += dt*Kp[:-1]*C[k,:-1]
    return C

def conc_space_2d1c(t, Jpx, Jnx, Jpy, Jny, Kpx, Knx, Kpy, Kny):
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

def conc_space_1d2c(t, Jp1, Jn1, Jp2, Jn2, Kp1, Kn1, Kp2, Kn2, K12, K21):
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
        C1[k+1,:,:] += dt*K21*C1[k,:,:]
        C2[k+1,:,:] += dt*K12*C2[k,:,:]
    return C1, C2

def conc_space_2d2c(t, 
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
        C1[k+1,:,:] += dt*K21*C1[k,:,:]
        C2[k+1,:,:] += dt*K12*C2[k,:,:]
    return C1, C2
