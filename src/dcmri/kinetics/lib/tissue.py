import copy
import numpy as np

from dcmri.kinetics.lib import blocks
import dcmri.kinetics.lib as pk



def dpars_tissue(p, H=0.45):

    p = copy.deepcopy(p)

    if {'vb'}.issubset(p):
        p['vp'] = (1 - H) * p['vb']

    if {'ve', 'vp'}.issubset(p):
        p['vi'] = p['ve'] - p['vp']

    elif {'vp', 'vi'}.issubset(p):
        p['ve'] = p['vp'] + p['vi']

    if {'vb', 'vi'}.issubset(p):
        p['vc'] = 1 - p['vb'] - p['vi']

    return p

# def _all_pars(kin, wex, seq, p):

#     #pars = _model_pars(kin, wex, seq)
#     p = {par: p[par] for par in pars}

#     try:
#         p['Fp'] = p['Fb'] * (1 - p['H'])
#     except KeyError:
#         pass
#     try:
#         p['vp'] = p['vb'] * (1 - p['H'])
#     except KeyError:
#         pass
#     try:
#         p['Ktrans'] = _div(p['Fp'] * p['PS'], p['Fp'] + p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['ve'] = p['vi'] + p['vc']
#     except KeyError:
#         pass
#     try:
#         p['E'] = _div(p['PS'], p['Fp'] + p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['Ti'] = _div(p['vi'], p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['Tp'] = _div(p['vp'], p['PS'] + p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Tb'] = _div(p['vp'], p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Te'] = _div(p['ve'], p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Twc'] = _div(1 - p['vb'] - p['vi'], p['PSc'])
#     except KeyError:
#         pass
#     try:
#         p['Twi'] = _div(p['vi'], p['PSc'] + p['PSe'])
#     except KeyError:
#         pass
#     try:
#         p['Twb'] = _div(p['vb'], p['PSe'])
#     except KeyError:
#         pass
#     try:
#         p['FAcorr'] = p['B1corr'] * p['FA']
#     except KeyError:
#         pass

#     return p


# def _div(a, b):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.where(b == 0, 0, np.divide(a, b))
        

def conc_tissue_u(ca, t=None, dt=1.0, Fb=None):
    ca = np.array(ca)
    C = pk.conc_trap(Fb * ca, t=t, dt=dt)
    return C.reshape(1, -1)
    
def conc_tissue_fx(ca, t=None, dt=1.0, H=None, ve=None, Fb=None):
    ca = np.array(ca)
    if Fb == 0:
        ce = ca*0
    else:
        Fp = (1-H)*Fb
        ce = pk.flux_comp(ca/(1-H), ve/Fp, t=t, dt=dt)
    return ve*ce.reshape(1, -1)

def conc_tissue_nx(ca, t=None, dt=1.0, vb=None, Fb=None):
    ca = np.array(ca)
    if Fb == 0:
        Cb = ca*0
    else:
        Cb = pk.conc_comp(Fb*ca, vb/Fb, t=t, dt=dt)
    return Cb.reshape(1, -1)

def conc_tissue_nxp(ca, t=None, dt=1.0, vb=None, Fb=None):
    ca = np.array(ca)
    if Fb == 0:
        Cb = ca*0
    else:
        Cb = pk.conc_plug(Fb*ca, vb/Fb, t=t, dt=dt)
    return Cb.reshape(1, -1)

def conc_tissue_wv(ca, t=None, dt=1.0, H=None, vi=None, Ktrans=None):
    ca = np.array(ca)
    if Ktrans == 0:
        ci = ca*0
    else:
        ci = pk.flux_comp(ca/(1-H), vi/Ktrans, t=t, dt=dt)
    return vi*ci.reshape(1, -1)

def conc_tissue_hfu(ca, t=None, dt=1.0, H=None, vb=None, PS=None):
    ca = np.array(ca)
    vp = vb*(1-H)
    cp = ca/(1-H)
    Ci = pk.conc_trap(PS*cp, t=t, dt=dt)
    return np.stack((vp*cp, Ci)) 

def conc_tissue_hf(ca, t=None, dt=1.0, H=None, vi=None, vb=None, PS=None):
    ca = np.array(ca)
    vp = vb*(1-H)
    ca = ca/(1-H)
    Cp = vp*ca
    if PS == 0:
        Ci = 0*ca
    else:
        Ci = pk.conc_comp(PS*ca, vi/PS, t=t, dt=dt)
    return np.stack((Cp, Ci))

def conc_tissue_2cu(ca, t=None, dt=1.0, H=None, vb=None, Fb=None, PS=None):
    if np.isinf(Fb):
        return conc_tissue_hfu(ca, t=t, dt=dt, H=H, vb=vb, PS=PS)
    ca = np.array(ca)
    vp = (1 - H) * vb
    Fp = (1 - H) * Fb
    ca = ca / (1 - H)
    if Fp+PS == 0:
        return np.zeros((2, len(ca)))
    Tp = vp/(Fp+PS)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    if vp == 0:
        Ktrans = PS*Fp/(PS+Fp)
        Ci = pk.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ci = pk.conc_trap(PS*Cp/vp, t=t, dt=dt)
    return np.stack((Cp, Ci))

def conc_tissue_2cx(ca, t=None, dt=1.0, H=None, vi=None, vb=None, Fb=None, PS=None):
    ca = np.array(ca)
    vp = (1-H)*vb
    Fp = (1-H)*Fb
    if np.isinf(Fp):
        return conc_tissue_hf(ca, t=t, dt=dt, H=H, vi=vi, vb=vb, PS=PS)

    ca = ca/(1-H)
    J = Fp*ca

    if Fp+PS == 0:
        Cp = np.zeros(len(ca))
        Ce = np.zeros(len(ca))
        return np.stack((Cp, Ce))

    Tp = vp/(Fp+PS)
    E = PS/(Fp+PS)

    if PS == 0:
        Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
        Ci = np.zeros(len(ca))
        return np.stack((Cp, Ci))

    Ti = vi/PS
    C = pk.conc_2cxm(J, [Tp, Ti], E, t=t, dt=dt)
    return C
    




def flux_tissue_u(ca, t=None, dt=1.0, Fb=None):
    ca = np.array(ca)
    return pk.flux(Fb*ca, model='trap')


def flux_tissue_nx(ca, t=None, dt=1.0, vb=None, Fb=None):
    ca = np.array(ca)
    if Fb == 0:
        return np.zeros(len(ca))
    return pk.flux(Fb*ca, vb/Fb, t=t, dt=dt, model='comp')

def flux_tissue_nxp(ca, t=None, dt=1.0, vb=None, Fb=None):
    ca = np.array(ca)
    if Fb == 0:
        return np.zeros(len(ca))
    return pk.flux(Fb*ca, vb/Fb, t=t, dt=dt, model='plug')

def flux_tissue_fx(ca, t=None, dt=1.0, H=None, ve=None, Fb=None):
    ca = np.array(ca)
    if Fb == 0:
        return np.zeros(len(ca))
    Fp = Fb*(1-H)
    return pk.flux(Fb*ca, ve/Fp, t=t, dt=dt, model='comp')

def flux_tissue_wv(ca, t=None, dt=1.0, H=None, vi=None, Ktrans=None):
    ca = np.array(ca)
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = Ktrans*ca
    if Ktrans != 0:
        J[0, 1, :] = pk.flux(Ktrans*ca, vi/Ktrans, t=t, dt=dt, model='comp')
    return J

def flux_tissue_hfu(ca, t=None, dt=1.0, H=None, PS=None):
    ca = np.array(ca)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = PS*ca/(1-H)
    return J


def flux_tissue_hf(ca, t=None, dt=1.0, H=None, vi=None, PS=None):
    ca = np.array(ca)
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.inf
    J[1, 0, :] = PS*ca
    if PS == 0:
        J[0, 1, :] = 0*ca
    else:
        J[0, 1, :] = pk.flux(PS*ca, vi/PS, t=t, dt=dt, model='comp')
    return J


def flux_tissue_2cu(ca, t=None, dt=1.0, H=None, vb=None, Fb=None, PS=None):
    ca = np.array(ca)
    C = conc_tissue_2cu(ca, t=t, dt=dt, H=H, vb=vb, Fb=Fb, PS=PS)
    ca = ca/(1-H)
    Fp = Fb*(1-H)
    J = np.zeros(((2, 2, len(ca))))
    if vb == 0:
        if Fp+PS != 0:
            Ktrans = Fp*PS/(Fp+PS)
            J[0, 0, :] = Fp*ca
            J[1, 0, :] = Ktrans*ca
    else:
        J[0, 0, :] = Fp*C[0, :]/vb
        J[1, 0, :] = PS*C[0, :]/vb
    return J


def flux_tissue_2cx(ca, t=None, dt=1.0, H=None, vb=None, vi=None, Fb=None, PS=None):
    ca = np.array(ca)

    if np.isinf(Fb):
        return flux_tissue_hf(ca, t=t, dt=dt, H=H, vi=vi, PS=PS)

    if Fb == 0:
        return np.zeros((2, 2, len(ca)))

    Fp = Fb*(1-H)

    if PS == 0:
        Jp = flux_tissue_nx(ca, t=t, dt=dt, vb=vb, Fb=Fb)
        J = np.zeros((2, 2, len(ca)))
        J[0, 0, :] = Jp
        return J
    
    C = conc_tissue_2cx(ca, t=t, dt=dt, H=H, vb=vb, vi=vi, Fb=Fb, PS=PS)
    # Derive standard parameters
    vp = vb*(1-H)
    Tp = vp/(Fp+PS)
    Te = vi/PS
    E = PS/(Fp+PS)
    # Build the system matrix K
    T = [Tp, Te]
    E = [
        [1-E, 1],
        [E,   0],
    ]
    return blocks._J_ncomp(C, T, E)


# def flux_tissue_2cf(ca, t=None, dt=1.0, vp=None, Fp=None, PS=None, Te=None):
#     if Fp+PS == 0:
#         return np.zeros((2, 2, len(ca)))
#     # Derive standard parameters
#     Tp = vp/(Fp+PS)
#     E = PS/(Fp+PS)
#     J = Fp*ca
#     T = [Tp, Te]
#     # Solve the system explicitly
#     t = utils.tarray(len(J), t=t, dt=dt)
#     Jo = np.zeros((2, 2, len(t)))
#     J0 = pk.flux(J, T[0], t=t, model='comp')
#     J10 = E*J0
#     Jo[1, 0, :] = J10
#     Jo[1, 1, :] = pk.flux(J10, T[1], t=t, model='comp')
#     Jo[0, 0, :] = (1-E)*J0
#     return Jo
