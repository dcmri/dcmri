import numpy as np
import dcmri


def fit_aorta_signal_6(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None, baseline=None,
        **vars):

    if parset == 'TRISTAN':
        #           BAT,    CO,     T_hl,   D_hl,   Tp_o,   E_b
        pars = [    60,     100,    10,     0.2,    20,     0.05]
        lb = [      0,      0,      0,      0.05,   0,      0.01]
        ub = [      np.inf, np.inf, 30,     0.95,   60,     0.5]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(6)
        
    # Estimate BAT from data
    T, D = pars[2], pars[3]
    BAT = time[np.argmax(signal)] - (1-D)*T
    pars[0] = BAT

    # Estimate S0 from data
    if baseline is None:
        baseline = time[time <= BAT-20].size
    baseline = max([baseline, 1])
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:baseline]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.aorta_signal_6, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.aorta_signal_6(time, *pars, **vars)
    return pars, pcov, fit


def fit_aorta_signal_7(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None, baseline=None,
        **vars):

    if parset == 'TRISTAN':
        #           BAT,    CO,     T_hl,   E_o,    Tp_o,   Te_o,   E_b
        pars = [    60,     100,    10,     0.15,   20,     120,    0.05]
        lb = [      0,      0,      0,      0,      0,      0,      0.01]
        ub = [      np.inf, np.inf, 30,     0.5,    60,     800,    0.15]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(7)
        
    # Estimate BAT from data
    pars[0] = time[np.argmax(signal)] - pars[2]

    # Estimate S0 from data
    if baseline is None:
        baseline = time[time <= BAT-20].size
    baseline = max([baseline, 1])
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:baseline]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.aorta_signal_7, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.aorta_signal_7(time, *pars, **vars)
    return pars, pcov, fit

def fit_aorta_signal_8(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None, baseline=None,
        **vars):

    if parset == 'TRISTAN':
        #           BAT,    CO,     T_hl,   D_hl,   E_o,    Tp_o,   Te_o,   E_b
        pars = [    60,     100,    10,     0.2,    0.15,   20,     120,    0.05]
        lb = [      0,      0,      0,      0.05,   0,      0,      0,      0.01]
        ub = [      np.inf, np.inf, 30,     0.95,   0.5,    60,     800,    0.15]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(8)
        
    # Estimate BAT from data
    T, D = pars[2], pars[3]
    BAT = time[np.argmax(signal)] - (1-D)*T
    pars[0] = BAT

    # Estimate S0 from data
    if baseline is None:
        baseline = time[time <= BAT-20].size
    baseline = max([baseline, 1])
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:baseline]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.aorta_signal_8, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.aorta_signal_8(time, *pars, **vars)
    return pars, pcov, fit


def fit_aorta_signal_8b(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None,
        **vars):

    if parset == 'TRISTAN':
        #           BAT,    CO,     T_hl,   D_hl,   E_o,    Tp_o,   Te_o,   E_b
        pars = [    60,     100,    10,     0.2,    0.15,   20,     120,    0.05]
        lb = [      0,      0,      0,      0.05,   0,      0,      0,      0.01]
        ub = [      np.inf, np.inf, 30,     0.95,   0.5,    60,     800,    0.15]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(8)
        
    # Estimate BAT and S0 from data
    T, D = pars[2], pars[3]
    BAT = time[np.argmax(signal)] - (1-D)*T
    baseline = time[time <= BAT-20]
    n0 = baseline.size
    if n0 == 0: 
        n0 = 1
    Sref = dcmri.signal_sr(vars['R10'], 1, vars['TD'])
    S0 = np.mean(signal[:n0]) / Sref

    # Initialize variables
    pars[0] = BAT
    vars['S0'] = S0 

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.aorta_signal_8b, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-6,
    ) 
    fit = dcmri.aorta_signal_8b(time, *pars, **vars)
    return pars, pcov, fit


def fit_aorta_signal_9(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None, baseline=None,
        **vars):

    if parset == 'TRISTAN':
        #           BAT,    CO,     T_hl,   D_hl,   E_o,    Tp_o,   Te_o,   E_b,    TI
        pars = [    60,     100,    10,     0.2,    0.15,   20,     120,    0.05,   1.0]
        lb = [      0,      0,      0,      0.05,   0,      0,      0,      0.01,   0.0]
        ub = [      np.inf, np.inf, 30,     0.95,   0.5,    60,     800,    0.15,   10.0]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(9)
        
    # Estimate BAT from data
    T, D = pars[2], pars[3]
    BAT = time[np.argmax(signal)] - (1-D)*T
    pars[0] = BAT

    # Estimate S0 from data
    if baseline is None:
        baseline = time[time <= BAT-20].size
    baseline = max([baseline, 1])
    Sref = dcmri.signal_eqspgre(vars['R10'], 1, vars['TR'], vars['FA'], pars[8])
    vars['S0'] = np.mean(signal[:baseline]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.aorta_signal_9, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.aorta_signal_9(time, *pars, **vars)
    return pars, pcov, fit


def fit_tissue_signal_3(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset = None, baseline=None,
        **vars):

    if parset == 'bladder':
        #           vp,     PS,         ve
        pars = [    0.1,    30/6000,    0.2]
        lb = [      0,      0.0,        0]
        ub = [      np.inf,      np.inf,     np.inf]
        #ub = [      1,      np.inf,     1]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(4)

    # Estimate S0 from data
    if baseline is None:
        t, cb = vars['aif']
        BAT = t[np.argmax(cb)]
        baseline = time[time <= BAT-20].size
        baseline = max([baseline, 1])
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:baseline]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.tissue_signal_3, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.tissue_signal_3(time, *pars, **vars)
    return pars, pcov, fit


def fit_tissue_signal_4(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset = None,
        **vars):

    if parset == 'bladder':
        #           Fp,         vp,     PS,         ve
        pars = [    200/6000,   0.1,    30/6000,    0.2]
        lb = [      0,          0,      0,          0]
        ub = [      np.inf,     1,      np.inf,     1]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(4)

    # Estimate S0 from data
    t, cb = vars['aif']
    BAT = t[np.argmax(cb)]
    baseline = time[time <= BAT-20]
    n0 = baseline.size
    if n0 == 0: 
        n0 = 1
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.tissue_signal_4, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.tissue_signal_4(time, *pars, **vars)
    return pars, pcov, fit


def fit_kidney_signal_4(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset = None,
        **vars):

    if parset == 'Dogs':
        #           Fp,         Tp,     Ft,         Tt
        pars = [    200/6000,   5,      30/6000,    120]
        lb = [      0,          0,      0,          1]
        ub = [      np.inf,     8,      np.inf,     np.inf]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(4)

    # Estimate S0 from data
    BAT = time[np.argmax(vars['cb'])]
    baseline = time[time <= BAT-20]
    n0 = baseline.size
    if n0 == 0: 
        n0 = 1
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.kidney_signal_4, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.kidney_signal_4(time, *pars, **vars)
    return pars, pcov, fit


def fit_kidney_signal_4b(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset = None,
        **vars):

    if parset == 'Dogs':
        #           Fp,         Tp,     Ft,         Tt
        pars = [    200/6000,   5,      30/6000,    120]
        lb = [      0,          0,      0,          1]
        ub = [      np.inf,     8,      np.inf,     np.inf]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(4)

    # Estimate S0 from data
    t = vars['dt']*np.arange(len(vars['cb']))
    BAT = t[np.argmax(vars['cb'])]
    baseline = time[time <= BAT-30]
    n0 = baseline.size
    if n0 == 0: 
        n0 = 1
    Sref = dcmri.signal_srspgre(vars['R10'], 1, vars['TR'], vars['FA'], 0, vars['TD'])
    vars['S0'] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.kidney_signal_4b, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-6,
    ) 
    fit = dcmri.kidney_signal_4b(time, *pars, **vars)
    return pars, pcov, fit


def fit_kidney_signal_5b(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset = None,
        **vars):

    if parset == 'iBEAt':
        #           Fp,         Tp,     Ft,         Tt,     Ta
        pars = [    200/6000,   5,      30/6000,    120,    0]
        lb = [      0,          0,      0,          1,      0]
        ub = [      np.inf,     8,      np.inf,     np.inf, 3]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(5)

    # Estimate S0 from data
    t = vars['dt']*np.arange(len(vars['cb']))
    BAT = t[np.argmax(vars['cb'])]
    baseline = time[time <= BAT-30]
    n0 = baseline.size
    if n0 == 0: 
        n0 = 1
    Sref = dcmri.signal_srspgre(vars['R10'], 1, vars['TR'], vars['FA'], 0, vars['TD'])
    vars['S0'] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.kidney_signal_5b, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-6,
    ) 
    fit = dcmri.kidney_signal_5b(time, *pars, **vars)
    return pars, pcov, fit


def fit_kidney_signal_6(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset = None,
        **vars):

    if parset == 'iBEAt':
        #           Fp,         Tp,     Ft,         Tt,     Ta, S0
        pars = [    200/6000,   5,      30/6000,    120,    0,  1]
        lb = [      0,          0,      0,          1,      0,  0]
        ub = [      np.inf,     8,      np.inf,     np.inf, 3,  np.inf]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(6)

    # Estimate S0 from data
    t = vars['dt']*np.arange(len(vars['cb']))
    TTP = t[np.argmax(vars['cb'])]
    n0 = max([time[time<=TTP-30].size, 1])
    Sref = dcmri.signal_srspgre(vars['R10'], 1, vars['TR'], vars['FA'], 0, vars['TD'])
    pars[5] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.kidney_signal_6, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-6,
    ) 
    fit = dcmri.kidney_signal_6(time, *pars, **vars)
    return pars, pcov, fit


def fit_kidney_signal_9(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None,
        **vars):
    
    if parset == 'Dogs':
        #           FF_k,   F_k,    Tv,     h0, h1, h2, h3, h4, h5
        pars = [    0.2,    0.1,    3.0] +  [1]*5
        lb = [      0.01,   0,      1.0] +  [0]*5
        ub = [      1,      1,      10]  +  [np.inf]*5
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(9)

    # Estimate S0 from data
    t = vars['dt']*np.arange(len(vars['J_aorta']))
    BAT = t[np.argmax(vars['J_aorta'])]
    baseline = time[time <= BAT-20]
    n0 = baseline.size
    if n0 == 0: 
        n0 = 1
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.kidney_signal_9, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.kidney_signal_9(time, *pars, **vars)
    return pars, pcov, fit


def fit_kidney_cm_signal_9(
        time, signal_cortex, signal_medulla, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None,
        **vars):
    
    if parset == 'iBEAt':
        #           Fp,         Eg,     fc,     Tg,     Tv,     Tpt,    Tlh,    Tdt,    Tcd
        pars = [    200/6000,   0.15,   0.8,    4,      10,     60,     60,     30,     30]
        lb = [      0.01,       0,      0,      0,      0,      0,      0,      0,      0] 
        ub = [      1,          1,      1,      10,     30,     np.inf, np.inf, np.inf, np.inf] 
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(9)

    # Determine S0 from data
    t = vars['dt']*np.arange(len(vars['cb']))
    BAT = t[np.argmax(vars['cb'])]
    baseline = time[time <= BAT-20]
    n0 = max([baseline.size,1])
    Scref = dcmri.signal_srspgre(vars['R10c'], 1, vars['TR'], vars['FAc'], 0, vars['TD'])
    Smref = dcmri.signal_srspgre(vars['R10m'], 1, vars['TR'], vars['FAm'], 0, vars['TD'])
    vars['S0c'] = np.mean(signal_cortex[:n0]) / Scref
    vars['S0m'] = np.mean(signal_medulla[:n0]) / Smref

    # Perform the fit
    time = np.concatenate((time, time))
    pars, pcov = dcmri.curve_fit(
        dcmri.kidney_cm_signal_9, 
        time, np.concatenate((signal_cortex, signal_medulla)), 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.kidney_cm_signal_9(time, *pars, **vars)
    nt = int(len(time)/2)
    return pars, pcov, fit[:nt], fit[nt:]



def fit_liver_signal_5(time, signal, 
        pars=None, pfix=None, 
        xrange=None, xvalid=None, 
        bounds=(-np.inf, np.inf), 
        parset=None,
        **vars):
   
    if parset == 'TRISTAN':
        #         Te,     De,     ve,     k_he,       Th
        pars = [  30,     0.85,   0.3,    20/6000,    30*60]
        lb = [    0.1,    0,      0.01,   0,          10*60]
        ub = [    60,     1,      0.6,    np.inf,     10*60*60]
        bounds = (lb, ub)

    if pars is None:
        pars = np.zeros(5)

    # Estimate S0 from data
    t = vars['dt']*np.arange(len(vars['cb']))
    BAT = t[np.argmax(vars['cb'])]
    baseline = time[time <= BAT-20]
    n0 = max([baseline.size,1])
    Sref = dcmri.signal_spgress(vars['TR'], vars['FA'], vars['R10'], 1)
    vars['S0'] = np.mean(signal[:n0]) / Sref

    # Perform the fit
    pars, pcov = dcmri.curve_fit(
        dcmri.liver_signal_5, time, signal, 
        pars, vars=vars, 
        pfix=pfix,
        xrange=xrange, 
        xvalid=xvalid,
        bounds=bounds,
        xtol=1e-3,
    ) 
    fit = dcmri.liver_signal_5(time, *pars, **vars)
    return pars, pcov, fit

