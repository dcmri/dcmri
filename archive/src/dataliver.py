import pandas as pd
import numpy as np

def oneshot_onescan(subj):

    const = pd.read_excel(subj, sheet_name='const')
    const.set_index('name', inplace=True)
    dyn1 = pd.read_excel(subj, sheet_name='dyn1')
    molli1 = pd.read_excel(subj, sheet_name='MOLLI1')
    molli2 = pd.read_excel(subj, sheet_name='MOLLI2')
    dyn1.sort_values('time', inplace=True)
    molli1.sort_values('time', inplace=True)
    molli2.sort_values('time', inplace=True)
    t0 = dyn1.time.values[0]
    return (
        dyn1.time.values-t0, dyn1.fa.values, dyn1.aorta.values, dyn1.liver.values,
        molli1.time.values[0]-t0, molli1.aorta.values[0], molli1.liver.values[0],
        molli2.time.values[0]-t0, molli2.aorta.values[0], molli2.liver.values[0],
        const.at['weight', 'value'], const.at['dose1', 'value'], 
    )

def twoshot_twoscan(subj):

    const = pd.read_excel(subj, sheet_name='const')
    const.set_index('name', inplace=True)
    dyn1 = pd.read_excel(subj, sheet_name='dyn1')
    dyn2 = pd.read_excel(subj, sheet_name='dyn2')
    molli1 = pd.read_excel(subj, sheet_name='MOLLI1')
    molli2 = pd.read_excel(subj, sheet_name='MOLLI2')
    molli3 = pd.read_excel(subj, sheet_name='MOLLI3')
    dyn1.sort_values('time', inplace=True)
    dyn2.sort_values('time', inplace=True)
    molli1.sort_values('time', inplace=True)
    molli2.sort_values('time', inplace=True)
    molli3.sort_values('time', inplace=True)
    t0 = dyn1.time.values[0]
    return (
        dyn1.time.values-t0, dyn1.fa.values, dyn1.aorta.values, dyn1.liver.values,
        dyn2.time.values-t0, dyn2.fa.values, dyn2.aorta.values, dyn2.liver.values,
        molli1.time.values[0]-t0, molli1.aorta.values[0], molli1.liver.values[0],
        molli2.time.values[0]-t0, molli2.aorta.values[0], molli2.liver.values[0],
        molli3.time.values[0]-t0, molli3.aorta.values[0], molli3.liver.values[0],
        const.at['weight', 'value'], const.at['dose1', 'value'], const.at['dose2', 'value'],
    )

def oneshot_twoscan(subj):

    (   time1, fa1, aorta1, liver1, 
        time2, fa2, aorta2, liver2,
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2) = twoshot_twoscan(subj)
    i1 = np.nonzero(time2 < time2[0] + 10*60)[0]
    return (
        time1, fa1, aorta1, liver1,
        time2[i1], fa2[i1], aorta2[i1], liver2[i1], 
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
         weight, dose1, 
    )

def twoshot_onescan(subj):

    (   time1, fa1, aorta1, liver1, 
        time2, fa2, aorta2, liver2,
        T1time1, T1aorta1, T1liver1, 
        T1time2, T1aorta2, T1liver2,
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2) = twoshot_twoscan(subj)
    return (
        time2, fa2, aorta2, liver2, 
        T1time3, T1aorta3, T1liver3,
        weight, dose1, dose2, 
    )

