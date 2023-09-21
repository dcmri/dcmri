import pandas as pd
import numpy as np

def oneshot_onescan(subj):

    const = pd.read_excel(subj, sheet_name='const')
    const.set_index('name', inplace=True)
    dyn1 = pd.read_excel(subj, sheet_name='dyn1')
    molli1 = pd.read_excel(subj, sheet_name='MOLLI1')
    dyn1.sort_values('time', inplace=True)
    molli1.sort_values('time', inplace=True)
    t0 = dyn1.time.values[0]
    return (
        dyn1.time.values-t0, dyn1.fa.values, dyn1.aorta.values, dyn1.liver.values,
        molli1.time.values[0]-t0, molli1.aorta.values[0], molli1.liver.values[0],
        const.at['weight', 'value'], const.at['dose1', 'value'], 
    )



