
import os
import pandas as pd

import datakidney

from modelskidney import Aorta, Kidney


def fitdata(subj, path, show):

    (time1, fa1, aorta1, kidney, 
        T1time1, T1aorta1, T1kidney, 
        weight, dose1) = datakidney.oneshot_onescan(subj)

    aorta = Aorta()
    aorta.weight = weight
    aorta.dose = dose1
    aorta.set_xy(time1, aorta1)
    aorta.set_R10(T1time1, 1000.0/T1aorta1)
    aorta.estimate_p()
    aorta.fit_p()
    aorta.plot_fit(save=True, show=show, path=path)
    aorta.export_p(path=path)

    liver = Kidney(aorta)
    
    liver.set_xy(time1, kidney)
    liver.set_R10(T1time1, 1000.0/T1kidney)
    liver.estimate_p()
    liver.fit_p()
    liver.plot_fit(save=True, show=show, path=path)
    liver.export_p(path=path)

    return aorta.p, liver.p


 
if __name__ == "__main__":

    subject = ['v4_2']
    filepath = os.path.dirname(__file__)
    show = True

    for s in subject:

        subj = os.path.join(filepath, 'sourcedata', s+'.xlsx')
        path = os.path.join(filepath, 'results', s)
        _, _ = fitdata(subj, path, show)