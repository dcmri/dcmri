import numpy as np
import dcmri as dc


SHOW = True # Debugging mode
# SHOW = False # Production mode


def test_ui_artery_tissue():

    tissue = dc.ArteryTissue()
    vFAa, vFAt, siga, sigt = tissue.predict(SNR=50)
    tissue.print(round_to=3)
    tissue.plot(vFAa, vFAt, siga, sigt, round_to=3, show=SHOW)
    tissue.train(vFAa, vFAt, siga, sigt, xtol=1e-3)
    tissue.print(round_to=3)
    tissue.plot(vFAa, vFAt, siga, sigt, round_to=3, show=SHOW)
    assert round(tissue.pars['Fb'], 3) == round(dc.ArteryTissue().pars['Fb'], 3)


if __name__ == "__main__":


    test_ui_artery_tissue()

    print('All ui_artery_tissue tests passed!!')