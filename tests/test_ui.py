import json
import os
import zarr
import numpy as np
import pytest

from dcmri.ui import SuperModel, ParsView



import shutil


def test_pars_view():

    class A:
        def __init__(self):
            self._pars = {'a': 1}

        def print(self):
            print(self._pars['a'])

    class B:
        def __init__(self):
            self._a = A()
            self._pars = {'b': 2}
        def print(self):
            print(self._pars['b'])

    b = B()
    b._a.print()

    d = ParsView(b._pars, b._a._pars)
    d['a'] = 3
    b._a.print()
    assert b._a._pars['a'] == 3

    d = ParsView(b._a._pars, b._pars)
    d['a'] = -3
    b._a.print()
    assert b._a._pars['a'] == -3

    try:
        d['c'] = 0
    except KeyError:
        pass
    else:
        assert False


def test_io():
    model = SuperModel()
    folder = "test_model_zarr"
    wrong_folder = "wrong_model_zarr"
    
    try:
        model.save(folder)
        assert os.path.isdir(folder) # Check for directory, not file
        
        # To simulate a "wrong" model, we copy the directory and edit the attrs
        if os.path.exists(wrong_folder): shutil.rmtree(wrong_folder)
        shutil.copytree(folder, wrong_folder)
        
        # Edit the JSON metadata inside the folder
        z_wrong = zarr.open_group(wrong_folder, mode='a')
        meta = z_wrong.attrs.asdict()
        meta['model'] = 'Liver'
        z_wrong.attrs.put(meta)
        
        with pytest.raises(ValueError, match="belongs to Liver"):
            model.load(wrong_folder)
            
    finally:
        if os.path.exists(folder): shutil.rmtree(folder)
        if os.path.exists(wrong_folder): shutil.rmtree(wrong_folder)


if __name__ == "__main__":

    test_io()
    test_pars_view()

    print('All ui tests passed!!')