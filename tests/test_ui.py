import json
import os
import zarr
import numpy as np
import pytest

from dcmri.ui import SuperModel



import shutil

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

    print('All ui tests passed!!')