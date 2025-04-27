import os
import sys
import pickle
import shutil
import zipfile
import csv
from io import TextIOWrapper

import requests
import numpy as np

# filepaths need to be identified with importlib_resources
# rather than __file__ as the latter does not work at runtime
# when the package is installed via pip install

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources


KRUK_DATASETS = [
    'KRUK',
]
TRISTAN_DATASETS = [
    'tristan_humans_healthy_ciclosporin',
    'tristan_humans_healthy_controls_leeds',
    'tristan_humans_healthy_controls_sheffield',
    'tristan_humans_healthy_metformin',
    'tristan_humans_healthy_rifampicin',
    'tristan_humans_patients_rifampicin',
    'tristan_rats_healthy_multiple_dosing',
    'tristan_rats_healthy_reproducibility',
    'tristan_rats_healthy_six_drugs',
]
# TRISTAN_DATASETS = [
#     'tristan_rifampicin',
#     'tristan_gothenburg',
#     'tristan6drugs',
#     'tristan_repro',
#     'tristan_mdosing',
# ]
DMR_DATASETS = [
    'minipig_renal_fibrosis',
] + KRUK_DATASETS + TRISTAN_DATASETS



def fetch(dataset=None, clear_cache=False, download_all=False) -> dict:
    """Fetch a dataset included in dcmri

    Args:
        dataset (str, optional): name of the dataset. See below for options.
        clear_cache (bool, optional): When a dataset is fetched, it is 
          downloaded and then stored in a local cache memory for faster access 
          next time it is fetched. Set clear_cache=True to delete all data 
          in the cache memory. Default is False.
        download_all (bool, optional): By default only the dataset that is 
          fetched is downloaded. Set download_all=True to download all 
          datasets at once. This will cost some time but then offers fast and 
          offline access to all datasets afterwards. This will take up around 
          300 MB of space on your hard drive. Default is False.

    Returns:
        dict: Data as a dictionary.

    Notes:

        The following datasets are currently available:

        `Magnetic resonance renography <https://zenodo.org/records/15284968>`_

            - KRUK

        `TRISTAN Gadoxetate kinetics <https://zenodo.org/records/15285027>`_
        
            - tristan_humans_healthy_rifampicin
            - tristan_humans_healthy_metformin
            - tristan_humans_healthy_ciclosporin
            - tristan_humans_healthy_controls_leeds
            - tristan_humans_healthy_controls_sheffield
            - tristan_rats_healthy_six_drugs
            - tristan_rats_healthy_reproducibility
            - tristan_rats_healthy_multiple_dosing

        Other

            - minipig_renal_fibrosis: Kidney data in a minipig with 
              unilateral ureter stenosis. More detail in future versions..


    Example:

    Fetch the **tristan_humans_healthy_rifampicin** dataset and read it:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc
        >>> import pydmr
        
        # fetch dmr file
        >>> file = dc.fetch('tristan_humans_healthy_rifampicin')

        # read dmr file
        >>> data = pydmr.read(file)

    """

    if dataset is None:
        v = None
    elif dataset in DMR_DATASETS:
        v = _fetch_dmr(dataset)
    else:
        v = _fetch_dataset(dataset)

    if clear_cache:
        _clear_cache()

    if download_all:
        for d in KRUK_DATASETS+TRISTAN_DATASETS:
            _download(d)

    return v


def _clear_cache():
    """
    Clear the folder where the data downloaded via fetch are saved.

    Note if you clear the cache the data will need to be downloaded again 
    if you need them.
    """

    f = importlib_resources.files('dcmri.datafiles')
    for item in f.iterdir(): 
        if item.is_file(): 
            item.unlink() # Delete the file


def _fetch_dmr(dataset):

    f = importlib_resources.files('dcmri.datafiles')
    datafile = str(f.joinpath(dataset + '.dmr'))

    # If this is the first time the data are accessed, download them.
    if not os.path.exists(datafile + '.zip'):
        _download(dataset)

    return datafile


def _fetch_dataset(dataset):

    f = importlib_resources.files('dcmri.datafiles')
    datafile = str(f.joinpath(dataset + '.pkl'))

    # If this is the first time the data are accessed, download them.
    if not os.path.exists(datafile):
        _download(dataset)

    with open(datafile, 'rb') as f:
        v = pickle.load(f)

    return v


def _download(dataset): # add version keyword
        
    f = importlib_resources.files('dcmri.datafiles')

    if dataset in DMR_DATASETS:
        datafile = str(f.joinpath(dataset + '.dmr.zip'))
    else:
        datafile = str(f.joinpath(dataset + '.pkl'))

    if os.path.exists(datafile):
        return

    # Dataset location
    if dataset in KRUK_DATASETS:
        # version_doi = "14957345" # 0.0.0
        # version_doi = "15254891" # 0.0.1
        # version_doi = "15284968" # 0.0.2
        version_doi = "15285017" # 0.0.3
    elif dataset in TRISTAN_DATASETS:
        # version_doi = "14957321" # v0.0.0
        version_doi = "15285027" # v0.0.1
    else:
        raise ValueError(
            f'Dataset {dataset} does not exist. Please choose one of '
            f'{KRUK_DATASETS+TRISTAN_DATASETS}'
        )
    if dataset in DMR_DATASETS:
        file_url = "https://zenodo.org/records/" + version_doi + "/files/" + dataset + ".dmr.zip"
    else:
        file_url = "https://zenodo.org/records/" + version_doi + "/files/" + dataset + ".pkl"

    # Make the request and check for connection error
    try:
        file_response = requests.get(file_url) 
    except requests.exceptions.ConnectionError as err:
        raise requests.exceptions.ConnectionError(
            "\n\n"
            "A connection error occurred trying to download the test data \n"
            "from Zenodo. This usually happens if you are offline. The \n"
            "first time a dataset is fetched via dcmri.fetch you need to \n"
            "be online so the data can be downloaded. After the first \n"
            "time they are saved locally so afterwards you can fetch \n"
            "them even if you are offline. \n\n"
            "The detailed error message is here: " + str(err)) 
    
    # Check for other errors
    file_response.raise_for_status()

    # Save the file locally 
    with open(datafile, 'wb') as f:
        f.write(file_response.content)