import requests
import shutil
from pathlib import Path
import requests
import platformdirs

# Define standard user cache directory for dcmri
CACHE_DIR = Path(platformdirs.user_cache_dir(appname="dcmri"))



# Zenodo DOI of the repository
# DOIs need to be updated when new versions are created
DOI = {
    'MRR': "20364938",      # v0.0.4
    'TRISTAN': "20210596"   # v0.0.10
}

# Datasets available via fetch()
DATASETS = {
    'KRUK': {'doi': DOI['MRR'], 'ext': '.dmr.zip'},
    'minipig_renal_fibrosis': {'doi': DOI['MRR'], 'ext': '.dmr.zip'},
    'tristan_humans_healthy_controls': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_humans_healthy_ciclosporin': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_humans_healthy_metformin': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_humans_healthy_rifampicin': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_humans_patients_rifampicin': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_01_rifampicin_effect_size': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_02_rifampicin_effect_size': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_03_rifampicin_effect_size': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_04_rifampicin_effect_size': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_05_single_asunaprevir': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_06_single_pioglitazone': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_07_single_ketoconazole': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_08_single_cyclosporine': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_09_single_bosentan_high': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_10_single_bosentan': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_11_chronic_bosentan_placebo': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_12_single_rifampicin': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_13_field_strength': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_14_chronic_rifampicin_placebo': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
    'tristan_rats_study_15_chronic_cyclosporine_placebo': {'doi': DOI['TRISTAN'], 'ext': '.dmr.zip'},
}

def _get_cache_dir() -> Path:
    """Ensure cache directory exists and return its Path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR

def _download(dataset):
    cache_path = _get_cache_dir()
    datafile = cache_path / (dataset + DATASETS[dataset]["ext"])

    if datafile.exists():
        return

    # Dataset repository
    version_doi = DATASETS[dataset]["doi"]

    # Dataset download link
    file_url = (
        "https://zenodo.org/records/"
        + version_doi
        + "/files/"
        + dataset
        + DATASETS[dataset]["ext"]
    )

    # Make the request and check for connection error
    try:
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()
    except requests.exceptions.ConnectionError as err:
        raise requests.exceptions.ConnectionError(
            "\n\n"
            "A connection error occurred trying to download the test data \n"
            "from Zenodo. This usually happens if you are offline. The \n"
            "first time a dataset is fetched via dcmri.fetch you need to \n"
            "be online so the data can be downloaded. After the first \n"
            "time they are saved locally so afterwards you can fetch \n"
            "them even if you are offline. \n\n"
            "The detailed error message is here: " + str(err)
        )

    # Save the file locally chunk by chunk
    with open(datafile, "wb") as f:
        for chunk in file_response.iter_content(chunk_size=8192):
            f.write(chunk)

def _fetch_dataset(dataset):
    cache_path = _get_cache_dir()
    datafile = cache_path / (dataset + DATASETS[dataset]["ext"])

    # If this is the first time the data are accessed, download them.
    if not datafile.exists():
        _download(dataset)

    return str(datafile)


def clear_cache():
    """
    Clear the folder where the data downloaded via fetch are saved.
    """
    cache_path = _get_cache_dir()
    for item in cache_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def fetch(dataset, download_all=False) -> dict:
    """Fetch a dataset included in dcmri

    Args:
        dataset (str): name of the dataset. See below for options.
        download_all (bool, optional): By default only the dataset that is 
          fetched is downloaded. Set download_all=True to download all 
          datasets at once. This will cost some time but then offers fast and 
          offline access to all datasets afterwards. This will take up around 
          300 MB of space on your hard drive. Default is False.

    Returns:
        list: list of filepaths.

    Notes:

        The following datasets are currently available:

        `Magnetic resonance renography <https://zenodo.org/records/20364938>`_

            - KRUK

        `TRISTAN Gadoxetate kinetics <https://zenodo.org/records/20210596>`_
        
            - tristan_humans_healthy_controls
            - tristan_humans_healthy_ciclosporin
            - tristan_humans_healthy_metformin
            - tristan_humans_healthy_rifampicin
            - tristan_humans_patients_rifampicin
            - tristan_rats_study_01_rifampicin_effect_size
            - tristan_rats_study_02_rifampicin_effect_size
            - tristan_rats_study_03_rifampicin_effect_size
            - tristan_rats_study_04_rifampicin_effect_size
            - tristan_rats_study_05_single_asunaprevir
            - tristan_rats_study_06_single_pioglitazone
            - tristan_rats_study_07_single_ketoconazole
            - tristan_rats_study_08_single_cyclosporine
            - tristan_rats_study_09_single_bosentan_high
            - tristan_rats_study_10_single_bosentan
            - tristan_rats_study_11_chronic_bosentan_placebo
            - tristan_rats_study_12_single_rifampicin
            - tristan_rats_study_13_field_strength
            - tristan_rats_study_14_chronic_rifampicin_placebo
            - tristan_rats_study_15_chronic_cyclosporine_placebo

        Group downloads:
        
            - tristan_rats_healthy_six_drugs
            - tristan_rats_healthy_reproducibility
            - tristan_rats_healthy_multiple_dosing

        Other

            - minipig_renal_fibrosis: Kidney data in a minipig with 
              unilateral ureter stenosis. Data contributed by 
              `Nichlas Vous Christensen <https://www.au.dk/en/nvc@clin.au.dk>`_ 
              and 
              `Mohsen Redda <https://www.au.dk/en/au569527@biomed.au.dk>`_.

                Nikolaj Bøgh, Lotte B Bertelsen, 
                Camilla W Rasmussen, Sabrina K Bech, Anna K Keller, Mia G Madsen, 
                Frederik Harving, Thomas H Thorsen, Ida K Mieritz, Esben Ss Hansen, 
                Alkwin Wanders, Christoffer Laustsen. Metabolic MRI With 
                Hyperpolarized 13C-Pyruvate for Early Detection 
                of Fibrogenic Kidney Metabolism. 
                [`DOI <https://doi.org/10.1097/rli.0000000000001094>`_].


    Example:

    Fetch the **tristan_humans_healthy_rifampicin** dataset and read it:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc
        >>> import pydmr
        
        # fetch data file
        >>> file = dc.fetch('tristan_humans_healthy_rifampicin')

        # read data file
        >>> data = pydmr.read(file)

    """
    if download_all:
        for d in DATASETS.keys():
            _download(d)

    if dataset == 'tristan_rats_healthy_six_drugs':
        all_datasets = [
            'tristan_rats_study_05_single_asunaprevir',
            'tristan_rats_study_06_single_pioglitazone',
            'tristan_rats_study_07_single_ketoconazole',
            'tristan_rats_study_08_single_cyclosporine',
            'tristan_rats_study_09_single_bosentan_high',
            'tristan_rats_study_10_single_bosentan',
            'tristan_rats_study_11_chronic_bosentan_placebo',
            'tristan_rats_study_12_single_rifampicin',
        ]
        return [_fetch_dataset(d) for d in all_datasets]

    if dataset == 'tristan_rats_healthy_reproducibility':
        all_datasets = [
            'tristan_rats_study_01_rifampicin_effect_size',
            'tristan_rats_study_02_rifampicin_effect_size',
            'tristan_rats_study_03_rifampicin_effect_size',
            'tristan_rats_study_04_rifampicin_effect_size',
            'tristan_rats_study_05_single_asunaprevir',
            'tristan_rats_study_06_single_pioglitazone',
            'tristan_rats_study_07_single_ketoconazole',
            'tristan_rats_study_08_single_cyclosporine',
            'tristan_rats_study_09_single_bosentan_high',
            'tristan_rats_study_10_single_bosentan',
            'tristan_rats_study_11_chronic_bosentan_placebo',
            'tristan_rats_study_12_single_rifampicin',
            'tristan_rats_study_13_field_strength',
        ]
        return [_fetch_dataset(d) for d in all_datasets]

    if dataset == 'tristan_rats_healthy_multiple_dosing':
        all_datasets = [
            'tristan_rats_study_11_chronic_bosentan_placebo',
            'tristan_rats_study_14_chronic_rifampicin_placebo',
            'tristan_rats_study_15_chronic_cyclosporine_placebo',
        ]
        return [_fetch_dataset(d) for d in all_datasets]

    if dataset not in DATASETS:
        raise ValueError(
            f"Dataset {dataset} is unknown. Please choose one of {list(DATASETS.keys())}"
        )

    return _fetch_dataset(dataset)







