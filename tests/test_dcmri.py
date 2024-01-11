# tests/test_dcmri.py

# To run all tests:
# cd to dcmri top folder
# pytest --cov=dcmri --cov-report term-missing

from dcmri import __version__

def test_version():
    assert __version__ == "0.2.2"
