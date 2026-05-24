import pydmr
import pytest
import requests

import dcmri as dc


def test_fetch():
    dc.fetch(clear_cache=True)
    data = dc.fetch('tristan_rats_healthy_six_drugs')
    data = dc.fetch('tristan_rats_healthy_six_drugs')
    dmr = pydmr.read(data)
    assert 'FA' in dmr['data']

    # exceptions
    try:
        dc.fetch('x')
    except:
        pass
    else:
        assert False
    assert dc.fetch() is None
    dc.fetch(clear_cache=True)
    dc.fetch(download_all=True)
    dc.fetch(download_all=True)
    dc.fetch(clear_cache=True)



def test_fetch_data_connection_error(mocker):
    dc.fetch(clear_cache=True)

    # 1. Mock requests.get so it throws a ConnectionError when called
    mock_get = mocker.patch("requests.get")
    mock_get.side_effect = requests.exceptions.ConnectionError("Failed to establish a new connection")

    # 2. Assert that calling your function raises the ConnectionError
    with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
        # Pass a dummy URL to your function
        dc.fetch('tristan_rats_healthy_six_drugs')
    
    # 3. Assert that your custom, helpful error message is present in the output
    assert "A connection error occurred trying to download the test data" in str(exc_info.value)
    assert "from Zenodo. This usually happens if you are offline." in str(exc_info.value)
    assert "Failed to establish a new connection" in str(exc_info.value)



if __name__ == "__main__":

    test_fetch()

    print('All data tests passed!!')