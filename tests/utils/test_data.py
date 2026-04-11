import pydmr
import dcmri as dc


def test_fetch():
    data = dc.fetch('tristan_rats_healthy_six_drugs')
    dmr = pydmr.read(data)
    assert 'FA' in dmr['data']

if __name__ == "__main__":

    test_fetch()

    print('All data tests passed!!')