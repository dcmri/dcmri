import numpy as np


import dcmri.core.quantity as quantity



# -----------------------------------------------------------------------------
# Test Fixtures / Mock Data
# -----------------------------------------------------------------------------
MOCK_QUANTITIES = {
    'R1': {
        'name': 'Longitudinal relaxation rate',
        'unit': '1/s',
        'init': 1.2,
        'bounds': [0.0, 10.0],
        'type': 'relaxation',
        'group': 'electromagnetic',
    },
    'S0': {
        'name': 'Baseline signal intensity',
        'unit': 'au',
        'init': 500.0,
        'bounds': [0.0, np.inf],
        'type': 'signal',
        'group': 'electromagnetic',
    },
    'TR': {
        'name': 'Repetition time',
        'unit': 's',
        'init': 0.005,
        'bounds': [0.0, 1.0],
        'type': 'sequence',
        'group': 'electromagnetic',
    }
}


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------


def test_select_quantities():
    """Test structural property queries against the parameter dictionary attributes."""
    quantity.select_quantities()
    # Case 1: Select single matching field key
    relaxed_fields = quantity.select_quantities(lexicon=MOCK_QUANTITIES, type='relaxation')
    assert 'R1' in relaxed_fields
    assert 'S0' not in relaxed_fields
    assert relaxed_fields['R1']['name'] == 'Longitudinal relaxation rate'

    # Case 2: Select multiple matching fields criteria
    seq_fields = quantity.select_quantities(lexicon=MOCK_QUANTITIES, type='sequence', unit='s')
    assert 'TR' in seq_fields
    assert len(seq_fields) == 1

    # Case 3: Select targeting non-existent fields / missing metadata values
    empty_selection = quantity.select_quantities(lexicon=MOCK_QUANTITIES, non_existent_key='arbitrary')
    assert empty_selection == {}


if __name__ == '__main__':

    test_select_quantities()
    print('All lexicon tests passing!')