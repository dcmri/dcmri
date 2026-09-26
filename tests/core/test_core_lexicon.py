import numpy as np


import dcmri as dc
import dcmri.core.tools as tools



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
        'group': 'EM',
    },
    'S0': {
        'name': 'Baseline signal intensity',
        'unit': 'au',
        'init': 500.0,
        'bounds': [0.0, np.inf],
        'type': 'signal',
        'group': 'EM',
    },
    'TR': {
        'name': 'Repetition time',
        'unit': 's',
        'init': 0.005,
        'bounds': [0.0, 1.0],
        'type': 'sequence',
        'group': 'EM',
    }
}


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

def test_init():
    """Test parameter initialization, default fallbacks, subset selections, and overrides."""
    # Case 1: Initialize all parameters from lexicon
    p_all = dc.init()
    assert 'vb' in p_all
    p_all = dc.init(lexicon=MOCK_QUANTITIES)
    assert p_all == {'R1': 1.2, 'S0': 500.0, 'TR': 0.005}
    
    # Case 2: Initialize only a specified subset of keys
    p_subset = dc.init(pars=['R1', 'TR'], lexicon=MOCK_QUANTITIES)
    assert p_subset == {'R1': 1.2, 'TR': 0.005}
    assert 'S0' not in p_subset

    # Case 3: Overwrite with user-defined keyword args
    p_override = dc.init(lexicon=MOCK_QUANTITIES, R1=2.5, S0=600.0, unknown_param=999)
    assert p_override['R1'] == 2.5
    assert p_override['S0'] == 600.0
    assert 'unknown_param' not in p_override  # Verify non-lexicon keys are ignored


def test_bounds():
    """Test retrieving parameter bounds globally or as subsets."""
    # Case 1: Retrieve bounds for all parameters
    p_all = dc.bounds()
    assert 'vb' in p_all
    b_all = dc.bounds(lexicon=MOCK_QUANTITIES)
    assert b_all == {
        'R1': [0.0, 10.0],
        'S0': [0.0, np.inf],
        'TR': [0.0, 1.0]
    }

    # Case 2: Retrieve bounds for a specific slice of parameters
    b_subset = dc.bounds(pars=['S0'], lexicon=MOCK_QUANTITIES)
    assert b_subset == {'S0': [0.0, np.inf]}


def test_export_params():
    """Test structured parameter exporting with meta information and optional standard deviations."""
    values = {'R1': 1.5, 'S0': 450.0}
    sdevs = {'R1': 0.1}  # Omitting S0 to check fallback handling

    # Case 1: Standard export containing values and uncertainties
    exported = tools.export_params(values)
    exported = tools.export_params(values, group='EM')
    exported = tools.export_params(values, sdev=sdevs, lexicon=MOCK_QUANTITIES)
    exported = tools.export_params(values, sdev=sdevs)
    exported = tools.export_params(values, num_only=True)
    exported = tools.export_params(values, scalar_only=True, sdev=sdevs)
    
    assert exported['R1'] == {
        'name': 'Tissue R1',
        'unit': 'Hz',
        'value': 1.5,
        'sdev': 0.1
    }
    assert exported['S0'] == {
        'name': 'Signal scaling factor',
        'unit': 'a.u.',
        'value': 450.0,
        'sdev': None  # Missing sdev should map to None
    }

    # Case 2: Export without providing an sdev dictionary
    exported_no_sdev = tools.export_params(values, sdev=None, lexicon=MOCK_QUANTITIES)
    assert exported_no_sdev['R1']['sdev'] is None


def test_string_params():
    """Test construction of cleanly formatted console logging strings, edge data types, and rounding."""
    values = {'R1': 1.23456, 'S0': 500.12345, 'TR': np.array([0.05, 0.06])}
    sdevs = {'R1': 0.01234}

    # Case 1: String building with standard formatting and rounding
    strings = tools.string_params(values, sdev=sdevs, round_to=2)
    strings = tools.string_params(values, sdev=sdevs, round_to=2, lexicon=MOCK_QUANTITIES)
    
    # R1 has sdev, rounded to 2 decimals
    assert strings['R1'] == 'R1 = 1.23 +/- 0.01 1/s     (Longitudinal relaxation rate)'
    # S0 has no sdev, rounded to 2 decimals
    assert strings['S0'] == 'S0 = 500.12 au             (Baseline signal intensity)'

    # Case 2: Verification of scalar sequence unpacker handling list/ndarray with size 1
    values_with_array = {'R1': np.array([1.555])}
    strings_array = tools.string_params(values_with_array, round_to=1, lexicon=MOCK_QUANTITIES)
    assert strings_array['R1'] == 'R1 = array (1,) 1/s     (Longitudinal relaxation rate)'

    # Value = list
    values_with_array = {'R1': [1.555]}
    strings_array = tools.string_params(values_with_array, round_to=1, lexicon=MOCK_QUANTITIES)
    assert strings_array['R1'] == 'R1 = list (1) 1/s     (Longitudinal relaxation rate)'



def test_print_params():
    """Test console output capturing to verify the print wrapper properly outputs strings."""
    values = {'R1': 1.2}
    sdevs = {'R1': 0.1}

    # Execute printing function
    tools.print_params(values, sdev=sdevs, lexicon=MOCK_QUANTITIES)
    tools.print_params(values, sdev=sdevs)
    tools.print_params(values, sdev=sdevs, group='EM')


def test_select_params():
    """Test structural property queries against the parameter dictionary attributes."""
    tools.select_params()
    # Case 1: Select single matching field key
    relaxed_fields = tools.select_params(lexicon=MOCK_QUANTITIES, type='relaxation')
    assert 'R1' in relaxed_fields
    assert 'S0' not in relaxed_fields
    assert relaxed_fields['R1']['name'] == 'Longitudinal relaxation rate'

    # Case 2: Select multiple matching fields criteria
    seq_fields = tools.select_params(lexicon=MOCK_QUANTITIES, type='sequence', unit='s')
    assert 'TR' in seq_fields
    assert len(seq_fields) == 1

    # Case 3: Select targeting non-existent fields / missing metadata values
    empty_selection = tools.select_params(lexicon=MOCK_QUANTITIES, non_existent_key='arbitrary')
    assert empty_selection == {}


if __name__ == '__main__':

    test_init()
    test_bounds()
    test_export_params()
    test_string_params()
    test_print_params()
    test_select_params()
    print('All lexicon tests passing!')