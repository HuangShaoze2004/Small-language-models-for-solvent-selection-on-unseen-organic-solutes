"""Synthetic ThermoML schema fixtures; these values are not research results."""
import copy

from solvent_select.source_audit import extract


def document():
    return {'Citation': {'sDOI': '10.test/example'}, 'Compound': [
        {'RegNum': {'nOrgNum': 1}, 'sStandardInChI': 'InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H'},
        {'RegNum': {'nOrgNum': 2}, 'sStandardInChI': 'InChI=1S/H2O/h1H2'},
    ], 'PureOrMixtureData': [{
        'nPureOrMixtureDataNumber': 1,
        'Component': [{'RegNum': {'nOrgNum': 1}}, {'RegNum': {'nOrgNum': 2}}],
        'PhaseID': [{'ePhase': 'Liquid'}, {'ePhase': 'Crystal', 'RegNum': {'nOrgNum': 1}}],
        'Variable': [{'nVarNumber': 1, 'VariableID': {'VariableType': {'eTemperature': 'Temperature, K'}}}],
        'Constraint': [{'ConstraintID': {'ConstraintType': {'ePressure': 'Pressure, kPa'}}, 'nConstraintValue': 100}],
        'Property': [{'nPropNumber': 1, 'ePresentation': 'Direct value, X', 'PropPhaseID': {'ePropPhase': 'Liquid'},
                      'Property-MethodID': {'RegNum': {'nOrgNum': 1}, 'PropertyGroup': {'CompositionAtPhaseEquilibrium':
                          {'ePropName': 'Mole fraction', 'sMethodName': 'Analytical'}}}}],
        'NumValues': [{'VariableValue': [{'nVarNumber': 1, 'nVarValue': 298.15}],
                       'PropertyValue': [{'nPropNumber': 1, 'nPropValue': 1e-5}]}],
    }]}


def test_extract_requires_exact_endpoint_and_crystalline_solute():
    points, skips = extract(document())
    assert len(points) == 1
    p = points[0]
    assert p['solute_smiles'] == 'Oc1ccccc1' and p['solvent_smiles'] == 'O'
    assert p['reference_logx'] == -5 and p['pressure_kpa'] == 100
    d = document();d['PureOrMixtureData'][0]['PhaseID'] = [{'ePhase': 'Liquid'}]
    assert extract(d)[0] == []


def test_mass_fraction_is_not_silently_treated_as_mole_fraction():
    d = document()
    d['PureOrMixtureData'][0]['Property'][0]['Property-MethodID']['PropertyGroup']['CompositionAtPhaseEquilibrium']['ePropName'] = 'Mass fraction'
    assert extract(d)[0] == []


def test_mixtures_and_unmatched_conditioning_are_skipped():
    d = document()
    d['PureOrMixtureData'][0]['Component'].append({'RegNum': {'nOrgNum': 3}})
    assert extract(d)[0] == []
    d = document()
    d['PureOrMixtureData'][0]['Variable'].append({'nVarNumber': 2, 'VariableID': {'VariableType': {'sOtherVariable': 'pH'}}})
    points, skipped = extract(d)
    assert points == [] and skipped['additional_unmatched_condition'] == 1


def test_invalid_values_and_non_direct_properties_are_skipped():
    d = document()
    d['PureOrMixtureData'][0]['NumValues'][0]['PropertyValue'][0]['nPropValue'] = 1.2
    assert extract(d)[0] == []
    d = document();d['PureOrMixtureData'][0]['Property'][0]['ePresentation'] = 'Equation'
    assert extract(d)[0] == []
