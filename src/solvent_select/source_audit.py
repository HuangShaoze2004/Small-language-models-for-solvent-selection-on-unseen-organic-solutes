"""Conservative training-only archive cross-check; never changes any labels.

ThermoML is a second transcription, not a critically evaluated ground truth.
Only binary liquid/crystal equilibrium mole fractions with exact molecular
identities and temperatures are matched. Additional conditioning variables,
mixtures, ambiguous measurements, and non-mole-fraction properties are skipped.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .views import sha256_file, write_json


def structure(compound):
    value = compound.get('sStandardInChI')
    if not value:
        return None
    molecule = Chem.MolFromInchi(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else None


def extract(document):
    """Return only unambiguously identified pure-solvent solid solubilities."""
    compounds = {c['RegNum']['nOrgNum']: structure(c) for c in document.get('Compound', [])}
    doi = document['Citation']['sDOI'].lower().strip()
    points, skipped = [], Counter()
    for group in document.get('PureOrMixtureData', []):
        components = {c['RegNum']['nOrgNum'] for c in group.get('Component', [])}
        if len(components) != 2:
            skipped['not_binary_solute_pure_solvent'] += 1
            continue
        variables, constants = {}, {}
        extra_condition = False
        for variable in group.get('Variable', []):
            kind = variable.get('VariableID', {}).get('VariableType', {})
            if kind.get('eTemperature') == 'Temperature, K':
                variables[variable['nVarNumber']] = 'temperature_k'
            elif kind.get('ePressure') == 'Pressure, kPa':
                variables[variable['nVarNumber']] = 'pressure_kpa'
            else:
                extra_condition = True
        for constraint in group.get('Constraint', []):
            kind = constraint.get('ConstraintID', {}).get('ConstraintType', {})
            if kind.get('eTemperature') == 'Temperature, K':
                constants['temperature_k'] = constraint['nConstraintValue']
            elif kind.get('ePressure') == 'Pressure, kPa':
                constants['pressure_kpa'] = constraint['nConstraintValue']
            else:
                extra_condition = True
        if extra_condition:
            skipped['additional_unmatched_condition'] += 1
            continue
        for prop in group.get('Property', []):
            method = prop.get('Property-MethodID', {})
            specification = method.get('PropertyGroup', {}).get('CompositionAtPhaseEquilibrium', {})
            if specification.get('ePropName') != 'Mole fraction':
                skipped['not_equilibrium_mole_fraction'] += 1
                continue
            if prop.get('ePresentation') != 'Direct value, X':
                skipped['not_direct_property_value'] += 1
                continue
            if prop.get('PropPhaseID', {}).get('ePropPhase') != 'Liquid':
                skipped['not_liquid_property'] += 1
                continue
            if any(token in str(specification.get('sMethodName', '')).lower()
                   for token in ['calculated', 'correlation', 'simulation']):
                skipped['calculated_property'] += 1
                continue
            solute = method.get('RegNum', {}).get('nOrgNum')
            solids = {p.get('RegNum', {}).get('nOrgNum') for p in group.get('PhaseID', [])
                      if str(p.get('ePhase', '')).startswith(('Crystal', 'Solid'))}
            if solute not in components or solute not in solids:
                skipped['solute_crystal_not_identified'] += 1
                continue
            solvent = next(iter(components - {solute}))
            if not compounds.get(solute) or not compounds.get(solvent):
                skipped['missing_structure'] += 1
                continue
            for index, row in enumerate(group.get('NumValues', [])):
                conditions = constants.copy()
                for value in row.get('VariableValue', []):
                    if value['nVarNumber'] in variables:
                        conditions[variables[value['nVarNumber']]] = value['nVarValue']
                if 'temperature_k' not in conditions:
                    skipped['missing_temperature'] += 1
                    continue
                for value in row.get('PropertyValue', []):
                    if value['nPropNumber'] != prop['nPropNumber']:
                        continue
                    x = float(value['nPropValue'])
                    if not math.isfinite(x) or not 0 < x <= 1:
                        skipped['invalid_mole_fraction'] += 1
                        continue
                    points.append({'source_doi': doi, 'solute_smiles': compounds[solute],
                                   'solvent_smiles': compounds[solvent], 'temperature_k': float(conditions['temperature_k']),
                                   'pressure_kpa': conditions.get('pressure_kpa'), 'reference_x': x,
                                   'reference_logx': math.log10(x), 'group_number': group['nPureOrMixtureDataNumber'],
                                   'property_number': prop['nPropNumber'], 'row_index': index,
                                   'expanded_uncertainty': value.get('CombinedUncertainty', {}).get('nCombExpandUncertValue')})
    return points, dict(skipped)


def compare_training(measurements, manifest, downloaded, output, threshold=.3,
                     selection_rule='All NIST-available DOIs present only in primary training, fixed before numeric comparison.'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    split = pd.read_csv(manifest, dtype={'solute_id': str})
    train_ids = split.loc[split.split == 'train', 'solute_id'].tolist()
    columns = ['record_id', 'solute_id', 'solute_smiles', 'solvent_smiles', 'temperature_k', 'source_doi', 'logx']
    train = pd.read_parquet(measurements, columns=columns, filters=[('solute_id', 'in', train_ids)])
    download = json.loads(Path(downloaded).read_text())
    # Source selection was saved before this numeric audit; verify it has no non-training source.
    identity = pd.read_parquet(measurements, columns=['solute_id', 'source_doi'])
    other_ids = split.loc[split.split != 'train', 'solute_id'].tolist()
    forbidden = set(identity.loc[identity.solute_id.isin(other_ids), 'source_doi'])
    if set(download['files']) & forbidden:
        raise ValueError('Audit selection contains a heldout target source')
    points, skips, sources = [], Counter(), {}
    for doi, entry in download['files'].items():
        path = Path(entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Downloaded source changed: ' + doi)
        extracted, rejected = extract(json.loads(path.read_text()))
        points.extend(extracted)
        skips.update(rejected)
        sources[doi] = entry['sha256']
    keys = ['source_doi', 'solute_smiles', 'solvent_smiles', 'temperature_k']
    references = defaultdict(list)
    for point in points:
        references[tuple(point[k] for k in keys)].append(point)
    rows = []
    ambiguous = 0
    for row in train.to_dict('records'):
        matches = references.get(tuple(row[k] for k in keys), [])
        if not matches:
            continue
        # Distinct phase/pressure/source replicates must never be arbitrarily chosen.
        values = [p['reference_logx'] for p in matches]
        pressures = {p['pressure_kpa'] for p in matches}
        if max(values) - min(values) > 1e-10 or len(pressures) > 1:
            ambiguous += 1
            continue
        reference = matches[0]
        delta = row['logx'] - reference['reference_logx']
        rows.append({**row, **{k: reference[k] for k in ['reference_x', 'reference_logx', 'pressure_kpa', 'group_number', 'property_number', 'row_index', 'expanded_uncertainty']},
                     'archive_minus_reference_logx': delta, 'archive_to_reference_ratio': 10**delta,
                     'discrepancy_gt_0_3_dex': abs(delta) > threshold + 1e-12,
                     'reference_matches': len(matches), 'reference_document_sha256': sources[row['source_doi']]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No exact source/identity/temperature matches; no correction can be inferred')
    frame.to_csv(output / 'all_training_matches.csv', index=False)
    flags = frame[frame.discrepancy_gt_0_3_dex].sort_values('archive_minus_reference_logx', key=lambda x: x.abs(), ascending=False)
    flags.to_csv(output / 'training_discrepancies.csv', index=False)
    per_source = frame.groupby('source_doi').agg(matched_records=('record_id','size'),
                                               flagged_records=('discrepancy_gt_0_3_dex','sum'),
                                               max_abs_difference_dex=('archive_minus_reference_logx',lambda x:x.abs().max())).reset_index()
    per_source.to_csv(output / 'source_summary.csv', index=False)
    result = {'status': 'TRAINING_ARCHIVE_DISCREPANCY_AUDIT_NOT_LABEL_CORRECTION', 'training_records': len(train),
              'downloaded_training_only_sources': len(download['files']), 'parsed_pure_solvent_equilibrium_points': len(points),
              'exact_unambiguous_training_matches': len(frame), 'matched_sources': frame.source_doi.nunique(),
              'matched_training_solutes': frame.solute_id.nunique(), 'ambiguous_records_skipped': ambiguous,
              'threshold_dex': threshold, 'flagged_records': len(flags), 'flagged_sources': flags.source_doi.nunique(),
              'skip_reasons': dict(skips), 'test_target_values_read': False,
              'data_mutations': False, 'selection_rule': selection_rule,
              'interpretation': 'This measures agreement between archives under recorded identity/temperature. NIST does not guarantee critically evaluated correctness. Unrecorded phase/pH/pressure and transcription differences require primary-source review; do not automatically replace labels.',
              'inputs': {str(p): sha256_file(p) for p in [measurements, manifest, downloaded, __file__]},
              'outputs': {p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', default='data/processed/measurements.parquet')
    parser.add_argument('--manifest', default='data/splits/identity.csv')
    parser.add_argument('--download-manifest', default='artifacts/audit/source_discrepancy/nist_download_manifest.json')
    parser.add_argument('--output', default='artifacts/audit/source_discrepancy/training_archive_check')
    args = parser.parse_args()
    print(json.dumps(compare_training(args.measurements, args.manifest, args.download_manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
