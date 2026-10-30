"""Auditable immutable curation, molecule-level partitions and measured queries."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
REQUIRED = ['SMILES_Solute', 'Temperature_K', 'Solvent', 'SMILES_Solvent',
            'Solubility(mole_fraction)', 'Source']

def stable_id(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()[:24]

def canonical_identity(smiles: str) -> tuple[str, Chem.Mol]:
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError('invalid_smiles')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), molecule

def organic_parent(molecule: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        raise ValueError('nonorganic_solute')
    return max(organic, key=lambda m: (sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()),
                                      m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))

def exclusion_parent(molecule: Chem.Mol) -> tuple[str, Chem.Mol]:
    # These transformations define ONLY split/exclusion groups, never measurement identity.
    parent = organic_parent(molecule)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    parent = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
    Chem.RemoveStereochemistry(parent)
    return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False), parent

def normalize_doi(value: str) -> str:
    return str(value).strip().lower().removeprefix('https://doi.org/').removeprefix('http://dx.doi.org/').removeprefix('doi:').strip()

def assign_splits(groups: dict[str, list[str]], seed: int, fractions=(0.8, 0.1, 0.1)) -> dict[str, str]:
    """Deterministic label-blind grouped allocation by cumulative solute counts."""
    ordered = sorted(groups, key=lambda key: stable_id(seed, key))
    total = sum(len(v) for v in groups.values())
    boundaries = np.cumsum(fractions) * total
    counts = Counter()
    result = {}
    allocated = 0
    for key in ordered:
        midpoint = allocated + len(groups[key]) / 2
        index = min(int(np.searchsorted(boundaries, midpoint, side='right')), 2)
        split = ['train', 'val', 'test'][index]
        for solute_id in groups[key]:
            result[solute_id] = split
            counts[split] += 1
        allocated += len(groups[key])
    if total >= 30 and set(counts) != {'train', 'val', 'test'}:
        raise ValueError(f'Grouped split degenerate: {dict(counts)}')
    return result

def balanced_blocks(records: list[dict], max_size: int = 8) -> list[list[dict]]:
    """Hash ordering is independent of measurements; no candidate is discarded."""
    ordered = sorted(records, key=lambda r: stable_id('candidate-block', r['solvent_id']))
    count = math.ceil(len(ordered) / max_size)
    if not count:
        return []
    return [list(a) for a in np.array_split(np.array(ordered, dtype=object), count)]

def pair_label(a: float, b: float, tolerance: float = 0.1) -> str:
    if abs(a - b) <= tolerance + 1e-12:
        return 'TIE'
    return 'A' if a > b else 'B'

def curate(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_path = root / 'data/raw/BigSolDBv2.0.csv'
    payload_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if payload_hash != cfg['raw_sha256']:
        raise ValueError('Raw data checksum mismatch')
    raw = pd.read_csv(raw_path)
    if set(REQUIRED) - set(raw):
        raise ValueError(f'Missing required columns: {set(REQUIRED)-set(raw)}')
    audit = {'raw_sha256': payload_hash, 'raw_rows': len(raw),
             'raw_solute_smiles': int(raw.SMILES_Solute.nunique()),
             'raw_solvent_names': int(raw.Solvent.nunique()),
             'raw_source_dois': int(raw.Source.nunique()),
             'raw_missing': raw.isna().sum().to_dict(),
             'missing_conditions': ['pH', 'polymorph', 'equilibration_protocol'],
             'target': 'log10_mole_fraction', 'temperature_tolerance_k': 0.0}
    solutes, solvents, errors = {}, {}, {}
    for smiles in raw.SMILES_Solute.dropna().unique():
        try:
            canonical, mol = canonical_identity(smiles)
            organic = organic_parent(mol)
            if Descriptors.MolWt(organic) > cfg['organic_parent_max_molecular_weight'] or organic.GetNumHeavyAtoms() > cfg['organic_parent_max_heavy_atoms']:
                raise ValueError('outside_small_molecule_scope')
            parent_string, parent = exclusion_parent(mol)
            scaffold = MurckoScaffold.GetScaffoldForMol(parent)
            scaffold_text = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            solutes[smiles] = {'solute_id': stable_id('solute', canonical), 'solute_smiles': canonical,
                              'leakage_group': stable_id('parent', parent_string),
                              'exclusion_parent_smiles': parent_string,
                              'scaffold_text': scaffold_text,
                              'is_multifragment': len(Chem.GetMolFrags(mol)) > 1}
        except Exception as exc:
            errors[('solute', smiles)] = str(exc)
    for smiles in raw.SMILES_Solvent.dropna().unique():
        try:
            canonical, mol = canonical_identity(smiles)
            if len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError('disconnected_solvent_not_unambiguous_pure_solvent')
            solvents[smiles] = {'solvent_id': stable_id('solvent', canonical), 'solvent_smiles': canonical}
        except Exception as exc:
            errors[('solvent', smiles)] = str(exc)
    rows, exclusions = [], []
    for raw_row, item in enumerate(raw.to_dict('records')):
        reason = None
        if item['SMILES_Solute'] not in solutes:
            reason = 'solute:' + errors.get(('solute', item['SMILES_Solute']), 'missing')
        elif item['SMILES_Solvent'] not in solvents:
            reason = 'solvent:' + errors.get(('solvent', item['SMILES_Solvent']), 'missing')
        temperature, x = item['Temperature_K'], item['Solubility(mole_fraction)']
        if not np.isfinite(temperature) or temperature <= 0:
            reason = reason or 'invalid_temperature'
        if not np.isfinite(x) or not 0 < x <= 1:
            reason = reason or 'invalid_mole_fraction'
        doi = normalize_doi(item['Source'])
        if not doi.startswith('10.'):
            reason = reason or 'missing_or_invalid_doi'
        if reason:
            exclusions.append({'raw_row': raw_row, 'reason': reason})
            continue
        rows.append({**solutes[item['SMILES_Solute']], **solvents[item['SMILES_Solvent']],
                     'temperature_k': float(temperature), 'logx': float(np.log10(x)),
                     'mole_fraction': float(x), 'source_doi': doi, 'raw_row': raw_row})
    df = pd.DataFrame(rows)
    duplicate_rows, conflict_rows, cleaned = [], [], []
    keys = ['solute_id', 'solvent_id', 'temperature_k', 'source_doi']
    for key, group in df.groupby(keys, sort=True):
        span = float(group.logx.max() - group.logx.min())
        if len(group) > 1:
            duplicate_rows.append({'key': list(key), 'n': len(group), 'span_dex': span,
                                   'raw_rows': [int(v) for v in group.raw_row]})
        if span > cfg['duplicate_quarantine_span_dex']:
            conflict_rows.extend(group.to_dict('records'))
            continue
        entry = group.iloc[0].to_dict()
        entry['record_id'] = stable_id('measurement', *key)
        entry['logx'] = float(group.logx.median())
        entry['mole_fraction'] = float(10 ** entry['logx'])
        entry['raw_rows'] = json.dumps([int(v) for v in group.raw_row])
        entry['replicate_count'] = len(group)
        entry.pop('raw_row')
        cleaned.append(entry)
    df = pd.DataFrame(cleaned)
    identity = df.drop_duplicates('solute_id')[['solute_id', 'solute_smiles', 'leakage_group', 'exclusion_parent_smiles', 'scaffold_text', 'is_multifragment']].copy()
    acyclic = identity[identity.scaffold_text == ''].drop_duplicates('leakage_group').sort_values('leakage_group')
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in acyclic.exclusion_parent_smiles]
    distances = []
    for index in range(1, len(fps)):
        distances.extend(1 - v for v in DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index]))
    clusters = Butina.ClusterData(distances, len(fps), 1-cfg['acyclic_cluster_tanimoto_cutoff'], isDistData=True) if fps else []
    acyclic_groups = {}
    acyclic_keys = list(acyclic.leakage_group)
    for cluster in clusters:
        members = sorted(acyclic_keys[i] for i in cluster)
        group_id = stable_id('acyclic_cluster', members)
        for member in members:
            acyclic_groups[member] = group_id
    identity['scaffold_group'] = [stable_id('murcko', s) if s else acyclic_groups[g]
                                   for s, g in zip(identity.scaffold_text, identity.leakage_group)]
    df = df.merge(identity[['solute_id', 'scaffold_group']], on='solute_id', validate='many_to_one')
    audit.update({'excluded_rows_by_reason': dict(Counter(r['reason'] for r in exclusions)),
                  'pre_duplicate_rows': len(rows), 'duplicate_groups': duplicate_rows,
                  'quarantined_conflict_rows': len(conflict_rows), 'final_measurements': len(df),
                  'canonical_solutes': int(df.solute_id.nunique()), 'canonical_solvents': int(df.solvent_id.nunique()),
                  'leakage_groups': int(df.leakage_group.nunique()), 'scaffold_groups': int(df.scaffold_group.nunique()),
                  'multifragment_solutes_preserved': int(identity.is_multifragment.sum()),
                  'acyclic_solutes': int((identity.scaffold_text == '').sum())})
    condition_counts = df.groupby(['solute_id', 'source_doi', 'temperature_k']).solvent_id.nunique()
    for n in [2, 3]:
        keep = condition_counts[condition_counts >= n]
        audit[f'conditions_ge{n}'] = len(keep)
        audit[f'solutes_conditions_ge{n}'] = int(keep.index.get_level_values('solute_id').nunique())
    audit_dir = root / 'artifacts/audit'
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusions).to_csv(audit_dir/'excluded_raw_rows.csv', index=False)
    pd.DataFrame(conflict_rows).to_csv(audit_dir/'conflict_rows.csv', index=False)
    return df, identity, audit

def build(root: Path, config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    df, identity, audit = curate(root, cfg)
    processed = root/'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    split_dir = root/'data/splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed/'measurements.parquet', index=False)
    identity.to_parquet(processed/'solute_identities.parquet', index=False)
    # Materialize immutable molecule assignments BEFORE deriving any task examples.
    manifests = {}
    # The strict scaffold protocol is constructed later from this SAME global
    # holdout. Never recreate the superseded independent scaffold split.
    for protocol, group_column in [('identity', 'leakage_group')]:
        groups = identity.groupby(group_column).solute_id.apply(list).to_dict()
        assignments = assign_splits(groups, cfg['split_seed'], cfg['split_fractions'])
        manifest = identity[['solute_id', 'leakage_group', 'scaffold_group']].copy()
        manifest['split'] = manifest.solute_id.map(assignments)
        manifest.to_csv(split_dir/f'{protocol}.csv', index=False)
        assert manifest.groupby('leakage_group').split.nunique().max() == 1
        if protocol == 'scaffold':
            assert manifest.groupby('scaffold_group').split.nunique().max() == 1
        manifests[protocol] = (manifest, assignments)
    all_queries, all_targets = [], []
    for (solute, doi, temperature), condition in df.groupby(['solute_id', 'source_doi', 'temperature_k'], sort=True):
        if len(condition) < 2:
            continue
        base = {'solute_id': solute, 'solute_smiles': condition.iloc[0].solute_smiles,
                'leakage_group': condition.iloc[0].leakage_group, 'temperature_k': float(temperature)}
        source_condition_id = stable_id('physical_condition', solute, doi, temperature)
        for block in balanced_blocks(condition.to_dict('records'), cfg['max_candidates']):
            condition_id = stable_id('condition', solute, doi, temperature, sorted(r['solvent_id'] for r in block))
            for a, b in itertools.combinations(block, 2):
                a, b = sorted([a, b], key=lambda r: stable_id('pair-order', condition_id, r['solvent_id']))
                qid = stable_id('pair', condition_id, a['solvent_id'], b['solvent_id'])
                all_queries.append({**base, 'query_id': qid, 'condition_id': source_condition_id, 'block_id': condition_id, 'task': 'pair',
                                    'candidates': [{'id': cid, 'smiles': r['solvent_smiles'], 'record_id': r['record_id']}
                                                   for cid, r in [('A', a), ('B', b)]]})
                all_targets.append({'query_id': qid, 'choice': pair_label(a['logx'], b['logx'], cfg['pair_tie_dex']),
                                    'values': {'A': a['logx'], 'B': b['logx']}})
            if len(block) >= 3:
                ordered = sorted(block, key=lambda r: stable_id('rank-order', condition_id, r['solvent_id']))
                qid = stable_id('rank', condition_id)
                values = {f'S{i}': r['logx'] for i, r in enumerate(ordered)}
                all_queries.append({**base, 'query_id': qid, 'condition_id': source_condition_id, 'block_id': condition_id, 'task': 'rank',
                                    'candidates': [{'id': f'S{i}', 'smiles': r['solvent_smiles'], 'record_id': r['record_id']}
                                                   for i, r in enumerate(ordered)]})
                all_targets.append({'query_id': qid, 'ranking': sorted(values, key=lambda k: (-values[k], k)), 'values': values})
    audit['all_queries'] = len(all_queries)
    audit['tasks'] = dict(Counter(q['task'] for q in all_queries))
    audit['splits'] = {}
    for protocol, (manifest, assignments) in manifests.items():
        counts = {}
        query_dir = processed/protocol
        target_dir = root/'data/targets'/protocol
        query_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            with open(query_dir/f'{split}_queries.jsonl', 'w') as qf, open(target_dir/f'{split}.jsonl', 'w') as tf:
                task_counts = Counter()
                for query, target in zip(all_queries, all_targets):
                    if assignments[query['solute_id']] == split:
                        qf.write(json.dumps({**query, 'split': split}, separators=(',', ':'))+'\n')
                        tf.write(json.dumps(target, separators=(',', ':'))+'\n')
                        task_counts[query['task']] += 1
            ids = set(manifest.loc[manifest.split == split, 'solute_id'])
            counts[split] = {'solutes': len(ids), 'records': int(df.solute_id.isin(ids).sum()), 'queries': dict(task_counts)}
        audit['splits'][protocol] = counts
    audit['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    audit['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (root/'artifacts/audit/data_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    artifact_hashes = {}
    for base in [processed, split_dir, root/'data/targets']:
        for path in sorted(base.rglob('*')):
            if path.is_file():
                artifact_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'artifacts/audit/data_manifest.json').write_text(json.dumps(artifact_hashes, indent=2)+'\n')
    from .strict_split import build as build_strict
    strict = build_strict(root, seed=cfg['split_seed'], rebuild=True)
    audit['splits']['scaffold'] = {k: strict['counts'][k] for k in ['train', 'val', 'test']}
    audit['strict_scaffold_development_exclusions'] = strict['counts']['excluded']
    audit['strict_scaffold_policy'] = strict['assignment_policy']
    audit['strict_scaffold_amendment'] = 'artifacts/audit/scaffold_amendment.json'
    (root/'artifacts/audit/data_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    for base in [processed, split_dir, root/'data/targets']:
        for path in sorted(base.rglob('*')):
            if path.is_file():
                artifact_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'artifacts/audit/data_manifest.json').write_text(json.dumps(artifact_hashes, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)
    return audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--config', type=Path, default=Path('configs/data.yaml'))
    args = parser.parse_args()
    build(args.root.resolve(), args.config)
