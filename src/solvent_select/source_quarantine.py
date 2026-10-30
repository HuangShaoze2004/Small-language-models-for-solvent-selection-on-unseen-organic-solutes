"""Versioned conservative source quarantine with inherited, unchanged split roles."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

import pandas as pd

from .views import sha256_file, write_json


def filter_tasks(query_source, target_source, query_output, target_output, removed):
    """Copy measured questions/targets verbatim or remove their entire DOI group."""
    count = Counter()
    excluded = 0
    with Path(query_source).open('rb') as qs, Path(target_source).open('rb') as ts, \
            Path(query_output).open('wb') as qo, Path(target_output).open('wb') as to:
        for qline, tline in zip_longest(qs, ts):
            if qline is None or tline is None:
                raise ValueError('Unpaired query/target source')
            query, target = json.loads(qline), json.loads(tline)
            if query['query_id'] != target['query_id']:
                raise ValueError('Query/target ID mismatch')
            marked = [c['record_id'] in removed for c in query['candidates']]
            if any(marked) and not all(marked):
                raise ValueError('Whole-source quarantine must not alter a candidate subset')
            if any(marked):
                excluded += 1
            else:
                qo.write(qline);to.write(tline)
                count[query['task']] += 1
    return {'retained_by_task': dict(count), 'excluded_queries': excluded}


def retained_manifest(original, ids):
    if original.solute_id.duplicated().any():
        raise ValueError('Original split identities duplicated')
    return original[original.solute_id.isin(ids)].copy()


def build(root, policy_path, tag='sourceqa'):
    root = Path(root).resolve()
    policy_path = Path(policy_path)
    policy = json.loads(policy_path.read_text())
    for path, expected in policy['evidence'].items():
        if sha256_file(path) != expected:
            raise ValueError('Policy evidence changed')
    removed_dois = set(policy['source_dois'])
    flagged_path = next(path for path in policy['evidence'] if path.endswith('training_discrepancies.csv'))
    flagged = pd.read_csv(flagged_path)
    if set(flagged.source_doi) != removed_dois or not (flagged.archive_minus_reference_logx.abs() > policy['threshold_dex']).all():
        raise ValueError('Registered source list must exactly match the numeric training-only evidence')
    processed = root / 'data/processed' / tag
    targets = root / 'data/targets' / tag
    splits = root / 'data/splits' / tag
    audits = root / 'artifacts/audit' / tag
    if any(p.exists() for p in [processed, targets, splits, audits]):
        raise FileExistsError('Versioned source quarantine output exists; never overwrite')
    original_paths = [root / 'data/processed/measurements.parquet', root / 'data/processed/measurements_scaffold.parquet']
    original_paths += [root / 'data/splits' / n for n in ['identity.csv', 'scaffold.csv', 'scaffold_excluded.csv']]
    for protocol in ['identity', 'scaffold']:
        for partition in ['train', 'val', 'test']:
            original_paths += [root / f'data/processed/{protocol}/{partition}_queries.jsonl',
                               root / f'data/targets/{protocol}/{partition}.jsonl']
    before = {str(p): sha256_file(p) for p in original_paths}
    base = pd.read_parquet(root / 'data/processed/measurements.parquet')
    identity = pd.read_csv(root / 'data/splits/identity.csv', dtype={'solute_id': str})
    removal = base[base.source_doi.isin(removed_dois)].copy()
    if set(removal.source_doi) != removed_dois:
        raise ValueError('Some quarantined sources are absent from base data')
    assignment = dict(zip(identity.solute_id, identity.split))
    if set(removal.solute_id.map(assignment)) != {'train'}:
        raise ValueError('The registered policy may remove primary-training-only sources only')
    removed = set(removal.record_id)
    kept = base[~base.record_id.isin(removed)].copy()
    for p in [processed, targets, splits, audits]:
        p.mkdir(parents=True)
    kept.to_parquet(processed / 'measurements.parquet', index=False)
    removal[['record_id', 'solute_id', 'solute_smiles', 'solvent_smiles', 'source_doi', 'temperature_k']].to_csv(
        audits / 'quarantined_records.csv', index=False)
    identity_new = retained_manifest(identity, set(kept.solute_id))
    identity_new.to_csv(splits / 'identity.csv', index=False)
    strict_original = pd.read_csv(root / 'data/splits/scaffold.csv', dtype={'solute_id': str})
    strict_new = retained_manifest(strict_original, set(kept.solute_id))
    strict_new.to_csv(splits / 'scaffold.csv', index=False)
    strict_base = pd.read_parquet(root / 'data/processed/measurements_scaffold.parquet')
    strict_data = strict_base[~strict_base.record_id.isin(removed)].copy()
    strict_data.to_parquet(processed / 'measurements_scaffold.parquet', index=False)
    excluded = pd.read_csv(root / 'data/splits/scaffold_excluded.csv', dtype={'solute_id': str})
    excluded = retained_manifest(excluded, set(kept.solute_id))
    excluded.to_csv(splits / 'scaffold_excluded.csv', index=False)
    counts, task_counts = {}, {}
    for protocol, manifest, measurements in [('identity', identity_new, kept), ('scaffold', strict_new, strict_data)]:
        if set(manifest.solute_id) != set(measurements.solute_id):
            raise ValueError('Measurement and manifest universes differ')
        if manifest.groupby('leakage_group').split.nunique().max() != 1:
            raise ValueError('Parent-group leakage introduced')
        if protocol == 'scaffold' and manifest.groupby('scaffold_group').split.nunique().max() != 1:
            raise ValueError('Scaffold leakage introduced')
        old = identity if protocol == 'identity' else strict_original
        # Membership inheritance is explicit; dropping a train identity cannot shuffle anyone.
        merged = manifest.merge(old, on='solute_id', suffixes=('_new','_old'), validate='one_to_one')
        if not (merged.split_new == merged.split_old).all():
            raise ValueError('Existing split roles changed')
        if set(manifest.loc[manifest.split=='test','solute_id']) != set(old.loc[old.split=='test','solute_id']):
            raise ValueError('Global test identities changed')
        (processed / protocol).mkdir()
        (targets / protocol).mkdir()
        counts[protocol] = {}
        task_counts[protocol] = {}
        for partition in ['train','val','test']:
            qsource = root / f'data/processed/{protocol}/{partition}_queries.jsonl'
            tsource = root / f'data/targets/{protocol}/{partition}.jsonl'
            qdest, tdest = processed / protocol / f'{partition}_queries.jsonl', targets / protocol / f'{partition}.jsonl'
            stat = filter_tasks(qsource, tsource, qdest, tdest, removed)
            ids = set(manifest.loc[manifest.split==partition,'solute_id'])
            counts[protocol][partition] = {'solutes':len(ids),'records':int(measurements.solute_id.isin(ids).sum()),
                                           'queries':stat['retained_by_task']}
            task_counts[protocol][partition] = stat
            if partition=='test' or (protocol=='identity' and partition=='val'):
                if sha256_file(qsource)!=sha256_file(qdest) or sha256_file(tsource)!=sha256_file(tdest):
                    raise ValueError('Protected heldout query/target bytes changed')
    if before != {str(p):sha256_file(p) for p in original_paths}:
        raise ValueError('Original scientific artifacts changed')
    original_audit = json.loads((root/'artifacts/audit/data_audit.json').read_text())
    audit = {k:original_audit[k] for k in ['raw_sha256','raw_rows','pre_duplicate_rows','target','temperature_tolerance_k','missing_conditions']}
    audit.update(status='SOURCE_QUARANTINE_APPLIED',dataset_version=tag,post_duplicate_measurements=len(base),
                 final_measurements=len(kept),source_quarantined_records=len(removal),source_quarantined_dois=len(removed_dois),
                 canonical_solutes=kept.solute_id.nunique(),canonical_solvents=kept.solvent_id.nunique(),
                 leakage_groups=kept.leakage_group.nunique(),scaffold_groups=kept.scaffold_group.nunique(),
                 multifragment_solutes_preserved=int(kept[['solute_id','solute_smiles']].drop_duplicates().solute_smiles.str.contains('.',regex=False).sum()),
                 splits=counts,strict_scaffold_development_exclusions={'solutes':len(excluded),'records':int(kept.solute_id.isin(excluded.solute_id).sum())},
                 excluded_entire_solutes=sorted(set(base.solute_id)-set(kept.solute_id)),source_dois=sorted(removed_dois),
                 policy_path=str(policy_path),policy_sha256=sha256_file(policy_path),
                 grouping='Original full-identity/parent/scaffold groups and partition assignments inherited, never refitted.',
                 labels_replaced=0,global_test_query_target_bytes_identical=True,primary_validation_query_target_bytes_identical=True,
                 task_filter_counts=task_counts,original_artifact_sha256=before,code_sha256=sha256_file(__file__),
                 scope='Conservative quarantine of unresolved training-source discrepancies; not universal validation of remaining experimental values.')
    outputs = {str(p.relative_to(root)):sha256_file(p) for directory in [processed,targets,splits] for p in directory.rglob('*') if p.is_file()}
    write_json(audits/'data_manifest.json',outputs)
    write_json(audits/'data_audit.json',audit)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--policy',default='configs/source_quarantine_v1_reproduce.json')
    parser.add_argument('--tag',default='sourceqa')
    args=parser.parse_args()
    result=build(args.root,args.policy,args.tag)
    print(json.dumps({k:result[k] for k in ['status','final_measurements','canonical_solutes','source_quarantined_records','splits','global_test_query_target_bytes_identical']},indent=2))


if __name__=='__main__':
    main()
