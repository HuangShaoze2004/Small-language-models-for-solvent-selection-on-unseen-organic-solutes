"""Paired solute bootstrap for registered, seed-matched method comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import paired_bootstrap
from .llm import read_jsonl, validate_query_manifest
from .report import _query_context, _run_identity
from .views import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]


def load_evaluations(paths, freeze_manifest=None):
    tables, fingerprints, rows, identities_by_run = [], set(), [], []
    for path in paths:
        path = Path(path)
        metrics_path = path / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        cohort, family, partition = _query_context(metrics, str(path))
        if partition not in ('val', 'test') or cohort.startswith('unverified'):
            raise ValueError('Comparison requires a verified held-out query cohort')
        if partition == 'test':
            freeze = json.loads(Path(freeze_manifest).read_text()) if freeze_manifest else {}
            if freeze.get('status') != 'FROZEN' or freeze.get('query_sha256') != cohort:
                raise ValueError('Test comparison requires frozen query hash before reading outcomes')
        source = next(Path(key) for key in metrics['inputs'] if 'quer' in Path(key).name.lower())
        if not source.is_absolute():
            source = ROOT / source
        if sha256_file(source) != cohort:
            raise ValueError('Registered query source hash changed')
        queries = read_jsonl(source)
        if {q.get('split') for q in queries} != {partition}:
            raise ValueError('Query partition disagrees with evaluation metadata')
        metadata, _, method, seed = _run_identity(path.parent, metrics)
        # Preserve the dataset tag: sourceqa/identity must never silently fall
        # back to the older initial-version manifest with different training IDs.
        try:
            relative = source.relative_to(ROOT / 'data')
            family_index = relative.parts.index(family)
            tags = relative.parts[1:family_index]
        except (ValueError, IndexError):
            tags = ()
        manifest_path = ROOT / 'data/splits'
        for tag in tags:
            manifest_path /= tag
        manifest_path /= family + '.csv'
        expected_split = metadata.get('split_sha256', metadata.get('fingerprint', {}).get('manifest_sha256'))
        if expected_split and sha256_file(manifest_path) != expected_split:
            raise ValueError('Evaluation model and dataset-version split manifest disagree')
        if partition == 'test' and (freeze.get('split_sha256') != sha256_file(manifest_path)
                                    or Path(freeze.get('split_path', '')).resolve() != manifest_path.resolve()):
            raise ValueError('Test comparison requires the exact frozen split manifest')
        validate_query_manifest(queries, manifest_path)
        fingerprints.add((cohort, family, partition))
        identities_by_run.append({'method': method, 'seed': seed,
                                  'adapter_sha256': metadata.get('fingerprint', {}).get('adapter_sha256')})
        # Read qid/identity only first; do not infer pairing solely from aggregate solute counts.
        identities = pd.read_parquet(path / 'per_query.parquet', columns=['query_id', 'solute_id', 'task'])
        if identities.query_id.duplicated().any():
            raise ValueError('Duplicate scored query IDs')
        expected = pd.DataFrame([{key: q[key] for key in ['query_id', 'solute_id', 'task']} for q in queries])
        if not identities.sort_values('query_id').reset_index(drop=True).equals(expected.sort_values('query_id').reset_index(drop=True)):
            raise ValueError('Methods must score identical query identities from the registered source')
        rows.append(identities.sort_values('query_id').reset_index(drop=True))
        table = pd.read_csv(path / 'per_solute.csv', dtype={'solute_id': str})
        if table.duplicated(['solute_id', 'task']).any():
            raise ValueError('Duplicate solute/task observations')
        tables.append(table)
    if len(fingerprints) != 1 or any(not rows[0].equals(frame) for frame in rows[1:]):
        raise ValueError('Methods/seeds must score identical query/solute/task cohorts')
    support = set(map(tuple, tables[0][['solute_id', 'task']].to_numpy()))
    if any(set(map(tuple, t[['solute_id', 'task']].to_numpy())) != support for t in tables[1:]):
        raise ValueError('Methods/seeds must have identical solute/task support')
    return tables, next(iter(fingerprints)), identities_by_run


def compare(first, second, output, freeze_manifest=None, expected_seeds=3):
    if len(first) != len(second) or len(first) != expected_seeds:
        raise ValueError('Require matched preregistered seed counts/order')
    tables, cohort, identities = load_evaluations(list(first) + list(second), freeze_manifest)
    first_ids, second_ids = identities[:len(first)], identities[len(first):]
    first_seeds, second_seeds = [r['seed'] for r in first_ids], [r['seed'] for r in second_ids]
    if len(first) > 1 and (first_seeds != second_seeds or set(first_seeds) != {17, 42, 2026}):
        raise ValueError('Learning seeds must be17/42/2026 in matching order')
    is_primary = (all(r['method'] == 'llm:rag_sft:condition' for r in first_ids)
                  and all(r['method'] == 'llm:rag_sft:structure' for r in second_ids))
    if is_primary and any(not a['adapter_sha256'] or a['adapter_sha256'] != b['adapter_sha256']
                          for a, b in zip(first_ids, second_ids)):
        raise ValueError('Primary retrieval comparison must use the identical adapter per seed')
    a, b = tables[:len(first)], tables[len(first):]
    averaged_a = pd.concat(a).groupby(['solute_id', 'task'], as_index=False).mean(numeric_only=True)
    averaged_b = pd.concat(b).groupby(['solute_id', 'task'], as_index=False).mean(numeric_only=True)
    comparisons = []
    for task, metric in [('pair', 'pair_accuracy'), ('rank', 'top1'), ('rank', 'regret')]:
        estimate = paired_bootstrap(averaged_a, averaged_b, metric, task)
        estimates = [paired_bootstrap(x, y, metric, task, repetitions=1000) for x, y in zip(a, b)]
        estimate['individual_seed_differences'] = [e['difference_first_minus_second'] for e in estimates]
        comparisons.append(estimate)
    pair, _, regret = comparisons
    support = (pair['difference_first_minus_second'] >= .02 and pair['ci95'][0] > 0
               and regret['difference_first_minus_second'] <= -.05 and regret['ci95'][1] < 0
               and all(x > 0 for x in pair['individual_seed_differences'])
               and all(x < 0 for x in regret['individual_seed_differences']))
    result = {'status': 'VALIDATION_ONLY' if cohort[2] == 'val' else 'FROZEN_TEST_COMPARISON',
              'cohort_sha256': cohort[0], 'split_family': cohort[1], 'partition': cohort[2],
              'first_evaluations': list(map(str, first)), 'second_evaluations': list(map(str, second)),
              'seed_count': len(first), 'estimator': 'Average learning seeds within each solute, then paired solute bootstrap.',
              'run_identities': identities, 'is_primary_comparison': is_primary,
              'hypothesis_joint_criteria_supported': bool(support) if is_primary else None, 'comparisons': comparisons,
              'inputs': {str(Path(p) / name): sha256_file(Path(p) / name) for p in list(first) + list(second)
                         for name in ['metrics.json', 'per_solute.csv', 'per_query.parquet']},
              'code_sha256': sha256_file(__file__),
              'note': 'Support flag applies only to the preregistered condition-minus-structure direction. '
                      'Other comparisons are descriptive. Seed ordering must match registered seed metadata.'}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', nargs='+', required=True)
    parser.add_argument('--second', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-seeds', type=int, default=3)
    parser.add_argument('--freeze-manifest')
    args = parser.parse_args()
    print(json.dumps(compare(args.first, args.second, args.output, args.freeze_manifest, args.expected_seeds), indent=2))


if __name__ == '__main__':
    main()
