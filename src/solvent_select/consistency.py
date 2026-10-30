"""Label-free, fixed-cohort candidate-order and equivalent-SMILES checks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .evaluate import parsed_prediction
from .llm import validate_query_manifest
from .retrieval import canonical_smiles
from .views import read_query_metadata, sha256_file, task_solute_round_robin, write_json

SEED = 20261025


def read(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))


def stable_seed(value):
    return int(hashlib.sha256(f'{SEED}|{value}'.encode()).hexdigest()[:8], 16) % 2147483647 or 1


def alternate_smiles(smiles, key):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Invalid molecule for equivalent-notation check')
    options = Chem.MolToRandomSmilesVect(molecule, 16, randomSeed=stable_seed(key), isomericSmiles=True)
    result = next((s for s in options if s != smiles), smiles)
    if canonical_smiles(result) != canonical_smiles(smiles):
        raise ValueError('Randomized SMILES changed molecular identity')
    return result


def transformed(query, variant):
    result = copy.deepcopy(query)
    original_id = query['query_id']
    result['query_id'] = hashlib.sha256(f'consistency-v1|{variant}|{original_id}'.encode()).hexdigest()[:24]
    mapping = {c['id']: c['id'] for c in query['candidates']}
    changed = 0
    if variant == 'candidate_order':
        result['candidates'].reverse()
        if query['task'] == 'pair':
            mapping = {}
            for label, candidate in zip(('A', 'B'), result['candidates']):
                mapping[label] = candidate['id']
                candidate['id'] = label
    elif variant == 'equivalent_smiles':
        old = result['solute_smiles']
        result['solute_smiles'] = alternate_smiles(old, original_id + '|solute')
        changed += result['solute_smiles'] != old
        for candidate in result['candidates']:
            old = candidate['smiles']
            candidate['smiles'] = alternate_smiles(old, original_id + '|' + candidate['id'])
            changed += candidate['smiles'] != old
    else:
        raise ValueError('Unknown consistency variant')
    return result, {'query_id': result['query_id'], 'original_query_id': original_id,
                    'variant': variant, 'candidate_to_original': mapping, 'changed_smiles': changed,
                    'smiles_fields': 1 + len(query['candidates'])}


def build(queries_path, manifest_path, output, limit=512):
    queries = read(queries_path)
    if not queries or len({q['split'] for q in queries}) != 1 or queries[0]['split'] not in ('val', 'test'):
        raise ValueError('Consistency uses one held-out partition')
    validate_query_manifest(queries, manifest_path)
    partition = queries[0]['split']
    metadata = read_query_metadata(queries_path, partition)
    selected = set()
    for task, quota in [('pair', limit // 2), ('rank', limit - limit // 2)]:
        selected.update(task_solute_round_robin([r for r in metadata if r['task'] == task],
                                               quota, SEED, 'consistency/' + task))
    base = [q for q in queries if q['query_id'] in selected]
    variants, mappings = [], []
    for variant in ('candidate_order', 'equivalent_smiles'):
        for query in base:
            row, mapping = transformed(query, variant)
            variants.append(row)
            mappings.append(mapping)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new consistency output directory; never overwrite selections')
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'base_queries.jsonl', base)
    write_rows(output / 'transformed_queries.jsonl', variants)
    write_rows(output / 'mapping.jsonl', mappings)
    info = {'status': 'CANDIDATE_NOT_FROZEN', 'version': 1, 'split': partition,
            'selection_seed': SEED, 'selection_uses_labels': False, 'base_count': len(base),
            'transformed_count': len(variants), 'requested_count': limit,
            'source_queries': str(queries_path), 'split_manifest': str(manifest_path),
            'source_query_sha256': sha256_file(queries_path), 'split_sha256': sha256_file(manifest_path),
            'code_sha256': sha256_file(__file__), 'files': {},
            'note': 'Consistency is exact mapped class/ranking agreement; invalid output counts as inconsistent. '
                    'Equivalent SMILES includes solute and candidate solvents, with chirality/charge/fragments preserved. '
                    'Text retrieval is allowed to change its evidence under notation changes.'}
    for name in ('base_queries.jsonl', 'transformed_queries.jsonl', 'mapping.jsonl'):
        info['files'][name] = sha256_file(output / name)
    write_json(output / 'manifest.json', info)
    return info


def normalized(query, prediction, mapping=None):
    result = parsed_prediction(prediction)
    mapping = mapping or {c['id']: c['id'] for c in query['candidates']}
    if query['task'] == 'pair':
        choice = result.get('choice')
        if not isinstance(choice, str) or choice not in ('A', 'B', 'TIE'):
            return None
        return ('TIE',) if choice == 'TIE' else (mapping[choice],)
    ranking = result.get('ranking')
    expected = {c['id'] for c in query['candidates']}
    if (not isinstance(ranking, list) or not all(isinstance(c, str) for c in ranking)
            or len(ranking) != len(expected) or set(ranking) != expected):
        return None
    return tuple(mapping[c] for c in ranking)


def indexed(rows):
    result = {r['query_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate prediction/query IDs')
    return result


def score(folder, base_predictions, transformed_predictions, output, freeze_manifest=None):
    folder = Path(folder)
    info = json.loads((folder / 'manifest.json').read_text())
    if info['split'] == 'test':
        freeze = json.loads(Path(freeze_manifest).read_text()) if freeze_manifest else {}
        if freeze.get('status') != 'FROZEN' or freeze.get('consistency_manifest_sha256') != sha256_file(folder / 'manifest.json'):
            raise ValueError('Test consistency scoring requires its frozen manifest hash')
    for name, expected in info['files'].items():
        if sha256_file(folder / name) != expected:
            raise ValueError('Consistency input changed: ' + name)
    if sha256_file(info['split_manifest']) != info['split_sha256']:
        raise ValueError('Split manifest changed')
    base, variants, mappings = [indexed(read(folder / name)) for name in
                              ('base_queries.jsonl', 'transformed_queries.jsonl', 'mapping.jsonl')]
    # Verify actual partitions before loading any predictions.
    validate_query_manifest(list(base.values()) + list(variants.values()), info['split_manifest'])
    first, second = indexed(read(base_predictions)), indexed(read(transformed_predictions))
    if set(second) - set(variants):
        raise ValueError('Transformed predictions outside registered consistency cohort')
    rows = []
    for qid, query in variants.items():
        mapping = mappings[qid]
        original = base[mapping['original_query_id']]
        a = normalized(original, first.get(original['query_id'], {}))
        b = normalized(query, second.get(qid, {}), mapping['candidate_to_original'])
        valid = a is not None and b is not None
        rows.append({'query_id': qid, 'original_query_id': original['query_id'],
                     'variant': mapping['variant'], 'solute_id': query['solute_id'],
                     'condition_id': query.get('condition_id', original['query_id']), 'task': query['task'],
                     'both_valid': float(valid), 'exact_agreement': float(valid and a == b),
                     'top1_agreement': float(valid and a[0] == b[0]),
                     'any_smiles_changed': float(mapping['changed_smiles'] > 0),
                     'changed_smiles_fraction': mapping['changed_smiles'] / mapping['smiles_fields']})
    details = pd.DataFrame(rows)
    keys = ['variant', 'task', 'solute_id', 'condition_id']
    columns = ['both_valid', 'exact_agreement', 'top1_agreement', 'any_smiles_changed', 'changed_smiles_fraction']
    solutes = details.groupby(keys)[columns].mean().groupby(keys[:-1]).mean().reset_index()
    result = {'partition': info['split'], 'queries': len(details), 'base_queries': len(base), 'by_variant_task': {},
              'inputs': {str(p): sha256_file(p) for p in [folder / 'manifest.json', base_predictions, transformed_predictions]},
              'code_sha256': sha256_file(__file__)}
    for (variant, task), group in solutes.groupby(['variant', 'task']):
        result['by_variant_task'][variant + '/' + task] = {'solutes': len(group), **group[columns].mean().to_dict()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    details.to_parquet(output / 'per_query.parquet', index=False)
    solutes.to_csv(output / 'per_solute.csv', index=False)
    write_json(output / 'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'score'))
    parser.add_argument('--queries')
    parser.add_argument('--manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=512)
    parser.add_argument('--folder')
    parser.add_argument('--base-predictions')
    parser.add_argument('--transformed-predictions')
    parser.add_argument('--freeze-manifest')
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.queries, args.manifest, args.output, args.limit)
    else:
        result = score(args.folder, args.base_predictions, args.transformed_predictions, args.output, args.freeze_manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
