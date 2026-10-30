"""Audit saved SFT evidence against exact training membership and query order.

This reads query identities and measurement metadata, never target values. It
does not replace prompt allowlist/masking tests or declare a training complete.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .llm import read_jsonl, sample_queries, validate_query_manifest
from .retrieval import stable_hash
from .study import DEFAULT_REGISTRY, load_registry, verify_adapter_boundary
from .views import sha256_file, write_json


def audit_rows(queries, rows, train, config):
    if len(queries) != len(rows) or [q['query_id'] for q in queries] != [r['query_id'] for r in rows]:
        raise ValueError('Saved training evidence must match exact sampled query identities and order')
    records = train.set_index('record_id').to_dict('index')
    counter, modes, cards_seen = Counter(), Counter(), 0
    for query, row in zip(queries, rows):
        sid = str(query['solute_id'])
        expected = config['retrieval_mode']
        if expected == 'mixed':
            offset = int(stable_hash(sid)[:8], 16) % 3
            expected = ('text', 'structure', 'condition')[(counter[sid] + offset) % 3]
        counter[sid] += 1
        if row['retrieval_mode'] != expected:
            raise ValueError('Saved retrieval assignment differs from fixed within-solute cycling')
        modes[expected] += 1
        if (row['target_tokens'] <= 0 or row['input_tokens'] + row['target_tokens'] > config['max_sequence_length']
                or not 0 <= row['evidence_tokens'] <= config['max_evidence_tokens']):
            raise ValueError('Saved token audit violates the fixed context budget')
        cards = row['evidence']
        if len(cards) > 4 or len({card['solute_id'] for card in cards}) != len(cards):
            raise ValueError('Saved evidence violates card count or unique-solute budget')
        if (expected == 'none') != (len(cards) == 0):
            raise ValueError('Saved evidence availability disagrees with retrieval mode')
        for card in cards:
            if not 1 <= len(card['record_ids']) <= 8 or len(set(card['record_ids'])) != len(card['record_ids']):
                raise ValueError('Saved card has invalid/repeated measurement identities')
            if card['solute_id'] == sid or card['leakage_group'] == query['leakage_group']:
                raise ValueError('Training query retrieved its own solute or isolation group')
            for record_id in card['record_ids']:
                if record_id not in records:
                    raise ValueError('Saved evidence includes a non-training measurement')
                record = records[record_id]
                if any(record[key] != card[key] for key in ['solute_id', 'leakage_group', 'source_doi']):
                    raise ValueError('Saved evidence provenance disagrees with training metadata')
            cards_seen += 1
    return {'queries_checked': len(rows), 'query_solutes': len(counter),
            'cards_checked': cards_seen, 'retrieval_mode_counts': dict(modes),
            'self_or_same_group_retrievals': 0, 'non_training_records': 0}


def audit_training(run, protocol, output, registry_path=DEFAULT_REGISTRY):
    registry = load_registry(registry_path)
    config = registry['protocols'][protocol]
    run, output = Path(run), Path(output)
    if run.parent.resolve() != Path(config['runs_root']).resolve():
        raise ValueError('Training run is outside the intended data namespace')
    if output.exists():
        raise FileExistsError('Do not overwrite a training evidence audit')
    source = run / 'run.json'
    metadata_bytes = source.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata['status'] not in ['training', 'completed', 'interrupted']:
        raise ValueError('Training preparation is incomplete; saved evidence may still be written')
    verify_adapter_boundary(metadata, config, metadata['config'])
    if metadata.get('sampling_seed') != 20261025:
        raise ValueError('Unregistered training/validation sampling seed')
    manifest = pd.read_csv(config['split'], dtype={'solute_id': str})
    train_ids = set(manifest.loc[manifest.split == 'train', 'solute_id'])
    train = pd.read_parquet(config['measurements'], filters=[('solute_id', 'in', sorted(train_ids))],
                            columns=['record_id', 'solute_id', 'leakage_group', 'source_doi'])
    sources = {str(path): sha256_file(path) for path in [Path(registry_path), Path(config['split']),
               Path(config['measurements']), Path(__file__)]}
    result = {}
    for split, key in [('train', 'max_train_examples'), ('val', 'max_val_examples')]:
        path = Path(config['views']) / f'{split}_queries.jsonl'
        queries = read_jsonl(path)
        validate_query_manifest(queries, config['split'])
        if {q['split'] for q in queries} != {split}:
            raise ValueError('Training evidence queries have an incorrect partition')
        queries = sample_queries(queries, metadata['config'][key], 20261025)
        evidence = run / f'{split}_evidence.jsonl'
        result[split] = audit_rows(queries, read_jsonl(evidence), train, metadata['config'])
        expected_count = metadata['train_examples' if split == 'train' else 'val_examples']
        if result[split]['queries_checked'] != expected_count:
            raise ValueError('Actual saved evidence count differs from training metadata')
        for p in [path, evidence]:
            sources[str(p)] = sha256_file(p)
    output.mkdir(parents=True)
    snapshot = output / 'training_metadata_snapshot.json'
    snapshot.write_bytes(metadata_bytes)
    sources[str(snapshot)] = sha256_file(snapshot)
    record = {'status': 'PASSED_SAVED_TRAINING_EVIDENCE_AUDIT', 'protocol': protocol,
              'run': str(run), 'training_status_at_audit': metadata['status'],
              'training_metadata_source': str(source), 'training_metadata_sha256_at_audit': sha256_file(snapshot),
              'train': result['train'], 'val': result['val'], 'evidence_sha256': sources,
              'measurement_columns_read': ['record_id', 'solute_id', 'leakage_group', 'source_doi'],
              'target_values_read': False, 'test_queries_read': False,
              'scope': 'Actual saved train and teacher-forced validation evidence; full solute-group exclusion, '
                       'training-record membership, fixed mode cycling and reported token budgets. '
                       'Prompt allowlists and assistant-only token masking have separate software tests. '
                       'This audit does not claim a training or scientific experiment has completed.'}
    write_json(output / 'audit.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--protocol', choices=['identity', 'scaffold'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--registry', default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    result = audit_training(args.run, args.protocol, args.output, args.registry)
    print(json.dumps({key: result[key] for key in ['status', 'training_status_at_audit', 'train', 'val']}))


if __name__ == '__main__':
    main()
