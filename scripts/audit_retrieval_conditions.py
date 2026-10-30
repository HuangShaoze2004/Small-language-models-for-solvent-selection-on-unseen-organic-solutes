#!/usr/bin/env python3
"""Describe actual validation evidence selection using metadata only.

No experimental target column or evaluation score is read. These diagnostics
describe retrieved conditions and coverage, not chemical mechanisms or causal
effects on accuracy. They never alter retrieval weights or model inputs.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from solvent_select.llm import read_jsonl, validate_query_manifest
from solvent_select.retrieval import canonical_smiles
from solvent_select.study import evidence_name, load_registry
from solvent_select.views import sha256_file, write_json

MODES = ['text', 'structure', 'condition', 'no_temperature', 'no_solvent']
MEASURES = ['mean_selected_similarity', 'mean_temperature_gap_k', 'minimum_temperature_gap_k',
            'exact_temperature_card_fraction', 'mean_candidate_coverage', 'union_candidate_coverage',
            'any_card_covers_all_candidates', 'same_ordered_cards_as_condition', 'card_set_jaccard_with_condition']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing metadata diagnostics')
    registry = load_registry()
    sources, rows = {}, []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    @lru_cache(maxsize=8192)
    def fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Invalid retained solute structure')
        return generator.GetFingerprint(mol)
    @lru_cache(maxsize=200000)
    def similarity(first, second):
        return DataStructs.TanimotoSimilarity(fingerprint(first), fingerprint(second))
    for protocol, config in registry['protocols'].items():
        manifest = pd.read_csv(config['split'], dtype={'solute_id': str})
        train_ids = set(manifest.loc[manifest.split == 'train', 'solute_id'])
        # A projection with a training predicate excludes all label columns and
        # all held-out measurement rows before data deserialization.
        train = pd.read_parquet(config['measurements'], filters=[('solute_id', 'in', sorted(train_ids))],
                                columns=['record_id', 'solute_id', 'solute_smiles', 'solvent_smiles',
                                         'temperature_k', 'leakage_group', 'source_doi']).set_index('record_id')
        queries_path = Path(config['views']) / 'val_queries.jsonl'
        queries = read_jsonl(queries_path)
        validate_query_manifest(queries, config['split'])
        if {query['split'] for query in queries} != {'val'}:
            raise ValueError('Only registered validation evidence may be described')
        groups = manifest.set_index('solute_id').leakage_group.to_dict()
        predictions = {}
        for mode in MODES:
            path = Path(config['runs_root']) / evidence_name(protocol, config, mode) / 'predictions.jsonl'
            sidecar = path.with_suffix(path.suffix + '.run.json')
            if json.loads(sidecar.read_text())['status'] != 'completed':
                raise ValueError('Incomplete evidence run: ' + str(path))
            predictions[mode] = {row['query_id']: row['evidence'] for row in read_jsonl(path)}
            if set(predictions[mode]) != {query['query_id'] for query in queries}:
                raise ValueError('Evidence and registered query identities differ')
            sources[str(path)] = sha256_file(path)
            sources[str(sidecar)] = sha256_file(sidecar)
        for path in [config['split'], config['measurements'], queries_path]:
            sources[str(path)] = sha256_file(path)
        @lru_cache(maxsize=50000)
        def card_metadata(records):
            frame = train.loc[list(records)]
            if any(frame[column].nunique() != 1 for column in
                   ['solute_id', 'solute_smiles', 'temperature_k', 'leakage_group', 'source_doi']):
                raise ValueError('A retrieved card mixes solute identities or conditions')
            first = frame.iloc[0]
            return first, {canonical_smiles(s) for s in frame.solvent_smiles}
        for query in queries:
            candidate_smiles = {canonical_smiles(candidate['smiles']) for candidate in query['candidates']}
            reference = [card['card_id'] for card in predictions['condition'][query['query_id']]]
            for mode in MODES:
                cards = predictions[mode][query['query_id']]
                if not cards or len(cards) > 4:
                    raise ValueError('Unexpected empty/over-budget completed evidence bundle')
                gaps, similarities, coverages, union = [], [], [], set()
                selected = [card['card_id'] for card in cards]
                for card in cards:
                    metadata, solvents = card_metadata(tuple(card['record_ids']))
                    if (metadata.solute_id != card['solute_id'] or metadata.solute_id not in train_ids
                            or metadata.leakage_group == groups[query['solute_id']]
                            or metadata.source_doi != card['source_doi']):
                        raise ValueError('Retrieved provenance violates the training boundary')
                    gaps.append(abs(float(metadata.temperature_k) - query['temperature_k']))
                    similarities.append(similarity(query['solute_smiles'], metadata.solute_smiles))
                    coverages.append(len(solvents & candidate_smiles) / len(candidate_smiles))
                    union |= solvents & candidate_smiles
                rows.append({'protocol': protocol, 'mode': mode, 'query_id': query['query_id'],
                             'solute_id': query['solute_id'], 'condition_id': query['condition_id'], 'task': query['task'],
                             'mean_selected_similarity': np.mean(similarities),
                             'mean_temperature_gap_k': np.mean(gaps), 'minimum_temperature_gap_k': min(gaps),
                             'exact_temperature_card_fraction': np.mean(np.asarray(gaps) == 0),
                             'mean_candidate_coverage': np.mean(coverages),
                             'union_candidate_coverage': len(union) / len(candidate_smiles),
                             'any_card_covers_all_candidates': float(max(coverages) == 1),
                             'same_ordered_cards_as_condition': float(selected == reference),
                             'card_set_jaccard_with_condition': len(set(selected) & set(reference)) / len(set(selected) | set(reference))})
    output.mkdir(parents=True)
    details = pd.DataFrame(rows)
    details.to_parquet(output / 'per_query.parquet', index=False)
    per_condition = details.groupby(['protocol', 'mode', 'solute_id', 'condition_id', 'task'])[MEASURES].mean()
    per_solute = per_condition.groupby(['protocol', 'mode', 'solute_id', 'task']).mean().reset_index()
    per_solute.to_csv(output / 'per_solute.csv', index=False)
    summary = per_solute.groupby(['protocol', 'mode', 'task'])[MEASURES].mean()
    summary['solute_count'] = per_solute.groupby(['protocol', 'mode', 'task']).size()
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    write_json(output / 'manifest.json', {'status': 'PASSED_VALIDATION_METADATA_DIAGNOSTICS',
        'sources': sources, 'code_sha256': sha256_file(__file__),
        'measurement_columns_read': ['record_id', 'solute_id', 'solute_smiles', 'solvent_smiles',
                                     'temperature_k', 'leakage_group', 'source_doi'],
        'experimental_target_values_read': False, 'evaluation_scores_read': False,
        'retrieval_parameters_changed': False, 'test_queries_read': False,
        'similarity': 'Morgan radius2/2048bits without chirality, matching the existing retrieval implementation; identities remain full isomeric structures.',
        'aggregation': 'Query mean within condition, then condition mean within solute, then solute macro mean; tasks separate.',
        'scope': 'Descriptive actual selection and candidate coverage only; not a causal explanation of predictive performance or proof that temperature is unimportant in chemistry.',
        'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
