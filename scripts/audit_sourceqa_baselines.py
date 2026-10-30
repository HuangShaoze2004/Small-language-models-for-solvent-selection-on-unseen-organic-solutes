#!/usr/bin/env python3
"""Audit completed SOURCEQA models and actual evidence provenance, without test labels.

This is a separate arithmetic/provenance check, not an independent human review.
It neither retrains models nor changes their predictions or configurations.
"""
from datetime import datetime, timezone
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from solvent_select.views import sha256_file, write_json
from solvent_select.study import load_registry,evidence_name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/audit/dualqa/completed_baseline_audit_current.json')
    parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args()
    registry=load_registry()
    sources, models, evidence_results = {}, [], []
    pending=[]
    def pin(path):
        path = Path(path)
        sources[str(path)] = sha256_file(path)
        return path
    for protocol,config in registry['protocols'].items():
        data_path = Path(config['measurements'])
        manifest_path = Path(config['split'])
        manifest = pd.read_csv(pin(manifest_path), dtype={'solute_id': str})
        train_ids = set(manifest.loc[manifest.split == 'train', 'solute_id'])
        # Parquet predicate applies before labels are read; no test target values.
        train = pd.read_parquet(pin(data_path), filters=[('solute_id', 'in', sorted(train_ids))])
        metadata = pd.read_parquet(data_path, columns=['record_id', 'solute_id', 'leakage_group', 'scaffold_group'])
        assigned = metadata.merge(manifest[['solute_id', 'split']], on='solute_id', validate='many_to_one')
        assert not assigned.split.isna().any()
        assert assigned.groupby('leakage_group').split.nunique().max() == 1
        if protocol == 'scaffold':
            assert assigned.groupby('scaffold_group').split.nunique().max() == 1
        assert set(train.solute_id) == train_ids
        expected_scalers = {
            'temperature_mean': float(train.temperature_k.mean()),
            'temperature_std': max(float(train.temperature_k.std(ddof=0)), 1e-8),
            'target_mean': float(train.logx.mean()),
            'target_std': max(float(train.logx.std(ddof=0)), 1e-8)}
        expected_medians = None
        for kind, seeds in [('nearest', [17]), ('lgbm', [17, 42, 2026]), ('chemprop', [17, 42, 2026])]:
            for seed in seeds:
                name = f"{config['run_prefix']}{protocol}_nearest" if kind == 'nearest' else f"{config['run_prefix']}{protocol}_{kind}_{seed}"
                run = Path(config['runs_root']) / name
                if not (run/'metadata.json').exists() or json.loads((run/'metadata.json').read_text()).get('status')!='completed':
                    if not args.completed_only:raise ValueError('Registered property model incomplete: '+name)
                    pending.append(name);continue
                record = json.loads(pin(run / 'metadata.json').read_text())
                assert record['status'] == 'completed'
                assert record['measurement_sha256'] == sha256_file(data_path)
                assert record['split_sha256'] == sha256_file(manifest_path)
                assert record['train_records'] == len(train)
                if 'train_solutes' in record:
                    assert record['train_solutes'] == len(train_ids)
                assert record['target'] == 'log10_mole_fraction'
                implementation = pin('src/solvent_select/chemprop_runner.py' if kind == 'chemprop' else 'src/solvent_select/baselines.py')
                assert record['code_sha256'] == sha256_file(implementation)
                control = Path(config['consistency_runs_root']) / protocol / name / 'run.json'
                saved = json.loads(pin(control).read_text())
                assert saved['status'] == 'completed'
                if kind == 'chemprop':
                    checkpoint = pin(record['best_checkpoint'])
                    assert sha256_file(checkpoint) == saved['checkpoint_sha256']
                    scalers_path = pin(run / 'scalers.json')
                    assert sha256_file(scalers_path) == saved['scalers_sha256']
                    actual = json.loads(scalers_path.read_text())
                    assert actual.keys() == expected_scalers.keys()
                    assert all(np.isclose(actual[k], expected_scalers[k], rtol=1e-12, atol=1e-12) for k in actual)
                    check = 'Weights match actual consistency run; all temperature/target scalers independently recomputed on train only.'
                else:
                    weights = pin(run / 'model.joblib')
                    assert sha256_file(weights) == saved['model_sha256']
                    bundle = joblib.load(weights)
                    if kind == 'nearest':
                        stored = bundle['model'].train
                        a = stored.sort_values('record_id').reset_index(drop=True)
                        b = train.sort_values('record_id').reset_index(drop=True)
                        for column in ['record_id', 'solute_id', 'solute_smiles', 'solvent_smiles', 'temperature_k', 'logx']:
                            assert a[column].equals(b[column])
                        check = 'Every stored neighbor row and label equals the current training partition.'
                    else:
                        features = bundle['features']
                        if expected_medians is None:
                            # Deterministic descriptors may be cached for held-out
                            # structures; the fitted imputation population must
                            # still be exactly training rows, with their weights.
                            rows = np.asarray([np.concatenate((features._descriptor_cache[s], features._descriptor_cache[v], [t]))
                                for s, v, t in train[['solute_smiles', 'solvent_smiles', 'temperature_k']].itertuples(index=False, name=None)], dtype=np.float32)
                            expected_medians = np.zeros(rows.shape[1], dtype=np.float32)
                            for j in range(rows.shape[1]):
                                values = rows[np.isfinite(rows[:, j]), j]
                                if len(values):
                                    expected_medians[j] = np.median(values)
                        assert np.array_equal(features.medians, expected_medians)
                        check = 'All descriptor-imputation medians recomputed with training-row weights; caches are deterministic structural features, not labels.'
                models.append({'run': name, 'status': 'PASSED', 'check': check,
                               'training_records_recounted': len(train), 'training_solutes_recounted': len(train_ids),
                               'training_solute_count_present_in_model_metadata': 'train_solutes' in record})
        queries_path = pin(Path(config['views'])/'val_queries.jsonl')
        queries = {q['query_id']: q for q in map(json.loads, queries_path.read_text().splitlines())}
        query_groups = metadata.drop_duplicates('solute_id').set_index('solute_id').leakage_group.to_dict()
        training_records = set(train.record_id)
        for mode in ['text', 'structure', 'condition', 'no_temperature', 'no_solvent']:
            path = pin(Path(config['runs_root'])/evidence_name(protocol,config,mode)/'predictions.jsonl')
            prediction = [json.loads(line) for line in path.read_text().splitlines()]
            assert {r['query_id'] for r in prediction} == set(queries)
            cards_seen = 0
            for row in prediction:
                query = queries[row['query_id']]
                cards = row['evidence']
                assert len(cards) <= 4
                assert len({c['solute_id'] for c in cards}) == len(cards)
                assert row['input_tokens'] <= 3072
                for card in cards:
                    assert card['solute_id'] in train_ids
                    assert card['leakage_group'] != query_groups[query['solute_id']]
                    assert set(card['record_ids']) <= training_records
                    assert len(card['record_ids']) <= 8
                cards_seen += len(cards)
            evidence_results.append({'protocol': protocol, 'mode': mode, 'queries': len(prediction),
                                     'cards_checked': cards_seen, 'status': 'PASSED'})
    output = args.output
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    write_json(output, {'status': 'IN_PROGRESS_AUDIT_PASSED_COMPLETED_SUBSET' if pending else 'PASSED_COMPLETED_BASELINE_AUDIT', 'created_utc': datetime.now(timezone.utc).isoformat(),
                       'models': models, 'direct_evidence': evidence_results, 'evidence_sha256': sources,
                       'pending_models':pending,
                       'test_target_values_read': False, 'code_sha256': sha256_file(__file__),
                       'scope': 'Registered dualqa models and actual validation evidence outputs. Pending models remain explicit. Checks weights, train membership, fitted statistics and evidence provenance; not universal label verification or final paper/claim review.'})
    print(json.dumps({'output': output, 'models': len(models), 'evidence_modes': len(evidence_results)}))


if __name__ == '__main__':
    main()
