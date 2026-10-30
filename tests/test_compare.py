import json
from pathlib import Path

import pandas as pd
import pytest
import solvent_select.compare as comparison

from solvent_select.compare import compare, load_evaluations
from solvent_select.views import sha256_file


def make(tmp_path, name, seed=17, qid='q', partition='val', values=(1., 0.)):
    run = tmp_path / name
    folder = run / 'evaluation'
    folder.mkdir(parents=True)
    (run / 'metadata.json').write_text(json.dumps({'method': name.split('_')[0], 'seed': seed, 'status': 'completed'}))
    source = tmp_path / f'data/views/identity/{partition}_queries.jsonl'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_text(''.join(json.dumps({'query_id': name, 'solute_id': solute, 'task': task, 'split': partition}) + '\n'
                                  for name, solute, task in [(qid, 's1', 'pair'), (qid + '2', 's2', 'rank')]))
    manifest = tmp_path / 'data/splits/identity.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('solute_id,split\ns1,' + partition + '\ns2,' + partition + '\n')
    (folder / 'metrics.json').write_text(json.dumps({'inputs': {str(source): sha256_file(source)}}))
    pd.DataFrame({'query_id': [qid, qid + '2'], 'solute_id': ['s1', 's2'], 'task': ['pair', 'rank']}).to_parquet(folder / 'per_query.parquet')
    pd.DataFrame({'solute_id': ['s1', 's2'], 'task': ['pair', 'rank'], 'pair_accuracy': [values[0], None],
                  'top1': [None, values[1]], 'regret': [.2, .3]}).to_csv(folder / 'per_solute.csv', index=False)
    return folder


def test_methods_with_same_solutes_but_different_questions_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    a = make(tmp_path, 'a', qid='q')
    b = make(tmp_path, 'b', qid='other')
    with pytest.raises(ValueError, match='identical query'):
        load_evaluations([a, b])


def test_seed_order_is_checked_and_descriptive_comparison_has_no_primary_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    first = [make(tmp_path, f'a_{seed}', seed) for seed in [17, 42, 2026]]
    second = [make(tmp_path, f'b_{seed}', seed) for seed in [17, 42, 2026]]
    result = compare(first, second, tmp_path / 'result.json')
    assert result['hypothesis_joint_criteria_supported'] is None
    assert all(row['ci95'] == [0., 0.] for row in result['comparisons'])
    with pytest.raises(ValueError, match='matching order'):
        compare(first, second[::-1], tmp_path / 'bad.json')


def test_test_gate_precedes_outcome_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    folder = make(tmp_path, 'a', partition='test')
    (folder / 'per_solute.csv').unlink()
    with pytest.raises(ValueError, match='frozen query hash'):
        load_evaluations([folder])


def test_actual_partition_is_verified_before_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    folder = make(tmp_path, 'a')
    (tmp_path / 'data/splits/identity.csv').write_text('solute_id,split\ns1,test\ns2,test\n')
    (folder / 'per_solute.csv').unlink()
    with pytest.raises(ValueError, match='partition does not agree'):
        load_evaluations([folder])


def test_sourceqa_comparison_uses_versioned_manifest_before_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    folder = make(tmp_path, 'a')
    old = tmp_path / 'data/views/identity/val_queries.jsonl'
    tagged = tmp_path / 'data/views/sourceqa/identity/val_queries.jsonl'
    tagged.parent.mkdir(parents=True)
    tagged.write_bytes(old.read_bytes())
    metrics_path = folder / 'metrics.json'
    metrics_path.write_text(json.dumps({'inputs': {str(tagged): sha256_file(tagged)}}))
    manifest = tmp_path / 'data/splits/sourceqa/identity.csv'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('solute_id,split\ns1,test\ns2,test\n')
    (folder / 'per_solute.csv').unlink()
    # The old initial manifest still says val: choosing it would pass the
    # partition check and reach the deliberately absent outcome table.
    with pytest.raises(ValueError, match='partition does not agree'):
        load_evaluations([folder])


def test_primary_comparison_requires_identical_adapter_within_each_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'ROOT', tmp_path)
    first = [make(tmp_path, f'a_{seed}', seed) for seed in [17, 42, 2026]]
    second = [make(tmp_path, f'b_{seed}', seed) for seed in [17, 42, 2026]]
    for folders, mode in [(first, 'condition'), (second, 'structure')]:
        for folder, seed in zip(folders, [17, 42, 2026]):
            (folder.parent / 'metadata.json').write_text(json.dumps({
                'status': 'completed', 'seed': seed, 'method': f'llm:rag_sft:{mode}',
                'fingerprint': {'adapter_sha256': f'fixed-adapter-seed-{seed}'}}))
    result = compare(first, second, tmp_path / 'same_adapters.json')
    assert result['is_primary_comparison'] is True
    assert result['hypothesis_joint_criteria_supported'] is False
    path = second[1].parent / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata['fingerprint']['adapter_sha256'] = 'a-different-adapter'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='identical adapter per seed'):
        compare(first, second, tmp_path / 'wrong_adapter.json')
    assert not (tmp_path / 'wrong_adapter.json').exists()
