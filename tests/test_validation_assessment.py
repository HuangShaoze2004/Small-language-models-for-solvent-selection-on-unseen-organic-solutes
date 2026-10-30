"""Synthetic arithmetic/provenance fixtures; these are not research results."""
import copy
import json
from pathlib import Path

import pytest

import solvent_select.validation_assessment as assessment
from solvent_select.validation_assessment import joint_criteria
from solvent_select.views import sha256_file


SEEDS = [17, 42, 2026]
RULE = {'minimum_pair_accuracy_gain': .02, 'maximum_rank_regret_difference_dex': -.05}


def estimates():
    return [
        {'task': 'pair', 'metric': 'pair_accuracy', 'difference_first_minus_second': .03,
         'ci95': [.01, .05], 'individual_seed_differences': [.02, .03, .04]},
        {'task': 'rank', 'metric': 'regret', 'difference_first_minus_second': -.07,
         'ci95': [-.1, -.02], 'individual_seed_differences': [-.06, -.07, -.08]},
    ]


def test_joint_rule_requires_every_registered_effect_interval_and_seed_direction():
    rows = estimates()
    assert joint_criteria(rows, RULE, SEEDS)['joint_criteria_supported']
    cases = [(0, 'difference_first_minus_second', .019), (1, 'difference_first_minus_second', -.049),
             (0, 'ci95', [0., .05]), (1, 'ci95', [-.1, 0.]),
             (0, 'individual_seed_differences', [.06, .04, -.01]),
             (1, 'individual_seed_differences', [-.1, -.1, 0.])]
    for index, key, value in cases:
        changed = copy.deepcopy(rows)
        changed[index][key] = value
        result = joint_criteria(changed, RULE, SEEDS)
        assert not result['joint_criteria_supported']
        assert result['interpretation'] == 'NOT_SUPPORTED_BY_PREREGISTERED_JOINT_CRITERIA'


def test_joint_rule_accepts_prespecified_minimum_effect_boundary():
    rows = estimates()
    rows[0]['difference_first_minus_second'] = .02
    rows[1]['difference_first_minus_second'] = -.05
    assert joint_criteria(rows, RULE, SEEDS)['joint_criteria_supported']


@pytest.mark.parametrize('replacement', [[.03], [.03, .03, float('nan')]])
def test_partial_or_nonfinite_seed_values_cannot_support_a_conclusion(replacement):
    rows = estimates()
    rows[0]['individual_seed_differences'] = replacement
    with pytest.raises(ValueError, match='Incomplete/nonfinite'):
        joint_criteria(rows, RULE, SEEDS)


def test_metric_cannot_be_replaced_by_a_more_favorable_secondary_metric():
    rows = estimates()
    rows[0]['metric'] = 'top1'
    with pytest.raises(ValueError, match='Missing or repeated'):
        joint_criteria(rows, RULE, SEEDS)


def minimal_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry_path = Path('registry.json')
    registry_path.write_text('{}')
    config = {'run_prefix': 'example_', 'runs_root': 'runs'}
    registry = {'analysis_plan': 'analysis.json', 'protocols': {'identity': config, 'scaffold': config},
                'queues': [{'plan': 'queue.json', 'state': 'state.json'}], 'validation_decision': 'decision.json'}
    plan = {'status': 'REGISTERED_BEFORE_TEST', 'registry_sha256': sha256_file(registry_path),
            'primary_protocol': 'identity', 'seeds': SEEDS, 'primary_contrast': RULE,
            'bootstrap': {'repetitions': 10000, 'seed': 20261025, 'unit': 'solute', 'paired': True}}
    Path('analysis.json').write_text(json.dumps(plan))
    Path('queue.json').write_text('{}')
    monkeypatch.setattr(assessment, 'load_registry', lambda path: registry)
    monkeypatch.setattr(assessment, 'queue_evidence', lambda *args: {'pending': ['one'], 'errors': []})
    monkeypatch.setattr(assessment, 'registered_runs', lambda *args: [('identity', 'pending_method', Path('missing'))])
    monkeypatch.setattr(assessment, 'inspect_evaluation', lambda *args: None)
    return registry_path


def test_incomplete_study_records_only_a_snapshot_and_no_partial_seed_comparison(tmp_path, monkeypatch):
    registry = minimal_registry(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not bootstrap an incomplete seed set')
    monkeypatch.setattr(assessment, 'compare', forbidden)
    result = assessment.assess('snapshot', registry)
    assert result['status'] == 'VALIDATION_PROGRESS_ONLY'
    assert result['completed_methods'] == 0
    assert result['pending_methods'] == ['pending_method']
    assert result['test_used'] is False
    assert not Path('decision.json').exists()
    assert all(row['status'] == 'PENDING_ALL_THREE_MATCHED_SEEDS' for row in result['contrasts'].values())
    with pytest.raises(FileExistsError):
        assessment.assess('snapshot', registry)


def test_incomplete_study_cannot_record_a_pretest_decision(tmp_path, monkeypatch):
    registry = minimal_registry(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='every registered method'):
        assessment.assess('snapshot', registry, record_decision=True)
    assert not Path('snapshot').exists()
    assert not Path('decision.json').exists()


def test_changed_receipt_fails_before_any_scores_are_inspected(tmp_path, monkeypatch):
    registry = minimal_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(assessment, 'queue_evidence', lambda *args: {'pending': [], 'errors': ['Changed output']})
    def forbidden(*args, **kwargs):
        raise AssertionError('Changed receipts must stop assessment before reading scores')
    monkeypatch.setattr(assessment, 'inspect_evaluation', forbidden)
    with pytest.raises(ValueError, match='no longer verify'):
        assessment.assess('snapshot', registry)
    assert not Path('snapshot').exists()


def test_non_validation_cohort_is_rejected_before_numerical_tables(tmp_path, monkeypatch):
    folder = tmp_path / 'evaluation'
    folder.mkdir()
    for name in ['metrics.json', 'per_query.parquet', 'per_solute.csv']:
        (folder / name).write_text('{}')
    query = tmp_path / 'val_queries.jsonl'
    query.write_text('public validation identity')
    monkeypatch.setattr(assessment, '_query_context', lambda *args: (sha256_file(query), 'identity', 'test'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Test outcomes must not be loaded')
    monkeypatch.setattr(assessment, 'load_evaluations', forbidden)
    with pytest.raises(ValueError, match='exact registered validation'):
        assessment.inspect_evaluation(folder, 'identity', {'views': tmp_path}, {})


def test_all_registered_methods_must_map_to_the_exact_protocol_namespace():
    registry = {'protocols': {p: {'run_prefix': p + '_data_', 'runs_root': p + '_runs'}
                              for p in ['identity', 'scaffold']}}
    plan = {'property_methods': {p: assessment.property_names(p, c) for p, c in registry['protocols'].items()},
            'direct_evidence_modes': ['text', 'structure', 'condition', 'no_temperature', 'no_solvent'],
            'llm_validation_tasks_to_carry_to_test': ['identity_data_identity_qwen_base_17_none_val',
                                                    'scaffold_data_scaffold_qwen_base_17_none_val']}
    assert len(assessment.registered_runs(registry, plan)) == 26
    unknown = copy.deepcopy(plan)
    unknown['llm_validation_tasks_to_carry_to_test'].append('legacy_scaffold_qwen_base_17_none_val')
    with pytest.raises(ValueError, match='not assigned to exactly one'):
        assessment.registered_runs(registry, unknown)
    wrong_bank = copy.deepcopy(plan)
    wrong_bank['property_methods']['scaffold'][0] = 'old_scaffold_nearest'
    with pytest.raises(ValueError, match='namespace disagree'):
        assessment.registered_runs(registry, wrong_bank)


def test_complete_decision_retains_both_protocols_and_pins_its_assessment(tmp_path, monkeypatch):
    registry_path = minimal_registry(tmp_path, monkeypatch)
    registry = assessment.load_registry(registry_path)
    Path('state.json').write_text('{}')
    rows = [(protocol, f'example_{protocol}_qwen_rag_{seed}_{mode}_val', Path(f'fixture/{protocol}/{seed}/{mode}'))
            for protocol in ['identity', 'scaffold'] for mode in ['condition', 'structure'] for seed in SEEDS]
    monkeypatch.setattr(assessment, 'registered_runs', lambda *args: rows)
    monkeypatch.setattr(assessment, 'inspect_evaluation', lambda *args: {'status': 'VERIFIED_COMPLETED_VALIDATION'})
    monkeypatch.setattr(assessment, 'queue_evidence', lambda *args: {'pending': [], 'errors': []})
    def fixture_comparison(first, second, output):
        assert len(first) == len(second) == 3
        result = {'status': 'VALIDATION_ONLY', 'is_primary_comparison': True, 'comparisons': estimates(),
                  'hypothesis_joint_criteria_supported': True, 'inputs': {}}
        Path(output).write_text(json.dumps(result))
        return result
    monkeypatch.setattr(assessment, 'compare', fixture_comparison)
    result = assessment.assess('snapshot', registry_path, record_decision=True)
    assert result['status'] == 'VALIDATION_DECISION_RECORDED'
    assert result['selected_methods'] == 'ALL_REGISTERED'
    assert result['registered_methods'] == result['completed_methods'] == len(rows)
    assert result['test_used'] is False
    assert result['contrasts']['identity']['scope'] == 'PRIMARY'
    assert result['contrasts']['scaffold']['scope'] == 'STRICT_EXTRAPOLATION_CHECK'
    saved = json.loads(Path(registry['validation_decision']).read_text())
    assert saved['evidence_sha256']['snapshot/assessment.json'] == sha256_file('snapshot/assessment.json')
    for path, expected in saved['evidence_sha256'].items():
        assert sha256_file(path) == expected
