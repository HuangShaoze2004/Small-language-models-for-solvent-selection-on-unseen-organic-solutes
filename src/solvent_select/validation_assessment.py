"""Assess the registered validation study without selecting methods or releasing tests.

Incomplete studies produce progress snapshots. A recorded decision requires all
registered validation evaluations and queue receipts, irrespective of score.
This is an arithmetic assessment, not a claim of publication readiness or an
independent pre-freeze review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .compare import compare, load_evaluations
from .freeze import queue_evidence
from .report import _query_context, _run_identity
from .study import DEFAULT_REGISTRY, evidence_name, load_registry, property_names
from .views import sha256_file, write_json


def registered_runs(registry, analysis):
    rows = []
    assigned_llm = []
    modes = analysis['direct_evidence_modes']
    if len(modes) != 5 or set(modes) != {'text', 'structure', 'condition', 'no_temperature', 'no_solvent'}:
        raise ValueError('Every registered direct-evidence mode must be retained')
    for protocol, config in registry['protocols'].items():
        names = property_names(protocol, config)
        if names != analysis['property_methods'][protocol]:
            raise ValueError('Registered property methods and protocol namespace disagree')
        names += [evidence_name(protocol, config, mode) for mode in modes]
        llm = [name for name in analysis['llm_validation_tasks_to_carry_to_test']
               if name.startswith(config['run_prefix'] + protocol + '_')]
        names += llm
        assigned_llm += llm
        for name in names:
            rows.append((protocol, name, Path(config['runs_root']) / name / 'evaluation_view10000'))
    if len({name for _, name, _ in rows}) != len(rows):
        raise ValueError('Duplicate registered method IDs')
    if sorted(assigned_llm) != sorted(analysis['llm_validation_tasks_to_carry_to_test']):
        raise ValueError('A registered LM method was not assigned to exactly one protocol')
    return rows


def joint_criteria(comparisons, primary_contrast, seeds):
    """Apply every preregistered criterion; failure is not proof of no effect."""
    def select(task, metric):
        found = [row for row in comparisons if row['task'] == task and row['metric'] == metric]
        if len(found) != 1:
            raise ValueError('Missing or repeated registered comparison metric')
        return found[0]
    pair, regret = select('pair', 'pair_accuracy'), select('rank', 'regret')
    for row in [pair, regret]:
        values = [row['difference_first_minus_second'], *row['ci95'], *row['individual_seed_differences']]
        if (len(row['ci95']) != 2 or row['ci95'][0] > row['ci95'][1]
                or len(row['individual_seed_differences']) != len(seeds)
                or not all(np.isfinite(value) for value in values)):
            raise ValueError('Incomplete/nonfinite comparison estimates')
    checks = {
        'pair_minimum_effect': pair['difference_first_minus_second'] >= primary_contrast['minimum_pair_accuracy_gain'],
        'pair_ci_beneficial': pair['ci95'][0] > 0,
        'rank_regret_minimum_effect': regret['difference_first_minus_second'] <= primary_contrast['maximum_rank_regret_difference_dex'],
        'rank_regret_ci_beneficial': regret['ci95'][1] < 0,
        'all_seed_pair_directions_beneficial': all(v > 0 for v in pair['individual_seed_differences']),
        'all_seed_rank_regret_directions_beneficial': all(v < 0 for v in regret['individual_seed_differences']),
    }
    return {'criteria': {key: bool(value) for key, value in checks.items()},
            'joint_criteria_supported': bool(all(checks.values())),
            'interpretation': ('SUPPORTED_ON_VALIDATION' if all(checks.values())
                               else 'NOT_SUPPORTED_BY_PREREGISTERED_JOINT_CRITERIA')}


def inspect_evaluation(folder, protocol, config, evidence):
    """Check public partition and numerical provenance before interpreting scores."""
    required = [folder / name for name in ['metrics.json', 'per_query.parquet', 'per_solute.csv']]
    if not all(path.exists() for path in required):
        return None
    metric = json.loads(required[0].read_text())
    cohort, family, partition = _query_context(metric, str(folder))
    expected_query = Path(config['views']) / 'val_queries.jsonl'
    if (partition != 'val' or family != protocol or cohort != sha256_file(expected_query)):
        raise ValueError('Only the exact registered validation cohort may be assessed: ' + str(folder))
    expected_target = Path(config['views']) / 'val_targets.jsonl'
    inputs = {str(Path(path).resolve()): digest for path, digest in metric.get('inputs', {}).items()}
    for path in [expected_query, expected_target]:
        if inputs.get(str(path.resolve())) != sha256_file(path):
            raise ValueError('Missing or changed validation input provenance: ' + str(path))
    for path, expected in metric['inputs'].items():
        if sha256_file(path) != expected:
            raise ValueError('Evaluation input changed: ' + path)
        evidence[str(path)] = expected
    tables, _, identities = load_evaluations([folder])
    metadata, metadata_path, _, _ = _run_identity(folder.parent, metric)
    if metadata.get('status') != 'completed' or metadata_path is None:
        raise ValueError('Validation inference/model is not completed: ' + str(folder))
    frame = tables[0]
    for task, values in metric['by_task'].items():
        group = frame[frame.task == task]
        if len(group) != values['solutes']:
            raise ValueError('Per-solute support disagrees with metrics')
        for name, reported in values.items():
            if name not in frame.columns or name in ['solute_id', 'task']:
                continue
            mean = group[name].mean()
            if (reported is None) != bool(np.isnan(mean)) or (
                    reported is not None and not np.isclose(reported, mean, rtol=1e-9, atol=1e-12)):
                raise ValueError('Per-solute numerical source disagrees with metrics: ' + name)
    for path in required + [metadata_path]:
        evidence[str(path)] = sha256_file(path)
    return {'status': 'VERIFIED_COMPLETED_VALIDATION', 'method': identities[0]['method'],
            'seed': identities[0]['seed'], 'cohort_sha256': cohort,
            'queries': metric['queries'], 'by_task': metric['by_task']}


def assess(output, registry_path=DEFAULT_REGISTRY, record_decision=False):
    registry = load_registry(registry_path)
    analysis_path = Path(registry['analysis_plan'])
    analysis = json.loads(analysis_path.read_text())
    if (analysis.get('status') != 'REGISTERED_BEFORE_TEST'
            or analysis.get('registry_sha256') != sha256_file(registry_path)
            or analysis.get('primary_protocol') != 'identity'
            or analysis.get('seeds') != [17, 42, 2026]):
        raise ValueError('Unrecognized or changed registered analysis')
    # The comparison implementation has a fixed, preregistered bootstrap. Refuse
    # a different specification instead of silently evaluating another rule.
    if (analysis['bootstrap']['repetitions'] != 10000 or analysis['bootstrap']['seed'] != 20261025
            or analysis['bootstrap']['unit'] != 'solute' or analysis['bootstrap']['paired'] is not True):
        raise ValueError('Unsupported bootstrap specification')
    decision = Path(registry['validation_decision'])
    output = Path(output)
    if output.exists() or (record_decision and decision.exists()):
        raise FileExistsError('Preserve existing assessments and decisions; choose a new snapshot path')
    evidence = {str(registry_path): sha256_file(registry_path), str(analysis_path): sha256_file(analysis_path),
                str(Path(__file__)): sha256_file(__file__)}
    audits = [queue_evidence(stage['plan'], stage['state']) for stage in registry['queues']]
    errors = [error for audit in audits for error in audit['errors']]
    if errors:
        raise ValueError('Validation receipts no longer verify: ' + repr(errors))
    rows, completed = [], set()
    for protocol, name, folder in registered_runs(registry, analysis):
        result = inspect_evaluation(folder, protocol, registry['protocols'][protocol], evidence)
        rows.append({'protocol': protocol, 'run': name, **(result or {'status': 'PENDING_REGISTERED_VALIDATION'})})
        if result:
            completed.add(name)
    missing = [row['run'] for row in rows if row['run'] not in completed]
    pending_tasks = sum(len(audit['pending']) for audit in audits)
    if record_decision and (missing or pending_tasks):
        raise ValueError('A validation decision requires every registered method and queue receipt; '
                         f'{len(missing)} evaluations and {pending_tasks} tasks remain')
    output.mkdir(parents=True)
    contrasts = {}
    for protocol, config in registry['protocols'].items():
        first = [f"{config['run_prefix']}{protocol}_qwen_rag_{seed}_condition_val" for seed in analysis['seeds']]
        second = [f"{config['run_prefix']}{protocol}_qwen_rag_{seed}_structure_val" for seed in analysis['seeds']]
        if not set(first + second) <= completed:
            contrasts[protocol] = {'status': 'PENDING_ALL_THREE_MATCHED_SEEDS',
                                   'missing_runs': sorted(set(first + second) - completed)}
            continue
        path = output / f'{protocol}_condition_minus_structure.json'
        root = Path(config['runs_root'])
        comparison = compare([root / name / 'evaluation_view10000' for name in first],
                             [root / name / 'evaluation_view10000' for name in second], path)
        if comparison['status'] != 'VALIDATION_ONLY' or not comparison['is_primary_comparison']:
            raise ValueError('Expected matched-adapter retrieval comparison on validation only')
        judgment = joint_criteria(comparison['comparisons'], analysis['primary_contrast'], analysis['seeds'])
        if judgment['joint_criteria_supported'] != comparison['hypothesis_joint_criteria_supported']:
            raise ValueError('Comparison and registered decision criteria disagree')
        evidence[str(path)] = sha256_file(path)
        evidence.update(comparison['inputs'])
        contrasts[protocol] = {'status': 'COMPLETED_VALIDATION_CONTRAST',
                              'scope': 'PRIMARY' if protocol == analysis['primary_protocol'] else 'STRICT_EXTRAPOLATION_CHECK',
                              **judgment, 'comparison': str(path)}
    for stage in registry['queues']:
        for path in [stage['plan'], stage['state']]:
            # Live queue state is intentionally excluded from a durable
            # progress snapshot's hash claims; terminal decisions pin it.
            if Path(path).exists() and (record_decision or path == stage['plan']):
                evidence[str(path)] = sha256_file(path)
    result = {'status': 'VALIDATION_DECISION_RECORDED' if record_decision else 'VALIDATION_PROGRESS_ONLY',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'test_used': False,
              'selected_methods': 'ALL_REGISTERED', 'registered_methods': len(rows),
              'completed_methods': len(completed), 'pending_methods': missing,
              'pending_queue_tasks': pending_tasks, 'runs': rows, 'contrasts': contrasts,
              'evidence_sha256': evidence,
              'scope': 'Validation arithmetic and provenance only. An unsupported joint criterion does not prove zero effect. '
                       'No score-dependent method exclusion, test release, pre-freeze review, manuscript writing or publication-readiness claim.'}
    summary = output / 'assessment.json'
    write_json(summary, result)
    if record_decision:
        result['assessment_path'] = str(summary)
        result['evidence_sha256'] = {**evidence, str(summary): sha256_file(summary)}
        decision.parent.mkdir(parents=True, exist_ok=True)
        write_json(decision, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--registry', default=DEFAULT_REGISTRY)
    parser.add_argument('--record-decision', action='store_true')
    args = parser.parse_args()
    result = assess(args.output, args.registry, args.record_decision)
    print(json.dumps({key: result[key] for key in ['status', 'registered_methods', 'completed_methods',
                                                 'pending_methods', 'pending_queue_tasks', 'contrasts']}, indent=2))


if __name__ == '__main__':
    main()
