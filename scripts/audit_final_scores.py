#!/usr/bin/env python3
"""Audit completed frozen-test scores with the independently tested grader.

No model calls or production parser/evaluator imports. The frozen task graph
defines the complete cohort and methods; incomplete receipts are rejected.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    'independent_grader', Path(__file__).with_name('audit_validation_scores.py'))
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def option(command, key):
    return command[command.index(key) + 1]


def audit_method(task, state, sources):
    command = task['command']
    paths = {key: Path(option(command, '--' + key))
             for key in ['queries', 'targets', 'predictions', 'output', 'freeze-manifest']}
    release = json.loads(paths['freeze-manifest'].read_text())
    if (str(paths['queries']) != release['query_path']
            or str(paths['targets']) != release['target_path']):
        raise ValueError('Evaluation does not match frozen cohort')
    for dep in task['depends_on']:
        receipt = state['tasks'][dep]
        if receipt['status'] != 'completed':
            raise ValueError('Prediction dependency is incomplete')
        for name, expected in receipt['output_sha256'].items():
            if grader.sha256_file(name) != expected:
                raise ValueError('Prediction receipt hash changed: ' + name)
    split = Path(release['split_path'])
    measured_path = Path(release['measurement_path'])
    roles = pd.read_csv(split, dtype={'solute_id': str}).set_index('solute_id').split
    if roles.index.has_duplicates:
        raise ValueError('Repeated manifest identity')
    queries = grader.read(paths['queries'])
    qids = {q['query_id'] for q in queries}
    if (len(qids) != len(queries) or not queries
            or any(q.get('split') != 'test' or roles.get(q['solute_id']) != 'test' for q in queries)):
        raise ValueError('Wrong frozen test membership')
    measured = pd.read_parquet(measured_path, columns=['record_id', 'logx'],
                               filters=[('solute_id', 'in', list(roles[roles == 'test'].index))])
    if measured.record_id.duplicated().any() or not all(math.isfinite(x) for x in measured.logx):
        raise ValueError('Invalid test measurement cohort')
    native = measured.set_index('record_id').logx.to_dict()
    rows = grader.read(paths['targets'])
    targets = {row['query_id']: row for row in rows}
    if len(targets) != len(rows) or set(targets) != qids:
        raise ValueError('Test target membership differs')
    for q in queries:
        target = targets[q['query_id']]
        if target['values'] != {c['id']: native[c['record_id']] for c in q['candidates']}:
            raise ValueError('Targets do not match measured values')
        values = target['values']
        if q['task'] == 'pair' and target['choice'] != grader.choice(values):
            raise ValueError('Incorrect pair target')
        if q['task'] == 'rank' and target['ranking'] != sorted(values, key=lambda cid: (-values[cid], cid)):
            raise ValueError('Incorrect rank target')
    prediction_path, evaluation = paths['predictions'], paths['output']
    metric_path = evaluation / 'metrics.json'
    official = json.loads(metric_path.read_text())
    expected = {str(p.resolve()): grader.sha256_file(p)
                for p in [paths['queries'], paths['targets'], prediction_path]}
    if {str(Path(p).resolve()): h for p, h in official['inputs'].items()} != expected:
        raise ValueError('Evaluation numerical inputs changed')
    answers = {}
    is_property = '--property-predictions' in command
    if is_property:
        predictions = pd.read_parquet(prediction_path)
        if predictions.record_id.duplicated().any() or set(predictions.record_id) != set(native):
            raise ValueError('Incomplete property prediction cohort')
        values = predictions.set_index('record_id').pred_logx.to_dict()
        for q in queries:
            prediction = {c['id']: values[c['record_id']] for c in q['candidates']}
            if all(math.isfinite(float(x)) for x in prediction.values()):
                answers[q['query_id']] = ({'choice': grader.choice(prediction)} if q['task'] == 'pair'
                    else {'ranking': sorted(prediction, key=lambda cid: (-prediction[cid], cid))})
    else:
        predictions = grader.read(prediction_path)
        raw = {row['query_id']: row for row in predictions}
        if len(raw) != len(predictions) or set(raw) != qids:
            raise ValueError('Incomplete or repeated raw prediction cohort')
        for q in queries:
            row = raw[q['query_id']]
            answer = grader.decode(row['raw_output'], q)
            if answer != row['prediction'] or ((row['parse_error'] is None) != (answer is not None)):
                raise ValueError('Independent parsing differs')
            answers[q['query_id']] = answer
    computed = [grader.utility(q, targets[q['query_id']]['values'], answers.get(q['query_id'])) for q in queries]
    saved = pd.read_parquet(evaluation / 'per_query.parquet').set_index('query_id')
    if saved.index.has_duplicates or set(saved.index) != qids:
        raise ValueError('Wrong score cohort')
    largest = 0.
    for q, row in zip(queries, computed):
        actual = saved.loc[q['query_id']]
        if any(actual[k] != q[k] for k in ['solute_id', 'condition_id', 'task']):
            raise ValueError('Wrong score identities')
        for key, value in row.items():
            if not grader.same(actual[key], value):
                raise ValueError('Independent per-query score mismatch: ' + key)
            if math.isfinite(value):
                largest = max(largest, abs(float(actual[key]) - value))
    saved_solutes = pd.read_csv(evaluation / 'per_solute.csv', dtype={'solute_id': str}).set_index(['solute_id', 'task'])
    macro = grader.check_solute_scores(queries, computed, saved_solutes)
    if set(official['by_task']) != {q['task'] for q in queries}:
        raise ValueError('Wrong task summary cohort')
    for task_name, entry in official['by_task'].items():
        if (entry['queries'] != sum(q['task'] == task_name for q in queries)
                or entry['solutes'] != len({q['solute_id'] for q in queries if q['task'] == task_name})):
            raise ValueError('Incorrect task denominator')
        for key in grader.METRICS:
            if not grader.same(entry[key], macro.get((task_name, key))):
                raise ValueError('Independent population macro mismatch: ' + key)
    for path in [*paths.values(), split, measured_path, metric_path,
                 evaluation / 'per_query.parquet', evaluation / 'per_solute.csv']:
        if path.is_file():
            sources[str(path)] = grader.sha256_file(path)
    return {'run': evaluation.parent.name, 'protocol': task['protocol'], 'status': 'PASSED',
            'questions_checked': len(queries), 'invalid_or_missing_outputs': sum(r['covered'] == 0 for r in computed),
            'maximum_per_query_arithmetic_difference': largest,
            'raw_json_and_schema_checked': not is_property,
            'independent_metrics': {task_name: {key: macro.get((task_name, key)) for key in grader.METRICS}
                                    for task_name in ['pair', 'rank']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous audits')
    plan = json.loads(Path(args.plan).read_text())
    state = json.loads(Path(args.state).read_text())
    if grader.sha256_file(args.plan) != state['plan_sha256']:
        raise ValueError('Frozen plan checksum differs')
    tasks = [t for t in plan['tasks'] if t['id'].endswith('_evaluate')]
    if len(tasks) != 62 or len({t['id'] for t in tasks}) != 62:
        raise ValueError('Expected all 62 registered evaluations')
    sources = {str(p): grader.sha256_file(p) for p in [Path(__file__), Path(grader.__file__), Path(args.plan)]}
    results = []
    for task in tasks:
        receipt = state['tasks'][task['id']]
        if receipt['status'] != 'completed':
            raise ValueError('Final method evaluation is incomplete')
        for path, expected in receipt['output_sha256'].items():
            if grader.sha256_file(path) != expected:
                raise ValueError('Evaluation receipt hash changed')
        result = audit_method(task, state, sources)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'independent_metrics'}), flush=True)
    grader.write_json(output, {'status': 'PASSED_ALL_REGISTERED_FINAL_GRADING', 'runs': results,
        'pending_runs': [], 'test_targets_read': True, 'models_invoked': False,
        'evidence_sha256': sources,
        'scope': 'All 62 fixed final methods; raw parsing, measured targets, invalid penalties and every query/solute/population score independently recounted. Not an efficacy or publication decision.'})


if __name__ == '__main__':
    main()
