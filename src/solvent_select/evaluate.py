"""Measured-candidate evaluation, hierarchical macro metrics and paired bootstrap."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .data import pair_label

def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]

def ranking_from_values(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda k: (-values[k], k))

def property_predictions(queries: list[dict], predictions: pd.DataFrame, tie_dex=0.1) -> list[dict]:
    if predictions.record_id.duplicated().any():
        raise ValueError('Duplicate prediction IDs')
    indexed = dict(zip(predictions.record_id, predictions.pred_logx))
    output = []
    for query in queries:
        values = {c['id']: indexed.get(c['record_id'], np.nan) for c in query['candidates']}
        result = {'query_id': query['query_id'], 'predicted_values': values}
        if all(np.isfinite(v) for v in values.values()):
            if query['task'] == 'pair':
                result['choice'] = pair_label(values['A'], values['B'], tie_dex)
            else:
                result['ranking'] = ranking_from_values(values)
        output.append(result)
    return output

def parsed_prediction(pred: dict) -> dict:
    for key in ['parsed', 'parsed_output', 'prediction']:
        if isinstance(pred.get(key), dict):
            return pred[key]
    return pred

def evaluate_queries(queries, targets, predictions, tie_dex=0.1):
    if len({q['query_id'] for q in queries}) != len(queries):
        raise ValueError('Repeated query IDs in query manifest')
    gold = {r['query_id']: r for r in targets}
    guess = {r['query_id']: r for r in predictions}
    if len(gold) != len(targets) or len(guess) != len(predictions):
        raise ValueError('Repeated query IDs in targets/predictions')
    if set(gold) != {q['query_id'] for q in queries}:
        raise ValueError('Queries and target IDs disagree')
    if set(guess) - set(gold):
        raise ValueError('Prediction IDs outside query manifest')
    rows = []
    for q in queries:
        target = gold[q['query_id']]
        pred = parsed_prediction(guess.get(q['query_id'], {}))
        y = target['values']
        candidate_ids = [c['id'] for c in q['candidates']]
        if set(y) != set(candidate_ids):
            raise ValueError('Target and candidate mismatch')
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in y.values()):
            raise ValueError('Nonfinite or nonnumeric experimental value')
        best, worst = max(y.values()), min(y.values())
        row = {'query_id': q['query_id'], 'solute_id': q['solute_id'],
               'condition_id': q.get('condition_id', q['query_id']), 'task': q['task'],
               'pair_accuracy': np.nan, 'nontie_pair_accuracy': np.nan,
               'top1': np.nan, 'rank_tau': np.nan, 'regret': best-worst,
               'covered': 0.0, 'candidate_count': len(candidate_ids)}
        if q['task'] == 'pair':
            choice = pred.get('choice')
            valid = isinstance(choice, str) and choice in {'A', 'B', 'TIE'}
            correct = valid and choice == target['choice']
            row['pair_accuracy'] = float(correct)
            if target['choice'] != 'TIE':
                row['nontie_pair_accuracy'] = float(correct)
            if valid:
                # Tie choice is deterministic by chemistry, invariant to A/B permutation.
                from .retrieval import canonical_smiles
                selected = choice if choice != 'TIE' else min(
                    q['candidates'], key=lambda c: (canonical_smiles(c['smiles']), str(c['id'])))['id']
                row.update(covered=1.0, regret=best-y[selected])
        elif q['task'] == 'rank':
            ranking = pred.get('ranking')
            valid = (isinstance(ranking, list) and all(isinstance(c, str) for c in ranking)
                     and len(ranking) == len(candidate_ids) and set(ranking) == set(candidate_ids))
            row['top1'] = 0.0
            # Invalid rankings receive conservative tau=-1 rather than disappearing.
            row['rank_tau'] = -1.0
            if valid:
                positions = {cid: -i for i, cid in enumerate(ranking)}
                tau = kendalltau([y[c] for c in candidate_ids], [positions[c] for c in candidate_ids]).statistic
                row.update(covered=1.0, top1=float(best-y[ranking[0]] <= tie_dex+1e-12),
                           regret=best-y[ranking[0]], rank_tau=float(tau) if np.isfinite(tau) else np.nan)
        else:
            raise ValueError('Unknown task')
        rows.append(row)
    details = pd.DataFrame(rows)
    columns = ['pair_accuracy', 'nontie_pair_accuracy', 'top1', 'rank_tau', 'regret', 'covered']
    # Task-specific regret/coverage must not be dominated by the number of pair questions.
    per_condition = details.groupby(['solute_id', 'condition_id', 'task'])[columns].mean().reset_index()
    per_solute = per_condition.groupby(['solute_id', 'task'])[columns].mean().reset_index()
    metrics = {'queries': len(details), 'solutes': int(details.solute_id.nunique()),
               'missing_predictions': len(set(gold)-set(guess)), 'by_task': {}}
    for task, group in per_solute.groupby('task'):
        metrics['by_task'][task] = {'solutes': len(group), 'queries': int((details.task == task).sum()),
                                  'tau_defined_queries': int(details.loc[details.task == task, 'rank_tau'].notna().sum()) if task == 'rank' else None,
                                  'tau_defined_solutes': int(group.rank_tau.notna().sum()) if task == 'rank' else None,
                                  **{c: float(group[c].mean()) if group[c].notna().any() else None for c in columns}}
    return metrics, details, per_solute

def paired_bootstrap(first: pd.DataFrame, second: pd.DataFrame, metric: str, task: str,
                     seed: int = 20261025, repetitions: int = 10000) -> dict:
    a = first[first.task == task].groupby('solute_id')[metric].mean()
    b = second[second.task == task].groupby('solute_id')[metric].mean()
    if set(a.index) != set(b.index):
        raise ValueError('Paired methods must contain identical solutes')
    delta = (a-b.reindex(a.index)).dropna().to_numpy()
    if not len(delta):
        raise ValueError('No paired observations')
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        count = min(500, repetitions-start)
        estimates[start:start+count] = delta[rng.integers(len(delta), size=(count, len(delta)))].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {'metric': metric, 'task': task, 'difference_first_minus_second': float(delta.mean()),
            'ci95': [float(low), float(high)], 'solutes': len(delta), 'bootstrap_repetitions': repetitions,
            'bootstrap_seed': seed}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--property-predictions', action='store_true')
    parser.add_argument('--freeze-manifest', type=Path)
    args = parser.parse_args()
    queries = read_jsonl(args.queries)
    if any(q.get('split') == 'test' for q in queries):
        if not args.freeze_manifest or not args.freeze_manifest.exists():
            raise ValueError('Final test evaluation requires a freeze manifest')
        freeze = json.loads(args.freeze_manifest.read_text())
        expected = freeze.get('query_sha256')
        actual = hashlib.sha256(args.queries.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError('Test query hash not authorized by freeze manifest')
    targets = read_jsonl(args.targets)
    predictions = property_predictions(queries, pd.read_parquet(args.predictions)) if args.property_predictions else read_jsonl(args.predictions)
    metrics, details, per_solute = evaluate_queries(queries, targets, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    details.to_parquet(args.output/'per_query.parquet', index=False)
    per_solute.to_csv(args.output/'per_solute.csv', index=False)
    metadata = {**metrics, 'evaluation_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [args.queries, args.targets, args.predictions]}}
    (args.output/'metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
