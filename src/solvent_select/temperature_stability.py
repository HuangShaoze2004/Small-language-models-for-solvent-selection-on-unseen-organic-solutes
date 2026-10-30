"""Training-only description of measured solvent-order changes with temperature.

No interpolation, held-out labels, method selection or temperature-weight tuning.
All numerical data are loaded through an explicit training-solute predicate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import pair_label
from .report import bootstrap_mean
from .retrieval import stable_hash
from .study import load_registry
from .views import sha256_file, write_json

METRICS = ['label_changed', 'decisive_reversal', 'near_tie_transition', 'both_near_ties',
           'both_decisive', 'raw_sign_reversal', 'absolute_delta_change_dex',
           'delta_change_exceeds_tie_band', 'reversal_given_both_decisive']
DOI = re.compile(r'^10\.\d{4,9}/\S+$')


def read_training(config):
    split = pd.read_csv(config['split'], dtype={'solute_id': str})
    if split.solute_id.duplicated().any():
        raise ValueError('Duplicate manifest solute identities')
    ids = sorted(split.loc[split.split == 'train', 'solute_id'])
    identity = pd.read_parquet(config['measurements'], columns=['solute_id', 'leakage_group']).drop_duplicates()
    assigned = identity.merge(split[['solute_id', 'split']], on='solute_id', validate='one_to_one')
    if len(assigned) != len(identity) or assigned.groupby('leakage_group').split.nunique().max() != 1:
        raise ValueError('Manifest membership or solute-group isolation failed')
    frame = pd.read_parquet(config['measurements'],
        columns=['record_id', 'solute_id', 'source_doi', 'solvent_id', 'temperature_k', 'logx'],
        filters=[('solute_id', 'in', ids)])
    if set(frame.solute_id) != set(ids):
        raise ValueError('Numerical analysis must contain exactly the training solutes')
    return frame


def gap_label(gap, config):
    if gap <= 0 or not np.isfinite(gap):
        raise ValueError('Temperature comparisons require two distinct finite temperatures')
    for edge, label in zip(config['temperature_gap_upper_edges_kelvin'], config['temperature_gap_labels']):
        if gap <= edge + config['gap_float_epsilon']:
            return label
    return config['temperature_gap_labels'][-1]


def transitions(training, config):
    count = Counter(training_records=len(training), training_solutes=training.solute_id.nunique())
    eligible = training.source_doi.map(lambda value: isinstance(value, str) and DOI.fullmatch(value) is not None)
    count['missing_or_malformed_doi_records'] = int((~eligible).sum())
    frame = training[eligible].copy()
    key = ['solute_id', 'source_doi', 'solvent_id', 'temperature_k']
    if frame.duplicated(key).any():
        raise ValueError('Duplicate measured condition; do not silently refit the curation policy')
    if not np.isfinite(frame[['temperature_k', 'logx']].to_numpy()).all() or (frame.temperature_k <= 0).any():
        raise ValueError('Nonfinite labels or invalid Kelvin temperatures')
    rows = []
    for (sid, doi), group in frame.groupby(['solute_id', 'source_doi'], sort=True):
        count['source_groups'] += 1
        solvents = {solvent: {row.temperature_k: row for row in part.itertuples(index=False)}
                    for solvent, part in group.groupby('solvent_id', sort=True)}
        for a, b in combinations(sorted(solvents), 2):
            count['possible_solvent_pairs'] += 1
            common = sorted(solvents[a].keys() & solvents[b].keys())
            if len(common) < 2:
                count['pairs_zero_common_temperature' if not common else 'pairs_one_common_temperature'] += 1
                continue
            count['paired_temperature_series'] += 1
            series = stable_hash(json.dumps([sid, doi, a, b], separators=(',', ':')))[:24]
            points = []
            for t in common:
                first, second = solvents[a][t], solvents[b][t]
                delta = float(first.logx - second.logx)
                points.append((float(t), delta, pair_label(first.logx, second.logx, config['tie_dex']),
                               first.record_id, second.record_id))
            for i, j in combinations(range(len(points)), 2):
                low, high = points[i], points[j]
                both_decisive = low[2] != 'TIE' and high[2] != 'TIE'
                reversal = both_decisive and low[2] != high[2]
                change = abs(high[1] - low[1])
                rows.append({'solute_id': sid, 'source_doi': doi, 'series_id': series,
                    'solvent_a_id': a, 'solvent_b_id': b, 'temperature_low_k': low[0], 'temperature_high_k': high[0],
                    'gap_k': high[0] - low[0], 'gap_bin': gap_label(high[0] - low[0], config),
                    'adjacent_shared_temperatures': j == i + 1,
                    'low_record_a': low[3], 'low_record_b': low[4], 'high_record_a': high[3], 'high_record_b': high[4],
                    'delta_low_dex': low[1], 'delta_high_dex': high[1], 'label_low': low[2], 'label_high': high[2],
                    'label_changed': float(low[2] != high[2]), 'decisive_reversal': float(reversal),
                    'near_tie_transition': float((low[2] == 'TIE') != (high[2] == 'TIE')),
                    'both_near_ties': float(low[2] == high[2] == 'TIE'), 'both_decisive': float(both_decisive),
                    'raw_sign_reversal': float(low[1] * high[1] < 0), 'absolute_delta_change_dex': change,
                    'delta_change_exceeds_tie_band': float(change > config['tie_dex'] + config['tie_float_epsilon']),
                    'reversal_given_both_decisive': float(reversal) if both_decisive else np.nan})
    result = pd.DataFrame(rows)
    count['all_temperature_comparisons'] = len(result)
    count['eligible_solutes'] = result.solute_id.nunique() if len(result) else 0
    count['adjacent_temperature_comparisons'] = int(result.adjacent_shared_temperatures.sum()) if len(result) else 0
    return result, {key: int(value) for key, value in count.items()}


def summarize(frame, protocol, config):
    summaries, populations = [], []
    for relation in config['temperature_relations']:
        population = frame if relation == 'all_observed_temperature_pairs' else (
            frame[frame.adjacent_shared_temperatures] if len(frame) else frame)
        for label in ['ALL', *config['temperature_gap_labels']]:
            subset = population if label == 'ALL' else (population[population.gap_bin == label] if len(population) else population)
            if len(subset):
                series = subset.groupby(['solute_id', 'source_doi', 'series_id'])[METRICS].mean()
                source = series.groupby(['solute_id', 'source_doi']).mean()
                solutes = source.groupby('solute_id').mean()
                exported = solutes.reset_index().assign(protocol=protocol, relation=relation, gap_bin=label)
                populations.append(exported)
            else:
                solutes = pd.DataFrame(columns=METRICS)
            for metric in METRICS:
                values = solutes[metric].dropna()
                low, high = bootstrap_mean(values, config['bootstrap']['repetitions'], config['bootstrap']['seed'])
                summaries.append({'protocol': protocol, 'relation': relation, 'gap_bin': label, 'metric': metric,
                    'value': float(values.mean()) if len(values) else None, 'ci_low': low, 'ci_high': high,
                    'population_solutes': len(solutes), 'support_solutes': len(values), 'raw_temperature_comparisons': len(subset),
                    'source_series': subset.series_id.nunique() if len(subset) else 0,
                    'interpretation_scope': 'DESCRIPTIVE_SUFFICIENT_COHORT' if len(values) >= config['minimum_independent_solutes_for_interpretation'] else 'INSUFFICIENT_OR_EMPTY_COHORT'})
    return pd.DataFrame(summaries), pd.concat(populations, ignore_index=True) if populations else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/temperature_stability_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config.get('status') != 'REGISTERED_TRAINING_ONLY_DESCRIPTIVE_ANALYSIS' or config['allowed_partition'] != 'train'
            or config['tie_dex'] != .1 or config['temperature_gap_upper_edges_kelvin'] != [5., 10., 20.]
            or config['metrics'] != METRICS or sha256_file(config['registry']) != config['registry_sha256']):
        raise ValueError('Unknown/changed training-only analysis registration')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve completed analysis; choose a new output folder')
    output.mkdir(parents=True)
    summaries, populations, audits = [], [], {}
    inputs = {str(p): sha256_file(p) for p in [Path(args.config), Path(config['registry']), Path(__file__),
              Path('src/solvent_select/data.py'), Path('src/solvent_select/report.py')]}
    for protocol, layout in load_registry(config['registry'])['protocols'].items():
        print(protocol, 'reading training labels only', flush=True)
        training = read_training(layout)
        frame, counts = transitions(training, config)
        print(protocol, json.dumps(counts), flush=True)
        frame.to_parquet(output / f'{protocol}_temperature_comparisons.parquet', index=False)
        table, solutes = summarize(frame, protocol, config)
        summaries.append(table)
        populations.append(solutes)
        audits[protocol] = counts
        for p in [layout['measurements'], layout['split']]:
            inputs[p] = sha256_file(p)
        del frame, training
    pd.concat(summaries, ignore_index=True).to_csv(output / 'summary.csv', index=False)
    pd.concat(populations, ignore_index=True).to_csv(output / 'per_solute.csv', index=False)
    write_json(output / 'manifest.json', {'status': 'COMPLETED_TRAINING_ONLY_TEMPERATURE_DESCRIPTION',
        'protocols': audits, 'inputs': inputs, 'heldout_labels_read': False, 'model_predictions_or_scores_read': False,
        'retrieval_parameters_changed': False, 'method_selection_performed': False,
        'limitations': config['limits'], 'aggregation': config['aggregation'], 'bootstrap': config['bootstrap'],
        'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status': 'COMPLETED_TRAINING_ONLY_TEMPERATURE_DESCRIPTION', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
