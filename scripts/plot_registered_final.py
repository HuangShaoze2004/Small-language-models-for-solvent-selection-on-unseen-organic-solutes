#!/usr/bin/env python3
"""Plot all independently graded frozen-test methods; no seed dropping."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.report import bootstrap_mean
from solvent_select.study import load_registry
from solvent_select.validation_assessment import registered_runs
from solvent_select.views import sha256_file, write_json


def method_label(name, protocol, config):
    name = name.removesuffix('_test') + '_val' if name.endswith('_test') else name
    stem = name.removeprefix(config['run_prefix'] + protocol + '_')
    if stem == 'nearest':
        return 'Nearest neighbor'
    match = re.fullmatch(r'(lgbm|chemprop)_(17|42|2026)', stem)
    if match:
        return {'lgbm': 'LightGBM', 'chemprop': 'Chemprop'}[match[1]]
    modes = {'none': 'no retrieval', 'text': 'BM25', 'structure': 'structure', 'condition': 'condition',
             'no_temperature': 'without temperature', 'no_solvent': 'without solvent match'}
    match = re.fullmatch(r'evidence_(text|structure|condition|no_temperature|no_solvent)_val', stem)
    if match:
        return 'Fixed evidence / ' + modes[match[1]]
    match = re.fullmatch(r'qwen_(base|sft|rag)_(17|42|2026)_(none|text|structure|condition|no_temperature|no_solvent)_val', stem)
    if match:
        return {'base': 'Base LM', 'sft': 'Ordinary SFT', 'rag': 'Retrieval SFT'}[match[1]] + ' / ' + modes[match[3]]
    raise ValueError('Unknown registered method: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    snapshot = json.loads(Path(args.assessment).read_text())
    if snapshot['status'] != 'PASSED_ALL_REGISTERED_FINAL_GRADING' or snapshot['test_targets_read'] is not True:
        raise ValueError('Only the complete independent frozen-test score audit may supply this figure')
    for path, expected in snapshot['evidence_sha256'].items():
        if sha256_file(path) != expected:
            raise ValueError('Assessment evidence changed: ' + path)
    registry = load_registry()
    analysis = json.loads(Path(registry['analysis_plan']).read_text())
    specifications = [(protocol, name.removesuffix('_val')+'_test' if name.endswith('_val') else name,
        folder.parent.with_name(name.removesuffix('_val')+'_test' if name.endswith('_val') else name)/'evaluation_test_view10000')
        for protocol,name,folder in registered_runs(registry, analysis)]
    if set(row['run'] for row in snapshot['runs']) != {name for _, name, _ in specifications}:
        raise ValueError('Assessment and exact registered method set disagree')
    states = {row['run']: row for row in snapshot['runs']}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing figures; choose a new snapshot path')
    output.mkdir(parents=True)
    sources = {args.assessment: sha256_file(args.assessment), __file__: sha256_file(__file__)}
    rows, pending = [], []
    for protocol, config in registry['protocols'].items():
        groups = OrderedDict()
        for current, name, folder in specifications:
            if current == protocol:
                groups.setdefault(method_label(name, protocol, config), []).append((name, folder))
        for label, members in groups.items():
            complete = [name for name, _ in members if states[name]['status'] == 'PASSED']
            if len(complete) != len(members):
                pending.append({'protocol': protocol, 'method': label, 'completed_runs': len(complete),
                                'required_runs': len(members), 'missing': [name for name, _ in members if name not in complete]})
                continue
            frames = []
            for name, folder in members:
                path = folder / 'per_solute.csv'
                frame = pd.read_csv(path, dtype={'solute_id': str}).set_index(['solute_id', 'task']).sort_index()
                if frames and not frames[0].index.equals(frame.index):
                    raise ValueError('Seed cohorts differ: ' + label)
                frames.append(frame)
                sources[str(path)] = sha256_file(path)
            for task, metric in [('pair', 'pair_accuracy'), ('rank', 'top1'), ('rank', 'regret')]:
                vectors = [frame.xs(task, level='task')[metric] for frame in frames]
                matrix = pd.concat(vectors, axis=1)
                if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
                    raise ValueError('Missing/nonfinite required observations; no seed dropping')
                mean = matrix.mean(axis=1, skipna=False)
                low, high = bootstrap_mean(mean, 10000, 20261025)
                seed_means = matrix.mean(axis=0)
                rows.append({'protocol': protocol, 'data_tag': config['data_tag'], 'method': label, 'task': task,
                             'metric': metric, 'value': mean.mean(), 'ci_low': low, 'ci_high': high,
                             'solute_count': len(mean), 'runs': len(members),
                             'seed_sd': float(seed_means.std(ddof=1)) if len(members) > 1 else None,
                             'run_ids': ';'.join(name for name, _ in members)})
        table = pd.DataFrame([row for row in rows if row['protocol'] == protocol])
        fig, axes = plt.subplots(1, 3, figsize=(15, max(6, .37 * len(groups) + 1.7)),
                                 sharey=True, layout='constrained')
        for axis, (task, metric, title, scale) in zip(axes, [
                ('pair', 'pair_accuracy', 'Comparison accuracy (%)', 100),
                ('rank', 'top1', 'Best-solvent hit rate (%)', 100),
                ('rank', 'regret', 'Selection regret (log10 mole fraction)', 1)]):
            subset = table[(table.task == task) & (table.metric == metric)].set_index('method')
            for index, label in enumerate(groups):
                if label not in subset.index:
                    item = next(row for row in pending if row['protocol'] == protocol and row['method'] == label)
                    axis.text(.02, index, f"PENDING ({item['completed_runs']}/{item['required_runs']} runs)",
                              transform=axis.get_yaxis_transform(), fontsize=8, va='center', color='#777')
                    continue
                record = subset.loc[label]
                interval = np.maximum(0, [[record.value - record.ci_low], [record.ci_high - record.value]]) * scale
                axis.errorbar(record.value * scale, index, xerr=interval, fmt='o', markersize=4,
                              color='#31688e', capsize=2)
            axis.set_yticks(range(len(groups)), list(groups), fontsize=8)
            axis.set_xlabel(title, fontsize=9)
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='x', alpha=.2)
            axis.set_ylim(len(groups) - .4, -.6)
            if scale == 100:
                axis.set_xlim(0, 100)
            else:
                axis.set_xlim(left=0)
        fig.suptitle(f"Registered frozen test: {protocol} / {config['data_tag']}\n"
                     '95% solute-bootstrap intervals after averaging complete seed groups; fixed measured cohort', fontsize=11)
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'{protocol}_final.{suffix}', dpi=180)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    write_json(output / 'manifest.json', {'status': 'FINAL_METHODS_INCOMPLETE' if pending else 'FINAL_METHODS_COMPLETE',
               'pending_groups': pending, 'sources': sources, 'test_scores_read': True,
               'bootstrap_repetitions': 10000, 'bootstrap_seed': 20261025,
               'scope': 'Registered methods only. All scheduled seeds are required before a method estimate is plotted. '
                        'Intervals describe fixed-checkpoint variation over measured solutes, not uncertainty from training seeds or experimental error.',
               'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'summary_rows': len(rows), 'pending_groups': len(pending), 'output': str(output)}))


if __name__ == '__main__':
    main()
