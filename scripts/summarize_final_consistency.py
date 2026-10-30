#!/usr/bin/env python3
"""Summarize every independently audited final consistency method and control."""
import argparse
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.report import bootstrap_mean
from solvent_select.study import load_registry
from solvent_select.views import sha256_file, write_json

spec = importlib.util.spec_from_file_location('final_labels', Path(__file__).with_name('plot_registered_final.py'))
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)
METRICS = ['exact_agreement', 'top1_agreement', 'both_valid', 'any_smiles_changed', 'changed_smiles_fraction']


def method(name, protocol, config):
    name = name.removesuffix('_consistency_test')
    if name.endswith('_canonical_control'):
        original = name.removesuffix('_canonical_control')
        original = original + '_test' if '_evidence_' in original else original
        return labels.method_label(original, protocol, config) + ' + canonicalization'
    name = name + '_test' if '_qwen_' in name or '_evidence_' in name else name
    return labels.method_label(name, protocol, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    if (audit['status'] != 'PASSED_ALL_REGISTERED_FINAL_CONSISTENCY_COMPARISONS'
            or len(audit['comparisons']) != 64):
        raise ValueError('All64 final comparisons must be independently audited')
    for path, digest in audit['evidence_sha256'].items():
        if sha256_file(path) != digest:
            raise ValueError('Audited source changed: ' + path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous figures')
    output.mkdir(parents=True)
    registry = load_registry()
    groups = OrderedDict()
    sources = {args.audit: sha256_file(args.audit), __file__: sha256_file(__file__),
               str(Path(labels.__file__)): sha256_file(labels.__file__)}
    for row in audit['comparisons']:
        metric_path = Path(row['comparison'])
        run = metric_path.parent.parent.name
        protocols = [protocol for protocol, config in registry['protocols'].items()
                     if run.startswith(config['run_prefix'] + protocol + '_')]
        if len(protocols) != 1:
            raise ValueError('Unknown consistency protocol')
        protocol = protocols[0]
        label = method(run, protocol, registry['protocols'][protocol])
        groups.setdefault((protocol, label), []).append(metric_path.parent)
    rows = []
    for (protocol, label), folders in groups.items():
        # Every stochastic learning group has the fixed three checkpoints.
        stochastic = label in ['LightGBM', 'Chemprop', 'Chemprop + canonicalization'] or label.startswith(('Ordinary SFT', 'Retrieval SFT'))
        if len(folders) != (3 if stochastic else 1):
            raise ValueError('Incomplete registered seed group: ' + label)
        frames = []
        cohort = set()
        for folder in folders:
            meta = json.loads((folder / 'metrics.json').read_text())
            if meta['partition'] != 'test' or meta['queries'] != 1024:
                raise ValueError('Unexpected consistency population')
            cohort.update(digest for path, digest in meta['inputs'].items() if path.endswith('/manifest.json'))
            path = folder / 'per_solute.csv'
            frame = pd.read_csv(path, dtype={'solute_id': str}).set_index(['variant', 'task', 'solute_id']).sort_index()
            if frame.index.has_duplicates or (frames and not frames[0].index.equals(frame.index)):
                raise ValueError('Seed support differs')
            if not np.isfinite(frame[METRICS].to_numpy()).all():
                raise ValueError('Missing consistency observations')
            frames.append(frame[METRICS])
            for path in [path, folder / 'metrics.json']:
                sources[str(path)] = sha256_file(path)
        if len(cohort) != 1:
            raise ValueError('Different transformation cohorts')
        mean = sum(frames) / len(frames)
        for (variant, task), frame in mean.reset_index().groupby(['variant', 'task']):
            for metric in METRICS:
                low, high = bootstrap_mean(frame[metric], 10000, 20261025)
                rows.append({'protocol': protocol, 'method': label, 'variant': variant, 'task': task,
                             'metric': metric, 'value': frame[metric].mean(), 'ci_low': low, 'ci_high': high,
                             'solutes': len(frame), 'runs': len(frames), 'run_ids': ';'.join(f.parent.name for f in folders)})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'summary.csv', index=False)
    for protocol in registry['protocols']:
        names = [label for current, label in groups if current == protocol]
        fig, axes = plt.subplots(2, 2, figsize=(16, max(10, len(names) * .55)), layout='constrained')
        for i, variant in enumerate(['candidate_order', 'equivalent_smiles']):
            for j, task in enumerate(['pair', 'rank']):
                axis = axes[i, j]
                subset = table[(table.protocol == protocol) & (table.variant == variant)
                               & (table.task == task) & (table.metric == 'exact_agreement')].set_index('method').loc[names]
                y = np.arange(len(names))
                error = np.maximum(0, np.asarray([subset.value-subset.ci_low, subset.ci_high-subset.value])) * 100
                axis.errorbar(subset.value * 100, y, xerr=error, fmt='o', capsize=2, markersize=4, color='#31688e')
                axis.set_yticks(y, names, fontsize=8)
                axis.invert_yaxis()
                axis.set_xlim(-1, 102)
                axis.set_xlabel('Exact mapped-answer agreement (%)')
                axis.set_title(variant.replace('_', ' ') + ' / ' + task)
                axis.spines[['top', 'right']].set_visible(False)
                axis.grid(axis='x', alpha=.2)
        fig.suptitle(f'Frozen test consistency: {protocol}\nAll methods and controls; invalid outputs retain disagreement penalties')
        for extension in ['png', 'pdf']:
            fig.savefig(output / f'{protocol}_consistency.{extension}', dpi=180)
        plt.close(fig)
    write_json(output / 'manifest.json', {'status': 'ALL_FINAL_CONSISTENCY_GROUPS_COMPLETE',
        'comparisons': 64, 'method_groups': len(groups), 'summary_rows': len(rows), 'sources': sources,
        'bootstrap_repetitions': 10000, 'bootstrap_seed': 20261025,
        'scope': 'Seed average within solute, then solute bootstrap on fixed512 base questions and1024 transformations; repeated questions are not independent chemical samples. All registered LM, property and evidence groups included. Consistency alone does not establish correctness.',
        'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'method_groups': len(groups), 'summary_rows': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    main()
