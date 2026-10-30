#!/usr/bin/env python3
"""Rebuild validation consistency tables/figures from actual per-solute outputs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.report import bootstrap_mean
from solvent_select.views import sha256_file, write_json


def main():
    groups = {
        'Nearest neighbor': ['artifacts/consistency/classical_val_canonical_fix/identity_nearest/evaluation'],
        'LightGBM (3 seeds)': [f'artifacts/consistency/classical_val/identity_lgbm_{s}/evaluation' for s in [17, 42, 2026]],
        'BM25 (raw tokens)': ['artifacts/runs/identity_evidence_text_consistency_val/evaluation'],
        'BM25 + canonicalization': ['artifacts/consistency/canonical_text_control_val/evaluation'],
        'Structure evidence': ['artifacts/runs/identity_evidence_structure_consistency_val/evaluation'],
        'Condition evidence': ['artifacts/runs/identity_evidence_condition_consistency_val/evaluation'],
    }
    rows, sources = [], {}
    for method, directories in groups.items():
        frames, manifests = [], set()
        for folder in map(Path, directories):
            metadata = json.loads((folder / 'metrics.json').read_text())
            if metadata['partition'] != 'val':
                raise ValueError('Validation plot cannot silently include final test outcomes')
            manifests.update(v for k, v in metadata['inputs'].items() if k.endswith('/manifest.json'))
            path = folder / 'per_solute.csv'
            frame = pd.read_csv(path, dtype={'solute_id': str})
            if frame.duplicated(['variant', 'task', 'solute_id']).any():
                raise ValueError('Duplicate consistency observation')
            frames.append(frame.set_index(['variant', 'task', 'solute_id']).sort_index())
            sources[str(path)] = sha256_file(path)
            sources[str(folder / 'metrics.json')] = sha256_file(folder / 'metrics.json')
        if len(manifests) != 1 or any(not f.index.equals(frames[0].index) for f in frames):
            raise ValueError('Cannot pool unequal consistency cohorts')
        means = sum(frames) / len(frames)
        for (variant, task), group in means.reset_index().groupby(['variant', 'task']):
            for metric in ['exact_agreement', 'top1_agreement', 'both_valid', 'any_smiles_changed']:
                low, high = bootstrap_mean(group[metric], 10000, 20261025)
                rows.append({'method': method, 'variant': variant, 'task': task, 'metric': metric,
                             'value': group[metric].mean(), 'ci_low': low, 'ci_high': high,
                             'solutes': len(group), 'runs': len(directories),
                             'consistency_manifest_sha256': next(iter(manifests))})
    output = Path('artifacts/reports/consistency_validation')
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', sharex=True)
    for axis, task in zip(axes, ['pair', 'rank']):
        frame = summary[(summary.variant == 'equivalent_smiles') & (summary.task == task)
                        & (summary.metric == 'exact_agreement')].set_index('method').reindex(groups)
        y = np.arange(len(frame))
        axis.barh(y, frame.value * 100, color='#31688e')
        axis.errorbar(frame.value * 100, y, xerr=np.array([frame.value - frame.ci_low, frame.ci_high - frame.value]) * 100,
                      fmt='none', ecolor='black', capsize=3)
        axis.set_yticks(y, frame.index); axis.invert_yaxis(); axis.set_xlim(0, 103)
        axis.set_xlabel('Exact mapped-answer agreement (%)'); axis.set_title(task)
    fig.suptitle('Equivalent-SMILES checks on512 fixed validation questions')
    for extension in ['png', 'pdf']:
        fig.savefig(output / ('equivalent_smiles_agreement.' + extension), dpi=180)
    plt.close(fig)
    write_json(output / 'manifest.json', {'status': 'VALIDATION_ONLY', 'sources': sources,
               'code_sha256': sha256_file(__file__), 'bootstrap_seed': 20261025, 'bootstrap_repetitions': 10000,
               'estimator': 'Seed average within solute, then resample whole solutes; conditional on512 fixed queries.',
               'interpretation': 'Canonicalized BM25 is a cheap control: raw-token serialization sensitivity is not an intrinsic advantage unique to chemical retrieval. No LM invariance result is claimed.',
               'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    print(summary[summary.metric == 'exact_agreement'].to_string(index=False))


if __name__ == '__main__':
    main()
