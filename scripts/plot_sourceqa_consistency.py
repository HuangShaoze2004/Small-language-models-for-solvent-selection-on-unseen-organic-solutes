#!/usr/bin/env python3
"""Rebuild SOURCEQA validation sensitivity figures from measured outputs only."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.report import bootstrap_mean
from solvent_select.views import sha256_file, write_json


def groups(protocol):
    parent = Path('artifacts/consistency/sourceqa') / protocol
    prefix = f'sourceqa_{protocol}_'
    return {label: [parent / (prefix + suffix) / 'evaluation' for suffix in suffixes]
            for label, suffixes in {
                'Nearest neighbor': ['nearest'],
                'LightGBM': [f'lgbm_{s}' for s in [17, 42, 2026]],
                'Chemprop (raw input)': [f'chemprop_{s}' for s in [17, 42, 2026]],
                'Chemprop + canonicalization': [f'chemprop_{s}_canonical_control' for s in [17, 42, 2026]],
                'BM25 evidence (raw input)': ['evidence_text_val'],
                'BM25 + canonicalization': ['evidence_text_val_canonical_control'],
                'Structure evidence': ['evidence_structure_val'],
                'Condition evidence': ['evidence_condition_val'],
                'Condition without temperature': ['evidence_no_temperature_val'],
                'Condition without solvent': ['evidence_no_solvent_val'],
            }.items()}


def collect():
    rows, sources = [], {}
    for protocol in ['identity', 'scaffold']:
        for method, folders in groups(protocol).items():
            frames, manifests = [], set()
            for folder in folders:
                metadata = json.loads((folder / 'metrics.json').read_text())
                if metadata.get('partition') != 'val' or metadata.get('queries') != 1024:
                    raise ValueError('Require complete validation-only consistency cohorts')
                manifests.update(v for p, v in metadata['inputs'].items() if p.endswith('/manifest.json'))
                path = folder / 'per_solute.csv'
                frame = pd.read_csv(path, dtype={'solute_id': str})
                if frame.duplicated(['variant', 'task', 'solute_id']).any():
                    raise ValueError('Duplicate solute consistency observation')
                frames.append(frame.set_index(['variant', 'task', 'solute_id']).sort_index())
                for path in [path, folder / 'metrics.json', folder.parent / 'run.json']:
                    sources[str(path)] = sha256_file(path)
            if len(manifests) != 1 or any(not frames[0].index.equals(f.index) for f in frames):
                raise ValueError('Cannot pool unequal consistency cohorts')
            averaged = sum(frames) / len(frames)
            for (variant, task), frame in averaged.reset_index().groupby(['variant', 'task']):
                for metric in ['exact_agreement', 'top1_agreement', 'both_valid', 'any_smiles_changed']:
                    low, high = bootstrap_mean(frame[metric], 10000, 20261025)
                    rows.append({'protocol': protocol, 'method': method, 'variant': variant,
                                 'task': task, 'metric': metric, 'value': float(frame[metric].mean()),
                                 'ci_low': low, 'ci_high': high, 'solutes': len(frame),
                                 'runs': len(folders), 'cohort_sha256': next(iter(manifests))})
    return pd.DataFrame(rows), sources


def main():
    summary, sources = collect()
    output = Path('artifacts/reports/sourceqa_consistency_validation')
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for i, protocol in enumerate(['identity', 'scaffold']):
        for j, task in enumerate(['pair', 'rank']):
            frame = summary[(summary.protocol == protocol) & (summary.task == task)
                            & (summary.variant == 'equivalent_smiles')
                            & (summary.metric == 'exact_agreement')].set_index('method').reindex(groups(protocol))
            axis = axes[i, j]
            y = np.arange(len(frame))
            colors = ['#aaa' if 'canonicalization' in method else '#31688e' for method in frame.index]
            axis.barh(y, frame.value * 100, color=colors)
            error = np.maximum(0., np.array([frame.value-frame.ci_low, frame.ci_high-frame.value]) * 100)
            axis.errorbar(frame.value * 100, y, xerr=error, fmt='none', ecolor='black', capsize=2)
            axis.set_yticks(y, frame.index)
            axis.invert_yaxis()
            axis.set_xlim(0, 103)
            axis.set_title(f'{protocol}: {task}')
            axis.set_xlabel('Exact mapped-answer agreement (%)')
            axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Equivalent-SMILES sensitivity: fixed validation cohorts\nCanonicalization controls are preprocessing checks, not new methods')
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'equivalent_smiles.{suffix}', dpi=180)
    plt.close(fig)
    write_json(output / 'manifest.json', {'status': 'VALIDATION_ONLY', 'sources': sources,
        'code_sha256': sha256_file(__file__), 'bootstrap_repetitions': 10000, 'bootstrap_seed': 20261025,
        'estimator': 'Average3seeds within solute for learned property models, then bootstrap solutes; deterministic methods once. Conditional on512 selected queries per protocol.',
        'scope': 'SOURCEQA only. Raw and canonical controls both reported. No language-model result is inferred from direct-evidence results. Chemprop controls use its separately pinned RDKit; canonicalization changes22base structure fields in strict checks and0in identity checks.',
        'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    print(summary[(summary.variant == 'equivalent_smiles') & (summary.metric == 'exact_agreement')].to_string(index=False))


if __name__ == '__main__':
    main()
