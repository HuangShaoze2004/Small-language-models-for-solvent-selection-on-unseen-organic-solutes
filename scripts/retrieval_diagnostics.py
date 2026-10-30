#!/usr/bin/env python3
"""Label-free validation diagnostics of actual budgeted retrieval and ablations."""
from pathlib import Path
import argparse
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.llm import _inference_input, _tokenizer, load_config, read_jsonl, validate_query_manifest
from solvent_select.retrieval import RetrievalIndex
from solvent_select.views import sha256_file, write_json

MODES = ['condition', 'structure', 'text', 'no_temperature', 'no_solvent']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',default='data/views/identity/val_queries.jsonl')
    parser.add_argument('--manifest',default='data/splits/identity.csv')
    parser.add_argument('--measurements',default='data/processed/measurements.parquet')
    parser.add_argument('--config',default='configs/llm_probe.yaml')
    parser.add_argument('--output',default='artifacts/reports/retrieval_diagnostics_val')
    args=parser.parse_args()
    source = args.queries
    queries = read_jsonl(source)
    if {q['split'] for q in queries} != {'val'}:
        raise ValueError('This diagnostic is validation-only')
    validate_query_manifest(queries, args.manifest)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    index = RetrievalIndex.from_files(args.measurements, args.manifest)
    config = load_config(args.config)
    tokenizer = _tokenizer(config)
    condition_selection, rows = {}, []
    started = time.monotonic()
    for mode in MODES:
        for q in queries:
            start = time.monotonic()
            _, evidence = _inference_input(q, tokenizer, config, index, mode)
            ids = [row['card_id'] for row in evidence.audit]
            if mode == 'condition':
                condition_selection[q['query_id']] = ids
            else:
                reference = condition_selection[q['query_id']]
            reference = condition_selection[q['query_id']]
            sets = [{m['solvent_smiles'] for m in card['measurements']} for card in evidence.payload]
            pairs = [(a['smiles'], b['smiles']) for i, a in enumerate(q['candidates']) for b in q['candidates'][i+1:]]
            support = sum(any(a in s and b in s for s in sets) for a, b in pairs) / len(pairs)
            distances = [abs(float(card['temperature_K']) - q['temperature_k']) for card in evidence.payload]
            rows.append({'query_id': q['query_id'], 'solute_id': q['solute_id'], 'condition_id': q['condition_id'],
                         'task': q['task'], 'mode': mode, 'temperature_k': q['temperature_k'],
                         'cards': len(ids), 'evidence_tokens': evidence.token_count,
                         'candidate_coverage': np.mean([v > 0 for v in evidence.candidate_coverage.values()]),
                         'all_candidates_covered': float(all(v > 0 for v in evidence.candidate_coverage.values())),
                         'supported_pair_fraction': support,
                         'mean_temperature_distance_k': float(np.mean(distances)) if distances else None,
                         'exact_temperature_card_fraction': float(np.mean(np.asarray(distances) <= 1e-9)) if distances else 0.,
                         'same_cards_as_condition': float(ids == reference),
                         'card_set_jaccard_with_condition': len(set(ids) & set(reference)) / len(set(ids) | set(reference)) if ids or reference else 1.,
                         'retrieval_wall_seconds': time.monotonic() - start,
                         'card_ids': json.dumps(ids)})
        print(mode, 'completed', flush=True)
    details = pd.DataFrame(rows)
    columns = ['cards', 'evidence_tokens', 'candidate_coverage', 'all_candidates_covered', 'supported_pair_fraction',
               'mean_temperature_distance_k', 'exact_temperature_card_fraction', 'same_cards_as_condition',
               'card_set_jaccard_with_condition', 'retrieval_wall_seconds']
    per_solute = (details.groupby(['mode', 'task', 'solute_id', 'condition_id'])[columns].mean()
                  .groupby(['mode', 'task', 'solute_id']).mean().reset_index())
    summary = per_solute.groupby(['mode', 'task'])[columns].mean().reset_index()
    details.to_parquet(output / 'per_query.parquet', index=False)
    per_solute.to_csv(output / 'per_solute.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, task in zip(axes, ['pair', 'rank']):
        group = summary[summary.task == task].set_index('mode').reindex(MODES)
        ax.barh(MODES, group.supported_pair_fraction * 100, color='#31688e')
        ax.set_xlim(0, 100); ax.set_xlabel('Supported candidate pairs (%)'); ax.set_title(task)
        ax.invert_yaxis()
    fig.suptitle('Training evidence coverage on validation queries; labels unused')
    for extension in ['png', 'pdf']:
        fig.savefig(output / ('supported_pair_coverage.' + extension), dpi=180)
    plt.close(fig)
    write_json(output / 'manifest.json', {'status': 'VALIDATION_LABEL_FREE_DIAGNOSTICS', 'queries': len(queries),
               'modes': MODES, 'elapsed_seconds': time.monotonic() - started,
               'averaging': 'physical condition, then solute, then population; separate tasks',
               'test_labels_read': False, 'heldout_target_files_read': [],
               'inputs': {p: sha256_file(p) for p in [source, args.manifest, args.measurements,
                                                     args.config, 'src/solvent_select/llm.py',
                                                     'src/solvent_select/retrieval.py', __file__]},
               'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
               'interpretation': 'Evidence support is distinct from JSON validity. A valid TIE produced without a comparable training pair is not evidence-supported. These diagnostics do not score heldout answers or establish model effectiveness.'})


if __name__ == '__main__':
    main()
