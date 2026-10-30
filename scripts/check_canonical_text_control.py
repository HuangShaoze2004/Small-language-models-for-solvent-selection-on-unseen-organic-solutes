#!/usr/bin/env python3
"""Control: canonicalize the retrieval query before ordinary BM25 matching.

This cheap preprocessing control prevents treating raw-SMILES serialization
sensitivity as an inherent advantage unique to structural/condition retrieval.
It does not change the primary benchmark, whose inputs are already canonical.
"""
from pathlib import Path
import copy
import json
import time

from solvent_select.consistency import read, score, write_rows
from solvent_select.llm import _tokenizer, _inference_input, load_config, validate_query_manifest
from solvent_select.retrieval import RetrievalIndex, canonical_smiles, direct_evidence_prediction
from solvent_select.views import sha256_file, write_json


def main():
    queries = read('data/consistency/identity_val/transformed_queries.jsonl')
    validate_query_manifest(queries, 'data/splits/identity.csv')
    output = Path('artifacts/consistency/canonical_text_control_val')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = load_config('configs/llm_probe.yaml', retrieval_mode='text')
    tokenizer = _tokenizer(config)
    index = RetrievalIndex.from_files('data/processed/measurements.parquet', 'data/splits/identity.csv')
    predictions = []
    started = time.monotonic()
    for query in queries:
        canonical = copy.deepcopy(query)
        canonical['solute_smiles'] = canonical_smiles(query['solute_smiles'])
        for candidate in canonical['candidates']:
            candidate['smiles'] = canonical_smiles(candidate['smiles'])
        _, evidence = _inference_input(canonical, tokenizer, config, index, 'text')
        predictions.append({'query_id': query['query_id'],
                            'prediction': direct_evidence_prediction(canonical, evidence)['prediction']})
    write_rows(output / 'predictions.jsonl', predictions)
    result = score('data/consistency/identity_val',
                   'artifacts/runs/identity_evidence_text_val_view/predictions.jsonl',
                   output / 'predictions.jsonl', output / 'evaluation')
    write_json(output / 'run.json', {'status': 'completed', 'elapsed_seconds': time.monotonic() - started,
                                    'control': 'canonical SMILES before BM25 retrieval; same4cards1536tokens',
                                    'code_sha256': sha256_file(__file__), 'test_labels_read': False,
                                    'interpretation': 'If invariance is restored, raw-token inconsistency is removable preprocessing sensitivity, not an intrinsic chemistry-method improvement.'})
    print(json.dumps(result['by_variant_task'], indent=2))


if __name__ == '__main__':
    main()
