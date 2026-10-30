#!/usr/bin/env python3
"""Validation-only evidence consistency, including a cheap BM25 control."""
import argparse
import copy
import json
from pathlib import Path
import time

from solvent_select.consistency import read, score, write_rows
from solvent_select.llm import LLMConfig, _inference_input, _tokenizer, validate_query_manifest
from solvent_select.retrieval import RetrievalIndex, canonical_smiles, direct_evidence_prediction
from solvent_select.views import sha256_file, write_json
from solvent_select.study import DEFAULT_REGISTRY,load_registry,evidence_name


def canonical_query(query):
    query = copy.deepcopy(query)
    query['solute_smiles'] = canonical_smiles(query['solute_smiles'])
    for candidate in query['candidates']:
        candidate['smiles'] = canonical_smiles(candidate['smiles'])
    return query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['identity', 'scaffold'], required=True)
    parser.add_argument('--mode', choices=['text', 'structure', 'condition', 'no_temperature', 'no_solvent'], required=True)
    parser.add_argument('--canonicalize-query', action='store_true')
    args = parser.parse_args()
    config_paths=load_registry()['protocols'][args.protocol]
    if args.canonicalize_query and args.mode != 'text':
        raise ValueError('The registered canonicalization control is BM25 only')
    name = evidence_name(args.protocol,config_paths,args.mode)
    run = Path(config_paths['runs_root']) / name
    base_path = run / 'predictions.jsonl'
    metadata_path = base_path.with_suffix('.jsonl.run.json')
    metadata = json.loads(metadata_path.read_text())
    if metadata['status'] != 'completed':
        raise ValueError('Require completed base predictions')
    fingerprint = metadata['fingerprint']
    manifest = config_paths['split']
    measurements = config_paths['measurements']
    if sha256_file(manifest) != fingerprint['manifest_sha256'] or sha256_file(measurements) != fingerprint['measurements_sha256']:
        raise ValueError('Training bank or split changed')
    folder = Path(config_paths['consistency_data_root'])/f'{args.protocol}_val'
    base, transformed = [read(folder / filename) for filename in ['base_queries.jsonl', 'transformed_queries.jsonl']]
    for queries in [base, transformed]:
        validate_query_manifest(queries, manifest)
        if {q['split'] for q in queries} != {'val'}:
            raise ValueError('Only validation diagnostics are allowed')
    # The cached base predictions are valid for this control only if its public
    # structures are unchanged by canonicalization in the same environment.
    if args.canonicalize_query and any(canonical_query(q) != q for q in base):
        raise ValueError('Canonical control would change base inputs; recompute both arms')
    destination = Path(config_paths['consistency_runs_root'])/args.protocol / (name + ('_canonical_control' if args.canonicalize_query else ''))
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    config = LLMConfig(**fingerprint['config'])
    tokenizer = _tokenizer(config)
    index = RetrievalIndex.from_files(measurements, manifest)
    predictions = []
    started = time.monotonic()
    for query in transformed:
        prepared = canonical_query(query) if args.canonicalize_query else query
        _, evidence = _inference_input(prepared, tokenizer, config, index, args.mode)
        direct = direct_evidence_prediction(prepared, evidence)
        predictions.append({'query_id': query['query_id'], 'prediction': direct['prediction'],
                            'evidence': evidence.audit, 'pair_coverage': direct['pair_coverage']})
    write_rows(destination / 'predictions.jsonl', predictions)
    result = score(folder, base_path, destination / 'predictions.jsonl', destination / 'evaluation')
    sources = [base_path, metadata_path, folder / 'manifest.json', Path(manifest), Path(measurements)]
    write_json(destination / 'run.json', {'status': 'completed', 'elapsed_seconds': time.monotonic() - started,
               'canonicalize_query': args.canonicalize_query, 'retrieval_mode': args.mode,
               'test_labels_read': False, 'code_sha256': sha256_file(__file__),
               'inputs': {str(p): sha256_file(p) for p in sources},
               'scope': '512 fixed validation questions; same training-only evidence bank,4cards1536tokens; no language model inference.'})
    print(name, args.canonicalize_query, {k: v['exact_agreement'] for k, v in result['by_variant_task'].items()}, flush=True)


if __name__ == '__main__':
    main()
