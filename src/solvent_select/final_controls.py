"""Frozen, CPU-only consistency checks; never reads held-out target values."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
import time

import pandas as pd

from .consistency import read, score, write_rows
from .evaluate import property_predictions
from .llm import _inference_input, _tokenizer, load_config, validate_query_manifest, write_json
from .release_guard import verify_release
from .retrieval import RetrievalIndex, canonical_smiles, direct_evidence_prediction
from .views import sha256_file


def molecular_rows(queries):
    converted, rows = [], []
    for original in queries:
        query = copy.deepcopy(original)
        for candidate in query['candidates']:
            candidate['record_id'] = query['query_id'] + ':' + candidate['id']
            rows.append({'record_id': candidate['record_id'], 'solute_id': query['solute_id'],
                         'solute_smiles': query['solute_smiles'], 'solvent_id': candidate['id'],
                         'solvent_smiles': candidate['smiles'], 'temperature_k': query['temperature_k']})
        converted.append(query)
    return converted, pd.DataFrame(rows)


def property_predictor(control):
    run = Path(control['model_run'])
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata['status'] != 'completed':
        raise ValueError('Property checkpoint is not complete')
    if metadata['method'] != 'chemprop':
        import joblib
        bundle = joblib.load(run / 'model.joblib')
        def predict(frame):
            if bundle['method'] == 'nearest':
                return bundle['model'].predict(frame)[0]
            return bundle['model'].predict(bundle['features'].transform(frame), num_threads=4)
        return predict, metadata
    import torch
    import lightning.pytorch as pl
    from .chemprop_runner import ChempropConfig, make_dataset, make_loader, make_model
    torch.set_num_threads(4)
    config = dataclasses.replace(ChempropConfig(**metadata['config']), num_workers=0, torch_threads=4)
    scalers = json.loads((run / 'scalers.json').read_text())
    model = make_model(config)
    checkpoint = torch.load(metadata['best_checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    def predict(frame):
        dataset = make_dataset(frame, scalers, labels=False)
        loader = make_loader(dataset, config, training=False, seed=metadata['seed'])
        return torch.cat(trainer.predict(model, dataloaders=loader)).cpu().numpy().reshape(-1) * scalers['target_std'] + scalers['target_mean']
    return predict, metadata


def evidence_predictor(control,split_path):
    config=load_config(control['config'],retrieval_mode=control['retrieval_mode'],seed=17)
    tokenizer=_tokenizer(config)
    index=RetrievalIndex.from_files(control['measurements'],split_path)
    def predict(cohort):
        rows,changes=[],0
        for original in cohort:
            query=copy.deepcopy(original)
            old=[query['solute_smiles']]+[c['smiles'] for c in query['candidates']]
            query['solute_smiles']=canonical_smiles(query['solute_smiles'])
            for candidate in query['candidates']:
                candidate['smiles']=canonical_smiles(candidate['smiles'])
            new=[query['solute_smiles']]+[c['smiles'] for c in query['candidates']]
            changes+=sum(a!=b for a,b in zip(old,new))
            _,evidence=_inference_input(query,tokenizer,config,index,control['retrieval_mode'])
            prediction=direct_evidence_prediction(query,evidence)
            rows.append({'query_id':query['query_id'],'prediction':prediction['prediction'],
                         'evidence':evidence.audit,'pair_coverage':prediction['pair_coverage']})
        return rows,changes
    return predict


def run_control(release_path, control_id, output):
    verification = verify_release(release_path)
    release = json.loads(Path(release_path).read_text())
    choices = [c for c in release['cpu_consistency_controls'] if c['id'] == control_id]
    if len(choices) != 1:
        raise ValueError('Control not uniquely registered in frozen release')
    control = choices[0]
    output = Path(output)
    if (output / 'run.json').exists():
        raise FileExistsError('Preserve completed control outputs; queue must reuse its receipt')
    folder = Path(release['consistency_manifest_path']).parent
    cohorts = [read(folder / f'{name}_queries.jsonl') for name in ['base', 'transformed']]
    for queries in cohorts:
        validate_query_manifest(queries, release['split_path'])
        if {q['split'] for q in queries} != {'test'}:
            raise ValueError('Only the frozen test consistency cohort is authorized')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base_changes = 0
    if control['kind'] == 'property':
        predict, metadata = property_predictor(control)
        for name, cohort in zip(['base', 'transformed'], cohorts):
            converted, frame = molecular_rows(cohort)
            if control['canonicalize_query']:
                for column in ['solute_smiles', 'solvent_smiles']:
                    normalized = frame[column].map(canonical_smiles)
                    if name == 'base':
                        base_changes += int((normalized != frame[column]).sum())
                    frame[column] = normalized
            values = predict(frame)
            rows = property_predictions(converted, pd.DataFrame({'record_id': frame.record_id, 'pred_logx': values}))
            write_rows(output / f'{name}_predictions.jsonl', rows)
    else:
        metadata = {}
        predict=evidence_predictor(control,release['split_path'])
        for name, cohort in zip(['base', 'transformed'], cohorts):
            rows,changes=predict(cohort)
            if name=='base':base_changes=changes
            write_rows(output / f'{name}_predictions.jsonl', rows)
    result = score(folder, output / 'base_predictions.jsonl', output / 'transformed_predictions.jsonl',
                   output / 'evaluation', freeze_manifest=release_path)
    write_json(output / 'run.json', {'status': 'completed', 'accelerator': 'cpu', 'elapsed_seconds': time.monotonic()-started,
               'control': control, 'seed': metadata.get('seed'), 'canonicalize_query': control['canonicalize_query'],
               'base_structure_fields_changed': base_changes, 'test_target_values_read': False,
               'release': verification, 'code_sha256': sha256_file(__file__),
               'scope': 'Label-free consistency; base and transformed predictions use the same CPU and checkpoint. Timings here are not GPU inference latency.'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--control-id', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run_control(args.release, args.control_id, args.output), indent=2))


if __name__ == '__main__':
    main()
