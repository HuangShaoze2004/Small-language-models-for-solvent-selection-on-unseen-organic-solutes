#!/usr/bin/env python3
"""CPU-only, label-free Chemprop input consistency from completed checkpoints."""
from pathlib import Path
import argparse
import dataclasses
import json
import time

import lightning.pytorch as pl
import pandas as pd
import torch

from check_classical_consistency import molecular_rows
from solvent_select.chemprop_runner import ChempropConfig,make_dataset,make_loader,make_model
from solvent_select.consistency import read,score,write_rows
from solvent_select.evaluate import property_predictions
from solvent_select.llm import validate_query_manifest
from solvent_select.views import sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=['identity','scaffold'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    name=f'sourceqa_{args.protocol}_chemprop_{args.seed}'
    run=Path('artifacts/runs_sourceqa')/name
    metadata=json.loads((run/'metadata.json').read_text())
    if metadata['status']!='completed':raise ValueError('Only completed checkpoints may be checked')
    for path,expected in [(metadata['measurements'],metadata['measurement_sha256']),(metadata['splits'],metadata['split_sha256'])]:
        if sha256_file(path)!=expected:raise ValueError('Model training inputs changed')
    folder=Path(f'data/consistency/sourceqa/{args.protocol}_val')
    queries=[read(folder/name) for name in ['base_queries.jsonl','transformed_queries.jsonl']]
    for cohort in queries:
        validate_query_manifest(cohort,metadata['splits'])
        if {q['split'] for q in cohort}!={'val'}:raise ValueError('Validation-only consistency diagnostic')
    destination=Path(f'artifacts/consistency/sourceqa/{args.protocol}')/name
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    config=dataclasses.replace(ChempropConfig(**metadata['config']),num_workers=0,torch_threads=4)
    scalers=json.loads((run/'scalers.json').read_text())
    model=make_model(config)
    checkpoint=torch.load(metadata['best_checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    trainer=pl.Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
    started=time.monotonic()
    for label,cohort in zip(['base','transformed'],queries):
        converted,frame=molecular_rows(cohort)
        dataset=make_dataset(frame,scalers,labels=False)
        loader=make_loader(dataset,config,training=False,seed=args.seed)
        predictions=trainer.predict(model,dataloaders=loader)
        values=torch.cat(predictions).cpu().numpy().reshape(-1)*scalers['target_std']+scalers['target_mean']
        output=property_predictions(converted,pd.DataFrame({'record_id':frame.record_id,'pred_logx':values}))
        write_rows(destination/f'{label}_predictions.jsonl',output)
    result=score(folder,destination/'base_predictions.jsonl',destination/'transformed_predictions.jsonl',destination/'evaluation')
    write_json(destination/'run.json',{'status':'completed','accelerator':'cpu','torch_threads':4,'elapsed_seconds':time.monotonic()-started,
               'checkpoint':metadata['best_checkpoint'],'checkpoint_sha256':sha256_file(metadata['best_checkpoint']),
               'scalers_sha256':sha256_file(run/'scalers.json'),'config':dataclasses.asdict(config),
               'test_labels_read':False,'code_sha256':sha256_file(__file__),
               'scope':'Base and transformed predictions recomputed on the same CPU device; inference timings here are diagnostics and not GPU latency.'})
    print(name,{key:value['exact_agreement'] for key,value in result['by_variant_task'].items()},flush=True)


if __name__=='__main__':main()
