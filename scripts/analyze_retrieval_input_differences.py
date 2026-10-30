#!/usr/bin/env python3
"""Exploratory validation diagnostics linking evidence changes and saved answers.

No inference, target reads, method selection, causal effects or final-test claims.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from solvent_select.study import load_registry,evidence_name
from solvent_select.views import sha256_file,write_json


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def payload_identity(row):
    return tuple((card['card_id'],tuple(card['record_ids'])) for card in row['evidence'])


def balanced(frame,metrics):
    if frame.empty:return {}
    # Query blocks and all solvent pairs sharing a physical condition count once.
    grouped=frame.groupby(['solute_id','condition_id'])[metrics].mean()
    return {key:float(value) for key,value in grouped.groupby(level=0).mean().mean().items()
            if pd.notna(value)}


def aligned(path,queries):
    rows=read(path)
    if len(rows)!=len(queries) or {r['query_id'] for r in rows}!={q['query_id'] for q in queries}:
        raise ValueError('Incomplete or duplicate diagnostic cohort')
    return {r['query_id']:r for r in rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='configs/retrieval_input_difference_diagnostic_v1.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve earlier diagnostics')
    plan=json.loads(Path(args.plan).read_text());registry=load_registry()
    sources={__file__:sha256_file(__file__),args.plan:sha256_file(args.plan)}
    input_rows=[];answers=[];pending=[]
    for protocol,config in registry['protocols'].items():
        qpath=Path(config['views'])/'val_queries.jsonl';queries=read(qpath)
        roles=pd.read_csv(config['split'],dtype={'solute_id':str}).set_index('solute_id').split.to_dict()
        if any(q.get('split')!='val' or roles.get(q['solute_id'])!='val' for q in queries):
            raise ValueError('Only validation is permitted')
        for path in [qpath,Path(config['split']),Path(config['measurements'])]:sources[str(path)]=sha256_file(path)
        references={}
        for mode in {m for pair in plan['mode_pairs'] for m in pair}:
            path=Path(config['runs_root'])/evidence_name(protocol,config,mode)/'predictions.jsonl'
            sidecar=path.with_suffix(path.suffix+'.run.json');metadata=json.loads(sidecar.read_text())
            if metadata['status']!='completed' or metadata['fingerprint']['query_sha256']!=sha256_file(qpath):
                raise ValueError('Reference must be completed and match cohort')
            references[mode]=aligned(path,queries)
            for file in [path,sidecar]:sources[str(file)]=sha256_file(file)
        for first,second in plan['mode_pairs']:
            records=[]
            for q in queries:
                a,b=references[first][q['query_id']],references[second][q['query_id']]
                records.append({**{k:q[k] for k in ['query_id','solute_id','condition_id','task']},
                                'same_payload':float(payload_identity(a)==payload_identity(b))})
            frame=pd.DataFrame(records)
            for task,subset in frame.groupby('task'):
                input_rows.append({'protocol':protocol,'first':first,'second':second,'task':task,
                    'queries':len(subset),'solutes':subset.solute_id.nunique(),
                    'same_payload_queries':int(subset.same_payload.sum()),
                    'same_payload_solute_macro':balanced(subset,['same_payload'])['same_payload']})
            for seed in plan['learning_seeds']:
                names=[config['run_prefix']+protocol+f'_qwen_rag_{seed}_{mode}_val' for mode in [first,second]]
                runs=[Path(config['runs_root'])/name for name in names]
                paths=[run/'predictions.jsonl' for run in runs]
                sidecars=[p.with_suffix(p.suffix+'.run.json') for p in paths]
                evaluations=[run/'evaluation_view10000' for run in runs]
                if not all(s.exists() and json.loads(s.read_text()).get('status')=='completed' and (e/'metrics.json').exists()
                           for s,e in zip(sidecars,evaluations)):
                    pending.append({'protocol':protocol,'seed':seed,'first':first,'second':second});continue
                metas=[json.loads(s.read_text()) for s in sidecars]
                if metas[0]['fingerprint']['adapter_sha256']!=metas[1]['fingerprint']['adapter_sha256']:
                    raise ValueError('Comparison must use the same trained adapter')
                predictions=[aligned(p,queries) for p in paths];scores=[]
                for mode,path,sidecar,evaluation,metadata in zip([first,second],paths,sidecars,evaluations,metas):
                    metric=json.loads((evaluation/'metrics.json').read_text())
                    expected={str(p.resolve()):sha256_file(p) for p in [qpath,path]}
                    actual={str(Path(p).resolve()):h for p,h in metric['inputs'].items()}
                    if any(actual.get(p)!=h for p,h in expected.items()):raise ValueError('Saved score provenance differs')
                    if metadata['fingerprint']['query_sha256']!=sha256_file(qpath):raise ValueError('Model cohort differs')
                    score=pd.read_parquet(evaluation/'per_query.parquet').set_index('query_id')
                    if score.index.has_duplicates or set(score.index)!={q['query_id'] for q in queries}:
                        raise ValueError('Wrong saved score cohort')
                    scores.append(score)
                    for p in [path,sidecar,evaluation/'metrics.json',evaluation/'per_query.parquet']:sources[str(p)]=sha256_file(p)
                    for qid,row in predictions[[first,second].index(mode)].items():
                        if payload_identity(row)!=payload_identity(references[mode][qid]):
                            raise ValueError('Model evidence differs from reference')
                rows=[]
                for q in queries:
                    qid=q['query_id'];a,b=[p[qid]['prediction'] for p in predictions]
                    record={**{k:q[k] for k in ['solute_id','condition_id','task']},
                            'same_payload':payload_identity(references[first][qid])==payload_identity(references[second][qid]),
                            'answer_changed':float(a!=b)}
                    if q['task']=='rank':
                        selected=lambda x:x['ranking'][0] if isinstance(x,dict) and isinstance(x.get('ranking'),list) and x['ranking'] else None
                        record['first_ranked_candidate_changed']=float(selected(a)!=selected(b))
                    for metric in ['pair_accuracy','top1','regret']:
                        record[metric+'_difference']=scores[0].loc[qid,metric]-scores[1].loc[qid,metric]
                    rows.append(record)
                frame=pd.DataFrame(rows)
                for task,task_rows in frame.groupby('task'):
                    metrics=['answer_changed','regret_difference']+(['pair_accuracy_difference'] if task=='pair' else ['first_ranked_candidate_changed','top1_difference'])
                    for group in plan['groups']:
                        subset=task_rows if group=='ALL' else task_rows[task_rows.same_payload==(group=='same_payload')]
                        answers.append({'protocol':protocol,'seed':seed,'first':first,'second':second,'task':task,'group':group,
                            'queries':len(subset),'solutes':subset.solute_id.nunique(),**balanced(subset,metrics)})
    write_json(output,{'status':'EXPLORATORY_VALIDATION_ONLY','plan':args.plan,'input_overlap':input_rows,
                      'answer_diagnostics':answers,'pending_comparisons':pending,'evidence_sha256':sources,
                      'test_targets_read':False,'models_invoked':False,'methods_or_hyperparameters_changed':False,
                      'interpretation':'Within-group descriptions are conditional on observed input changes, not randomized or causal effects. One-seed rows are not a three-seed efficacy claim. Batched numerical contexts can differ even for identical evidence.'})


if __name__=='__main__':main()
