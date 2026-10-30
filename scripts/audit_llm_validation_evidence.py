#!/usr/bin/env python3
"""Check actual completed LM evidence against the fixed same-budget workflow.

Prompt allowlists have separate software tests. This audit checks saved input
token counts, train-card selections, candidate coverage and adapter boundaries.
It never opens target values or scores model answers.
"""
import argparse
import json
from pathlib import Path

from solvent_select.llm import read_jsonl, validate_query_manifest
from solvent_select.report import _run_identity
from solvent_select.study import load_registry, evidence_name, verify_adapter_boundary
from solvent_select.views import sha256_file, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous actual-input audits')
    registry=load_registry();analysis=json.loads(Path(registry['analysis_plan']).read_text())
    rows,pending,sources=[],[],{__file__:sha256_file(__file__)}
    for protocol,config in registry['protocols'].items():
        query_path=Path(config['views'])/'val_queries.jsonl'
        queries=read_jsonl(query_path)
        validate_query_manifest(queries,config['split'])
        if {q['split'] for q in queries}!={'val'}:raise ValueError('Validation-only input audit')
        expected={q['query_id']:q for q in queries}
        sources[str(query_path)]=sha256_file(query_path)
        sources[config['split']]=sha256_file(config['split'])
        references={}
        for name in analysis['llm_validation_tasks_to_carry_to_test']:
            if not name.startswith(config['run_prefix']+protocol+'_'):continue
            folder=Path(config['runs_root'])/name
            path=folder/'predictions.jsonl';sidecar=path.with_suffix(path.suffix+'.run.json')
            if not sidecar.exists() or json.loads(sidecar.read_text()).get('status')!='completed':
                pending.append(name);continue
            metadata=json.loads(sidecar.read_text());cfg=metadata['fingerprint']['config'];mode=cfg['retrieval_mode']
            if (metadata['fingerprint']['query_sha256']!=sha256_file(query_path)
                    or metadata['fingerprint']['manifest_sha256']!=sha256_file(config['split'])
                    or cfg['max_sequence_length']!=3072 or cfg['max_evidence_tokens']!=1536 or cfg['max_new_tokens']!=256):
                raise ValueError('Wrong cohort, protocol or inference budget: '+name)
            identity,_,method,_=_run_identity(folder,{'inputs':{str(path):sha256_file(path)}})
            if identity.get('_training_path'):
                training=Path(identity['_training_path'])
                verify_adapter_boundary(json.loads(training.read_text()),config,cfg)
                sources[str(training)]=sha256_file(training)
            predictions=read_jsonl(path)
            if len(predictions)!=len(queries) or {p['query_id'] for p in predictions}!=set(expected):
                raise ValueError('Incomplete actual inference cohort')
            if mode!='none' and mode not in references:
                reference=Path(config['runs_root'])/evidence_name(protocol,config,mode)/'predictions.jsonl'
                references[mode]={r['query_id']:r for r in read_jsonl(reference)}
                if set(references[mode])!=set(expected):raise ValueError('Fixed-workflow cohort differs')
                sources[str(reference)]=sha256_file(reference)
            count=0
            for row in predictions:
                query=expected[row['query_id']]
                if any(row[key]!=query[key] for key in ['solute_id','task','split']):
                    raise ValueError('Saved input identity metadata differs from query')
                if row['input_tokens']+cfg['max_new_tokens']>cfg['max_sequence_length']:
                    raise ValueError('Actual input exceeds the registered inference budget')
                cards=row['evidence']
                if len(cards)>4 or len({c['solute_id'] for c in cards})!=len(cards):
                    raise ValueError('Actual input violates evidence-card budget')
                if mode=='none':
                    if cards or row['candidate_evidence_coverage']:raise ValueError('Plain inference has unexpected evidence')
                else:
                    fixed=references[mode][row['query_id']]
                    if (cards!=fixed['evidence'] or row['input_tokens']!=fixed['input_tokens']
                            or row['candidate_evidence_coverage']!=fixed['candidate_evidence_coverage']):
                        raise ValueError('LM and fixed workflow have different actual evidence/input budgets')
                count+=len(cards)
            rows.append({'run':name,'method':method,'queries_checked':len(predictions),'cards_checked':count,
                         'exact_fixed_workflow_match':mode!='none','plain_evidence_absent':mode=='none','status':'PASSED'})
            for p in [path,sidecar,folder/'experiment.json']:sources[str(p)]=sha256_file(p)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'IN_PROGRESS_PASSED_COMPLETED_LM_INPUTS' if pending else 'PASSED_ALL_REGISTERED_LM_INPUTS',
        'completed_inferences':rows,'pending_inferences':pending,'evidence_sha256':sources,
        'target_values_read':False,'test_queries_read':False,'answers_scored':False,
        'scope':'Actual saved train-card audit lists, token budgets and coverage match fixed workflows; adapter supervised/retrieval boundaries verified. Fixed cards have their separate training-membership audit. This is not an inspection of every rendered prompt string or an efficacy claim.'})
    print(json.dumps({'completed':len(rows),'pending':len(pending),'cards_checked':sum(r['cards_checked'] for r in rows)}))


if __name__=='__main__':main()
