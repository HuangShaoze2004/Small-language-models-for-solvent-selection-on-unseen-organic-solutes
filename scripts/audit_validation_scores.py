#!/usr/bin/env python3
"""Independent validation grading from raw outputs and measured values.

Does not import the production parser/evaluator/aggregator. Reads validation
targets only after validating the public cohort and manifest membership. Never
invokes a model, modifies predictions or opens experimental test values.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean

import pandas as pd
from rdkit import Chem

from solvent_select.study import load_registry,property_names,evidence_name
from solvent_select.views import sha256_file,write_json

METRICS=['pair_accuracy','nontie_pair_accuracy','top1','rank_tau','regret','covered']
BAND=.1+1e-12


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def choice(values):
    difference=float(values['A'])-float(values['B'])
    return 'TIE' if -BAND<=difference<=BAND else ('A' if difference>0 else 'B')


def decode(text,query):
    if not isinstance(text,str):return None
    text=text.strip()
    if text[:8]=='```json\n' and text[-3:]=='```':text=text[8:-3].strip()
    try:result=json.loads(text)
    except (ValueError,TypeError):return None
    if not isinstance(result,dict):return None
    if query['task']=='pair':
        value=result.get('choice')
        return result if list(result.keys())==['choice'] and isinstance(value,str) and value in ['A','B','TIE'] else None
    ranking=result.get('ranking');ids={c['id'] for c in query['candidates']}
    if (set(result)!= {'ranking'} or not isinstance(ranking,list) or any(type(x) is not str for x in ranking)
            or len(ranking)!=len(ids) or set(ranking)!=ids):return None
    return result


@lru_cache(maxsize=8192)
def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True,isomericSmiles=True)


def utility(query,truth,answer):
    best,worst=max(truth.values()),min(truth.values())
    row={name:math.nan for name in METRICS}
    row.update(covered=0.,regret=best-worst)
    if query['task']=='pair':
        expected=choice(truth)
        valid=answer is not None and isinstance(answer.get('choice'),str) and answer['choice'] in ['A','B','TIE']
        row['pair_accuracy']=float(valid and answer['choice']==expected)
        if expected!='TIE':row['nontie_pair_accuracy']=row['pair_accuracy']
        if valid:
            selected=answer['choice']
            if selected=='TIE':selected=sorted(query['candidates'],key=lambda c:(canonical(c['smiles']),c['id']))[0]['id']
            row.update(covered=1.,regret=best-truth[selected])
        return row
    row.update(top1=0.,rank_tau=-1.)
    ranking=answer.get('ranking') if answer is not None else None
    ids=[c['id'] for c in query['candidates']]
    valid=(isinstance(ranking,list) and all(type(x) is str for x in ranking)
           and len(ranking)==len(ids) and set(ranking)==set(ids))
    if valid:
        selected=ranking[0];position={cid:i for i,cid in enumerate(ranking)}
        concordant=discordant=0
        for a,b in combinations(ids,2):
            if truth[a]==truth[b]:continue
            aligned=(truth[a]>truth[b])==(position[a]<position[b])
            concordant+=int(aligned);discordant+=int(not aligned)
        total=len(ids)*(len(ids)-1)//2
        denominator=math.sqrt(total*(concordant+discordant))
        tau=(concordant-discordant)/denominator if denominator else math.nan
        row.update(covered=1.,top1=float(best-truth[selected]<=BAND),regret=best-truth[selected],rank_tau=tau)
    return row


def aggregate(queries,rows):
    condition=defaultdict(list)
    for query,row in zip(queries,rows):
        for name,value in row.items():
            if math.isfinite(value):condition[(query['solute_id'],query['condition_id'],query['task'],name)].append(value)
    solute=defaultdict(list)
    for (sid,cid,task,name),values in condition.items():solute[(sid,task,name)].append(fmean(values))
    by_solute={key:fmean(values) for key,values in solute.items()}
    population=defaultdict(list)
    for (sid,task,name),value in by_solute.items():population[(task,name)].append(value)
    macro={key:fmean(values) for key,values in population.items()}
    return by_solute,macro


def same(a,b):
    missing_a=a is None or (isinstance(a,(int,float)) and math.isnan(a))
    missing_b=b is None or (isinstance(b,(int,float)) and math.isnan(b))
    return missing_a and missing_b or (not missing_a and not missing_b and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12))


def check_solute_scores(queries,computed,saved_solutes):
    """Check complete group membership, including groups with undefined metrics."""
    expected={(q['solute_id'],q['task']) for q in queries}
    if saved_solutes.index.has_duplicates or set(saved_solutes.index)!=expected:
        raise ValueError('Wrong stored solute score cohort')
    by_solute,macro=aggregate(queries,computed)
    for (sid,task),row in saved_solutes.iterrows():
        for key in METRICS:
            if not same(row[key],by_solute.get((sid,task,key))):
                raise ValueError('Independent solute macro mismatch')
    return macro


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous independent score audits')
    registry=load_registry();analysis=json.loads(Path(registry['analysis_plan']).read_text())
    sources={__file__:sha256_file(__file__)};results=[];pending=[]
    for protocol,config in registry['protocols'].items():
        qpath=Path(config['views'])/'val_queries.jsonl';queries=read(qpath)
        manifest=pd.read_csv(config['split'],dtype={'solute_id':str});roles=manifest.set_index('solute_id').split.to_dict()
        if len({q['query_id'] for q in queries})!=len(queries) or any(q.get('split')!='val' or roles.get(q['solute_id'])!='val' for q in queries):
            raise ValueError('Only exact validation membership may be graded')
        # First explicit numerical read is validation-filtered, after authorization.
        ids=sorted(manifest.loc[manifest.split=='val','solute_id'])
        measured=pd.read_parquet(config['measurements'],columns=['record_id','logx'],filters=[('solute_id','in',ids)])
        if measured.record_id.duplicated().any() or not all(math.isfinite(x) for x in measured.logx):
            raise ValueError('Duplicate or nonfinite validation measurements')
        native=measured.set_index('record_id').logx.to_dict()
        tpath=Path(config['views'])/'val_targets.jsonl';target_rows=read(tpath)
        targets={t['query_id']:t for t in target_rows}
        if len(targets)!=len(target_rows):raise ValueError('Repeated validation target identifiers')
        if set(targets)!={q['query_id'] for q in queries}:raise ValueError('Validation target identities differ')
        for q in queries:
            target=targets[q['query_id']]
            if target['values']!={c['id']:native[c['record_id']] for c in q['candidates']}:
                raise ValueError('Validation targets differ from actual curated measurements')
            y=target['values']
            if q['task']=='pair' and target['choice']!=choice(y):raise ValueError('Incorrect stored pair truth')
            if q['task']=='rank' and target['ranking']!=sorted(y,key=lambda cid:(-y[cid],cid)):
                raise ValueError('Incorrect stored rank truth')
        for path in [qpath,tpath,Path(config['split']),Path(config['measurements'])]:sources[str(path)]=sha256_file(path)
        properties=property_names(protocol,config)
        names=properties+[evidence_name(protocol,config,mode) for mode in analysis['direct_evidence_modes']]
        names+=[n for n in analysis['llm_validation_tasks_to_carry_to_test'] if n.startswith(config['run_prefix']+protocol+'_')]
        for name in names:
            run=Path(config['runs_root'])/name;evaluation=run/'evaluation_view10000'
            metric=evaluation/'metrics.json'
            if not metric.exists():pending.append(name);continue
            is_property=name in properties
            predpath=run/('predictions_val.parquet' if is_property else 'predictions.jsonl')
            metadata_path=run/'metadata.json' if is_property else predpath.with_suffix(predpath.suffix+'.run.json')
            if json.loads(metadata_path.read_text()).get('status')!='completed':pending.append(name);continue
            official=json.loads(metric.read_text())
            expected={str(p.resolve()):sha256_file(p) for p in [qpath,tpath,predpath]}
            if {str(Path(p).resolve()):h for p,h in official['inputs'].items()}!=expected:
                raise ValueError('Evaluation numerical inputs changed')
            if is_property:
                predictions=pd.read_parquet(predpath)
                if predictions.record_id.duplicated().any() or set(predictions.record_id)!=set(native):raise ValueError('Incomplete property prediction records')
                values=predictions.set_index('record_id').pred_logx.to_dict()
                answers={}
                for q in queries:
                    p={c['id']:values.get(c['record_id'],math.nan) for c in q['candidates']}
                    if all(math.isfinite(float(x)) for x in p.values()):
                        answers[q['query_id']]={'choice':choice(p)} if q['task']=='pair' else {'ranking':sorted(p,key=lambda cid:(-p[cid],cid))}
            else:
                predictions=read(predpath)
                if len({p['query_id'] for p in predictions})!=len(predictions):raise ValueError('Repeated raw-output identifiers')
                raw={p['query_id']:p for p in predictions};answers={}
                if not set(raw)<={q['query_id'] for q in queries}:raise ValueError('Unknown raw-output IDs')
                for q in queries:
                    if q['query_id'] not in raw:continue
                    row=raw[q['query_id']];answer=decode(row['raw_output'],q)
                    if answer!=row['prediction'] or ((row['parse_error'] is None)!=(answer is not None)):
                        raise ValueError('Stored parsing disagrees with independent raw JSON/schema check')
                    answers[q['query_id']]=answer
            computed=[utility(q,targets[q['query_id']]['values'],answers.get(q['query_id'])) for q in queries]
            saved=pd.read_parquet(evaluation/'per_query.parquet').set_index('query_id')
            if len(saved)!=len(queries) or set(saved.index)!={q['query_id'] for q in queries}:raise ValueError('Wrong stored score cohort')
            largest=0.
            for q,row in zip(queries,computed):
                actual=saved.loc[q['query_id']]
                if any(actual[k]!=q[k] for k in ['solute_id','condition_id','task']):raise ValueError('Score identity differs')
                for key,value in row.items():
                    if not same(actual[key],value):raise ValueError('Independent per-query score mismatch: '+name+'/'+key)
                    if math.isfinite(value):largest=max(largest,abs(float(actual[key])-value))
            saved_solutes=pd.read_csv(evaluation/'per_solute.csv',dtype={'solute_id':str}).set_index(['solute_id','task'])
            macro=check_solute_scores(queries,computed,saved_solutes)
            if set(official['by_task'])!={q['task'] for q in queries}:raise ValueError('Wrong task summary cohort')
            for task,entry in official['by_task'].items():
                if entry['queries']!=sum(q['task']==task for q in queries):raise ValueError('Incorrect task denominator')
                if entry['solutes']!=len({q['solute_id'] for q in queries if q['task']==task}):raise ValueError('Incorrect solute denominator')
                for key in METRICS:
                    if not same(entry[key],macro.get((task,key))):raise ValueError('Independent population macro mismatch: '+key)
            result={'run':name,'protocol':protocol,'questions_checked':len(queries),'invalid_or_missing_outputs':sum(r['covered']==0 for r in computed),
                    'maximum_per_query_arithmetic_difference':largest,'raw_json_and_schema_checked':not is_property,
                    'status':'PASSED','independent_metrics':{task:{key:macro.get((task,key)) for key in METRICS} for task in ['pair','rank']}}
            results.append(result)
            for path in [predpath,metadata_path,metric,evaluation/'per_query.parquet',evaluation/'per_solute.csv']:sources[str(path)]=sha256_file(path)
            print(json.dumps({k:result[k] for k in ['run','questions_checked','invalid_or_missing_outputs','maximum_per_query_arithmetic_difference']}),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'IN_PROGRESS_INDEPENDENT_VALIDATION_GRADING_PASSED' if pending else 'PASSED_ALL_REGISTERED_VALIDATION_GRADING',
        'runs':results,'pending_runs':pending,'evidence_sha256':sources,'test_targets_read':False,'models_invoked':False,
        'scope':'Actual validation targets matched measured records; raw JSON independently decoded; every query, solute and population metric recomputed. Kendall tau-b uses concordant/discordant pair counts. Invalid/missing outputs retain penalties. This is not a final-test or efficacy-support decision.'})


if __name__=='__main__':main()
