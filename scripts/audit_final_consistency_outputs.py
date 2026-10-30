#!/usr/bin/env python3
"""Independently verify every registered frozen-test consistency comparison.

Checks chemical identity, label mappings, raw JSON, invalid penalties and all
hierarchical scores. Never invokes models or opens solubility target values.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem

from solvent_select.freeze import consistency_paths
from solvent_select.study import load_registry,queue_pairs
from solvent_select.views import sha256_file,write_json

spec=importlib.util.spec_from_file_location('independent_raw_parser',Path(__file__).with_name('audit_validation_scores.py'))
parser_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser_module)
METRICS=['both_valid','exact_agreement','top1_agreement','any_smiles_changed','changed_smiles_fraction']


def read(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    result={r['query_id']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Repeated query identity')
    return result


def chemical(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid structure')
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)


def verify_transform(base,query,mapping):
    variant=mapping['variant'];conversion=mapping['candidate_to_original']
    expected=hashlib.sha256(f"consistency-v1|{variant}|{base['query_id']}".encode()).hexdigest()[:24]
    if query['query_id']!=expected or mapping['original_query_id']!=base['query_id']:raise ValueError('Wrong transform identity')
    if any(base[k]!=query[k] for k in ['task','solute_id','split','condition_id','temperature_k','leakage_group']):
        raise ValueError('Transform changed query conditions')
    originals={c['id']:c for c in base['candidates']};current={c['id']:c for c in query['candidates']}
    if (len(current)!=len(originals) or set(conversion)!=set(current)
            or len(set(conversion.values()))!=len(originals) or set(conversion.values())!=set(originals)):
        raise ValueError('Non-bijective option mapping')
    if any(c['record_id']!=originals[conversion[cid]]['record_id'] for cid,c in current.items()):
        raise ValueError('Transform changed candidate measurement identity')
    pairs=[(base['solute_smiles'],query['solute_smiles'])]+[(originals[conversion[cid]]['smiles'],c['smiles']) for cid,c in current.items()]
    if any(chemical(a)!=chemical(b) for a,b in pairs):raise ValueError('Transform changed full chemical identity')
    changed=sum(a!=b for a,b in pairs)
    if variant=='candidate_order':
        if changed or [conversion[c['id']] for c in query['candidates']]!=list(reversed([c['id'] for c in base['candidates']])):
            raise ValueError('Incorrect candidate-order transformation')
    elif variant=='equivalent_smiles':
        if [conversion[c['id']] for c in query['candidates']]!=[c['id'] for c in base['candidates']]:
            raise ValueError('SMILES transform changed candidate order')
    else:raise ValueError('Unknown transformation')
    if changed!=mapping['changed_smiles'] or len(pairs)!=mapping['smiles_fields']:raise ValueError('Wrong representation-change counts')
    return changed,len(pairs)


def answer(row,query,conversion):
    if 'raw_output' in row:
        value=parser_module.decode(row['raw_output'],query)
        if value!=row.get('prediction'):raise ValueError('Saved raw-output parsing differs')
    else:
        value=row.get('prediction',row)
    if not isinstance(value,dict):return None
    if query['task']=='pair':
        choice=value.get('choice')
        if not isinstance(choice,str) or choice not in ['A','B','TIE']:return None
        return ('TIE',) if choice=='TIE' else (conversion[choice],)
    rank=value.get('ranking');ids={c['id'] for c in query['candidates']}
    if not isinstance(rank,list) or any(type(x) is not str for x in rank) or len(rank)!=len(ids) or set(rank)!=ids:return None
    return tuple(conversion[x] for x in rank)


def recount(records):
    conditions=defaultdict(list)
    for row in records:
        key=tuple(row[k] for k in ['variant','task','solute_id','condition_id'])
        for metric in METRICS:conditions[key+(metric,)].append(row[metric])
    solutes=defaultdict(list)
    for (variant,task,sid,cid,metric),values in conditions.items():solutes[(variant,task,sid,metric)].append(math.fsum(values)/len(values))
    means={key:math.fsum(values)/len(values) for key,values in solutes.items()}
    groups=defaultdict(list)
    for (variant,task,sid,metric),value in means.items():groups[(variant,task,metric)].append(value)
    return means,{key:math.fsum(values)/len(values) for key,values in groups.items()}


def main():
    command=argparse.ArgumentParser(description=__doc__);command.add_argument('--output',required=True);command.add_argument('--plan',required=True);command.add_argument('--state',required=True);args=command.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve earlier audits')
    plan=json.loads(Path(args.plan).read_text());state=json.loads(Path(args.state).read_text())
    if plan.get('partition')!='test' or sha256_file(args.plan)!=state['plan_sha256']:
        raise ValueError('Wrong frozen final plan')
    for file,digest in plan['release_sha256'].items():
        if sha256_file(file)!=digest:raise ValueError('Frozen release changed')
    paths=[]
    for task in plan['tasks']:
        selected=[Path(item['path']) for item in task['outputs'] if '/evaluation/metrics.json' in item['path'] and 'consistency' in task['id']]
        if not selected:continue
        receipt=state['tasks'][task['id']]
        if receipt['status']!='completed':raise ValueError('Incomplete consistency receipt')
        for file,digest in receipt['output_sha256'].items():
            if sha256_file(file)!=digest:raise ValueError('Consistency output receipt changed')
        paths.extend(selected)
    if len(paths)!=64 or len(set(paths))!=64:raise ValueError('Expected exactly64registered comparisons')
    sources={args.plan:sha256_file(args.plan),'scripts/audit_consistency_outputs.py':sha256_file('scripts/audit_consistency_outputs.py'),__file__:sha256_file(__file__),str(Path(__file__).with_name('audit_validation_scores.py')):sha256_file(Path(__file__).with_name('audit_validation_scores.py'))}
    results=[];cache={}
    for path in paths:
        metric=json.loads(path.read_text());inputs=metric['inputs']
        if metric.get('partition')!='test' or metric.get('queries')!=1024 or metric.get('base_queries')!=512:
            raise ValueError('Wrong held-out consistency cohort')
        for file,digest in inputs.items():
            if sha256_file(file)!=digest:raise ValueError('Consistency source changed')
            sources[file]=digest
        manifest=next(Path(p) for p in inputs if Path(p).name=='manifest.json');folder=manifest.parent
        if folder not in cache:
            info=json.loads(manifest.read_text())
            if info['split']!='test':raise ValueError('Only frozen-test metadata is authorized')
            roles=pd.read_csv(info['split_manifest'],dtype={'solute_id':str}).set_index('solute_id').split.to_dict()
            if sha256_file(info['split_manifest'])!=info['split_sha256']:raise ValueError('Changed role manifest')
            base=read(folder/'base_queries.jsonl');variants=read(folder/'transformed_queries.jsonl');mapping=read(folder/'mapping.jsonl')
            if len(base)!=512 or len(variants)!=1024 or set(mapping)!=set(variants):raise ValueError('Wrong transformation denominator')
            for file,digest in info['files'].items():
                if sha256_file(folder/file)!=digest:raise ValueError('Transform file changed')
                sources[str(folder/file)]=digest
            for query in list(base.values())+list(variants.values()):
                if query.get('split')!='test' or roles.get(query['solute_id'])!='test':raise ValueError('Non-test query')
            changes={qid:verify_transform(base[mapping[qid]['original_query_id']],query,mapping[qid]) for qid,query in variants.items()}
            cache[folder]=(base,variants,mapping,changes)
        base,variants,mappings,changes=cache[folder]
        predictions=[Path(p) for p in inputs if Path(p)!=manifest]
        if len(predictions)!=2:raise ValueError('Unknown prediction-source specification')
        first,second=[read(p) for p in predictions]
        # Production records inputs in base,transformed order. Verify by actualIDs.
        if not set(base)<=set(first) or set(second)!=set(variants):raise ValueError('Incomplete raw predictions')
        records=[]
        for qid,query in variants.items():
            mapping=mappings[qid];original=base[mapping['original_query_id']]
            a=answer(first[original['query_id']],original,{c['id']:c['id'] for c in original['candidates']})
            b=answer(second[qid],query,mapping['candidate_to_original']);valid=a is not None and b is not None
            changed,fields=changes[qid]
            records.append({'query_id':qid,'original_query_id':original['query_id'],'variant':mapping['variant'],
                'task':query['task'],'solute_id':query['solute_id'],'condition_id':query['condition_id'],
                'both_valid':float(valid),'exact_agreement':float(valid and a==b),'top1_agreement':float(valid and a[0]==b[0]),
                'any_smiles_changed':float(changed>0),'changed_smiles_fraction':changed/fields})
        saved_path=path.parent/'per_query.parquet';saved=pd.read_parquet(saved_path).set_index('query_id')
        if saved.index.has_duplicates or set(saved.index)!=set(variants):raise ValueError('Wrong saved score cohort')
        largest=0.
        for row in records:
            previous=saved.loc[row['query_id']]
            for key,value in row.items():
                if key=='query_id':continue
                if key in METRICS:
                    if not math.isclose(value,previous[key],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Per-query consistency score differs')
                    largest=max(largest,abs(value-previous[key]))
                elif value!=previous[key]:raise ValueError('Score identity differs')
        per_solute,population=recount(records)
        solute_path=path.parent/'per_solute.csv';stored=pd.read_csv(solute_path,dtype={'solute_id':str}).set_index(['variant','task','solute_id'])
        if stored.index.has_duplicates or set(stored.index)!={key[:3] for key in per_solute}:raise ValueError('Wrong solute cohort')
        for key,row in stored.iterrows():
            for column in METRICS:
                if not math.isclose(row[column],per_solute[key+(column,)],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Solute consistency differs')
        expected_groups={variant+'/'+task for variant,task,metric_name in population}
        if set(metric['by_variant_task'])!=expected_groups:raise ValueError('Wrong variant/task groups')
        for group,row in metric['by_variant_task'].items():
            variant,task=group.split('/')
            if row['solutes']!=len({sid for v,t,sid,m in per_solute if v==variant and t==task}):raise ValueError('Wrong support')
            for column in METRICS:
                if not math.isclose(row[column],population[(variant,task,column)],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Population consistency differs')
        for p in [path,saved_path,solute_path]:sources[str(p)]=sha256_file(p)
        results.append({'comparison':str(path),'queries_checked':1024,'maximum_per_query_discrepancy':largest,'status':'PASSED'})
        print(str(path),'PASSED',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'PASSED_ALL_REGISTERED_FINAL_CONSISTENCY_COMPARISONS','comparisons':results,
        'transformed_question_method_rows':sum(r['queries_checked'] for r in results),'target_values_read':False,'test_queries_read':True,
        'models_invoked':False,'evidence_sha256':sources,'scope':'Independent chemical/mapping/raw-answer/score audit; observations repeat molecules and are not independent sample counts.'})


if __name__=='__main__':main()
