#!/usr/bin/env python3
"""Rebuild question identities from metadata and test prompt field isolation.

Public test metadata are audited, but no target file is opened and no model
is invoked. Input sentinels are synthetic field-isolation checks, not results.
"""
import argparse
from collections import Counter
import copy
import hashlib
from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from solvent_select.llm import build_messages
from solvent_select.study import load_registry
from solvent_select.views import read_query_metadata,select_train_ids,select_eval_ids,sha256_file,write_json


def identity_hash(*parts):
    return hashlib.sha256(json.dumps(parts,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()[:24]


def query_digest(query):
    return hashlib.sha256(json.dumps(query,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def expected_questions(metadata,roles):
    """Independent balanced slicing and explicit metadata-only hash construction."""
    expected={}
    for (sid,doi,temp),condition in metadata.groupby(['solute_id','source_doi','temperature_k'],sort=True):
        if len(condition)<2:continue
        records=condition.to_dict('records')
        ordered=sorted(records,key=lambda r:identity_hash('candidate-block',r['solvent_id']))
        count=(len(ordered)+7)//8
        size,remainder=divmod(len(ordered),count)
        position=0
        for index in range(count):
            length=size+int(index<remainder);block=ordered[position:position+length];position+=length
            block_id=identity_hash('condition',sid,doi,float(temp),sorted(r['solvent_id'] for r in block))
            base={'solute_id':sid,'solute_smiles':block[0]['solute_smiles'],'leakage_group':block[0]['leakage_group'],
                  'temperature_k':float(temp),'condition_id':identity_hash('physical_condition',sid,doi,float(temp)),
                  'block_id':block_id,'split':roles[sid]}
            def store(qid,task,entries):
                if qid in expected:raise ValueError('Duplicate metadata-derived question ID')
                query={**base,'query_id':qid,'task':task,'candidates':[
                    {'id':name,'smiles':row['solvent_smiles'],'record_id':row['record_id']} for name,row in entries]}
                expected[qid]=(roles[sid],query_digest(query))
            for pair in combinations(block,2):
                a,b=sorted(pair,key=lambda r:identity_hash('pair-order',block_id,r['solvent_id']))
                store(identity_hash('pair',block_id,a['solvent_id'],b['solvent_id']),'pair',[('A',a),('B',b)])
            if len(block)>=3:
                ranked=sorted(block,key=lambda r:identity_hash('rank-order',block_id,r['solvent_id']))
                store(identity_hash('rank',block_id),'rank',[(f'S{i}',row) for i,row in enumerate(ranked)])
    return expected


def assert_prompt_isolation(query):
    before=build_messages(query)
    injected=copy.deepcopy(query)
    marker='FORBIDDEN_SYNTHETIC_TARGET_SENTINEL_92c7'
    injected.update(values={'A':marker},choice=marker,ranking=[marker],answer=marker,logx=marker,
                    logS=marker,delta_logS=marker,source_doi=marker)
    for candidate in injected['candidates']:
        candidate.update(logx=marker,solubility=marker,answer=marker,source_doi=marker)
    after=build_messages(injected)
    if before!=after or marker in json.dumps(after):raise ValueError('A forbidden target/source field entered the user message')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve earlier label-isolation audits')
    registry=load_registry();results=[]
    sources={__file__:sha256_file(__file__),'configs/study_registry_dualqa_v2.json':sha256_file('configs/study_registry_dualqa_v2.json')}
    columns=['record_id','solute_id','solute_smiles','solvent_id','solvent_smiles','source_doi',
             'temperature_k','leakage_group','scaffold_group']
    for protocol,config in registry['protocols'].items():
        metadata=pd.read_parquet(config['measurements'],columns=columns)
        manifest=pd.read_csv(config['split'],dtype={'solute_id':str})
        roles=manifest.set_index('solute_id').split.to_dict()
        if metadata.record_id.duplicated().any() or set(metadata.solute_id)!=set(roles):raise ValueError('Invalid measurement/manifest identity universe')
        joined=metadata[['solute_id','leakage_group','scaffold_group']].drop_duplicates().merge(manifest,on='solute_id',validate='one_to_one',suffixes=('_data','_split'))
        for key in ['leakage_group','scaffold_group']:
            if not (joined[key+'_data']==joined[key+'_split']).all():raise ValueError('Structure groups differ between manifest and measurements')
        if joined.groupby('leakage_group_data').split.nunique().max()!=1:raise ValueError('Parent group crosses partitions')
        if protocol=='scaffold' and joined.groupby('scaffold_group_data').split.nunique().max()!=1:raise ValueError('Scaffold group crosses partitions')
        original=pd.read_csv('data/splits/'+protocol+'.csv',dtype={'solute_id':str}).set_index('solute_id').split.to_dict()
        if any(original[sid]!=role for sid,role in roles.items()):raise ValueError('Surviving roles were changed by source curation')
        for row in metadata.itertuples(index=False):
            if (row.solute_id!=identity_hash('solute',row.solute_smiles)
                or row.solvent_id!=identity_hash('solvent',row.solvent_smiles)
                or row.record_id!=identity_hash('measurement',row.solute_id,row.solvent_id,float(row.temperature_k),row.source_doi)):
                raise ValueError('Measurement identity is not determined solely by chemical/condition metadata')
        expected=expected_questions(metadata,roles);partitions=[]
        for split in ['train','val','test']:
            full=Path(config['processed'])/f'{split}_queries.jsonl'
            seen=set()
            with full.open() as stream:
                for line in stream:
                    query=json.loads(line);qid=query['query_id']
                    if qid in seen or expected.get(qid)!=(split,query_digest(query)):
                        raise ValueError('Actual full question differs from metadata-only reconstruction')
                    seen.add(qid)
            if seen!={qid for qid,(role,_) in expected.items() if role==split}:raise ValueError('Missing/added full questions')
            qmeta=read_query_metadata(full,split)
            selected=select_train_ids(qmeta,10000,20261025) if split=='train' else select_eval_ids(qmeta,32,20261025)
            view=Path(config['views'])/f'{split}_queries.jsonl'
            actual=[]
            with view.open() as stream:
                for line in stream:
                    query=json.loads(line);actual.append(query['query_id'])
                    if expected.get(query['query_id'])!=(split,query_digest(query)):raise ValueError('Question view changed public/internal query metadata')
                    assert_prompt_isolation(query)
            if actual!=selected:raise ValueError('Published question-view selection cannot be replayed without targets')
            partitions.append({'split':split,'full_questions':len(seen),'view_questions':len(actual),
                               'metadata_only_view_replay_exact':True,'prompt_sentinel_checks':len(actual)})
            for path in [full,view]:sources[str(path)]=sha256_file(path)
        for path in [config['measurements'],config['split'],'data/splits/'+protocol+'.csv']:
            sources[str(path)]=sha256_file(path)
        results.append({'protocol':protocol,'measurements':len(metadata),'solute_identities':metadata.solute_id.nunique(),
                        'metadata_derived_questions':len(expected),'partitions':partitions})
        print(json.dumps(results[-1]),flush=True)
    for path in ['src/solvent_select/data.py','src/solvent_select/views.py','src/solvent_select/retrieval.py','src/solvent_select/llm.py']:
        sources[path]=sha256_file(path)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'PASSED_CURRENT_METADATA_QUERY_AND_PROMPT_LABEL_ISOLATION','protocols':results,
        'measurement_columns_read':columns,'target_files_read':[],'test_public_metadata_read':True,
        'test_predictions_generated':False,'model_invoked':False,'evidence_sha256':sources,
        'scope':'All question IDs, blocks, candidate orders and roles independently reconstructed from metadata. '
                'View selectors replayed with query metadata only. Synthetic forbidden-field injection cannot change message serialization on any published query view. '
                'Training retrieval cards have separate membership/budget audits; foundation pretraining contamination cannot be excluded.'})


if __name__=='__main__':main()
