"""Fold-specific source quarantine; strict validation/test bytes are untouched."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import pandas as pd

from .llm import write_json
from .views import sha256_file


def build(root='.',destination='.'):
    root,destination=Path(root),Path(destination)
    policy_path=root/'configs/strict_quarantine_v1.json'
    policy=json.loads(policy_path.read_text())
    evidence=root/policy['source_summary']
    if sha256_file(evidence)!=policy['source_summary_sha256']:
        raise ValueError('Strict training source evidence changed')
    source_protocol=json.loads((root/policy['source_protocol']).read_text())
    if sha256_file(root/policy['source_protocol'])!=policy['source_protocol_sha256']:
        raise ValueError('Strict training source protocol changed')
    summary=pd.read_csv(evidence)
    flagged=set(summary.loc[summary.flagged_records>0,'source_doi'])
    if flagged!=set(policy['source_dois']) or policy['threshold_dex']!=.3:
        raise ValueError('Quarantine must include exactly all sources flagged by the original tolerance')
    for path,key in [('measurement_path','measurement_sha256'),('split_path','split_sha256')]:
        if sha256_file(root/source_protocol[path])!=source_protocol[key]:
            raise ValueError('Original strict inputs changed')
    processed=destination/'data/processed/strictqa'
    targets=destination/'data/targets/strictqa/scaffold'
    splits=destination/'data/splits/strictqa'
    audit=destination/'artifacts/audit/strictqa'
    if any(p.exists() for p in [processed,targets,splits]) or (audit/'data_audit.json').exists():
        raise FileExistsError('Never overwrite an existing strictqa version')
    original=pd.read_parquet(root/policy['base_measurements'])
    old_split=pd.read_csv(root/policy['base_splits'],dtype={'solute_id':str})
    roles=old_split.set_index('solute_id').split.to_dict()
    removed=original[original.source_doi.isin(flagged)].copy()
    if set(removed.solute_id.map(roles))!={'train'}:
        raise ValueError('Fold-specific quarantine touches validation/test')
    kept=original[~original.record_id.isin(removed.record_id)].copy()
    new_split=old_split[old_split.solute_id.isin(kept.solute_id)].copy()
    for partition in ['val','test']:
        if set(old_split.loc[old_split.split==partition,'solute_id'])!=set(new_split.loc[new_split.split==partition,'solute_id']):
            raise ValueError('Held-out identities changed')
    assigned=kept.merge(new_split[['solute_id','split']],on='solute_id',validate='many_to_one')
    for group in ['leakage_group','scaffold_group']:
        if assigned.groupby(group).split.nunique().max()!=1:
            raise ValueError('Group crosses partitions after quarantine')
    for folder in [processed,targets,splits,audit]:folder.mkdir(parents=True,exist_ok=True)
    kept.to_parquet(processed/'measurements_scaffold.parquet',index=False)
    new_split.to_csv(splits/'scaffold.csv',index=False)
    shutil.copyfile(root/policy['base_exclusions'],splits/'scaffold_excluded.csv')
    removed[['record_id','solute_id','solute_smiles','solvent_smiles','source_doi','temperature_k']].to_csv(audit/'quarantined_records.csv',index=False)
    counts={};preserved={};inputs={str(policy_path):sha256_file(policy_path)}
    live_records=set(kept.record_id)
    for partition in ['train','val','test']:
        query_source=root/f'data/processed/scaffold/{partition}_queries.jsonl'
        target_source=root/f'data/targets/scaffold/{partition}.jsonl'
        query_out=processed/f'scaffold/{partition}_queries.jsonl'
        target_out=targets/(partition+'.jsonl')
        query_out.parent.mkdir(exist_ok=True)
        inputs[str(query_source)]=sha256_file(query_source);inputs[str(target_source)]=sha256_file(target_source)
        if partition=='train':
            selected=set();query_counts=Counter()
            with query_source.open() as source,query_out.open('w') as output:
                for line in source:
                    query=json.loads(line)
                    present=[c['record_id'] in live_records for c in query['candidates']]
                    if any(present) and not all(present):
                        raise ValueError('A source quarantine would create a partial candidate block')
                    if all(present):
                        selected.add(query['query_id']);query_counts[query['task']]+=1;output.write(line)
            copied=set()
            with target_source.open() as source,target_out.open('w') as output:
                for line in source:
                    qid=json.loads(line)['query_id']
                    if qid in selected:copied.add(qid);output.write(line)
            if copied!=selected:raise ValueError('Training query/target mismatch')
        else:
            shutil.copyfile(query_source,query_out);shutil.copyfile(target_source,target_out)
            query_counts=Counter(json.loads(line)['task'] for line in query_source.open())
            for source,out in [(query_source,query_out),(target_source,target_out)]:
                if sha256_file(source)!=sha256_file(out):raise ValueError('Held-out bytes changed')
                preserved[str(source)]=sha256_file(source)
        ids=set(new_split.loc[new_split.split==partition,'solute_id'])
        counts[partition]={'solutes':len(ids),'records':int(kept.solute_id.isin(ids).sum()),'queries':dict(query_counts)}
    result={'status':'STRICT_TRAINING_ONLY_SOURCE_QUARANTINE_APPLIED','dataset_version':'strictqa',
            'target':'log10_mole_fraction','threshold_dex':.3,'base_strict_measurements':len(original),
            'final_measurements':len(kept),'source_quarantined_records':len(removed),'source_quarantined_dois':len(flagged),
            'removed_training_identities':len(old_split)-len(new_split),'splits':{'scaffold':counts},
            'heldout_query_target_bytes_preserved':preserved,'all_surviving_roles_inherited':True,
            'validation_or_test_labels_used_for_quarantine':False,'numeric_labels_replaced':False,
            'historical_disclosure':'The previous primary-training audit examined some strict-development labels; those global-quarantine strict results are superseded. This amendment uses only strict-training-only source evidence and restores every original strict validation/test record.',
            'inputs':inputs,'code_sha256':sha256_file(__file__)}
    write_json(audit/'data_audit.json',result)
    outputs={str(p.relative_to(destination)):sha256_file(p) for folder in [processed,targets,splits] for p in sorted(folder.rglob('*')) if p.is_file()}
    write_json(audit/'data_manifest.json',{'status':result['status'],'outputs':outputs,'data_audit_sha256':sha256_file(audit/'data_audit.json')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--destination',default='.')
    args=parser.parse_args()
    result=build(args.root,args.destination)
    print(json.dumps({key:result[key] for key in ['status','source_quarantined_records','removed_training_identities','splits']},indent=2))


if __name__=='__main__':main()
