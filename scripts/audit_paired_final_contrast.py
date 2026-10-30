#!/usr/bin/env python3
"""Independently recount a three-seed frozen-test contrast from query scores.

No imports of production comparison, aggregation or bootstrap implementations.
Bootstrap uses multinomial counts; percentile interpolation is implemented here.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from solvent_select.study import load_registry
from solvent_select.views import sha256_file,write_json


def hierarchical_values(frame,task,metric):
    conditions=defaultdict(list)
    for row in frame[frame.task==task].to_dict('records'):
        value=float(row[metric])
        if not math.isfinite(value):raise ValueError('Nonfinite required score')
        conditions[(row['solute_id'],row['condition_id'])].append(value)
    solutes=defaultdict(list)
    for (sid,_),values in conditions.items():solutes[sid].append(math.fsum(values)/len(values))
    return {sid:math.fsum(values)/len(values) for sid,values in solutes.items()}


def paired_values(first,second,task,metric):
    if not first or len(first)!=len(second):raise ValueError('Mismatched seed counts')
    a=[hierarchical_values(frame,task,metric) for frame in first]
    b=[hierarchical_values(frame,task,metric) for frame in second]
    ids=sorted(a[0])
    if not ids or any(set(x)!=set(ids) for x in a+b):raise ValueError('Mismatched solute support')
    delta={sid:math.fsum(x[sid]-y[sid] for x,y in zip(a,b))/len(a) for sid in ids}
    individual=[math.fsum(x[sid]-y[sid] for sid in ids)/len(ids) for x,y in zip(a,b)]
    return delta,individual


def percentile(sorted_values,fraction):
    position=(len(sorted_values)-1)*fraction
    lo,hi=math.floor(position),math.ceil(position)
    return float(sorted_values[lo]+(position-lo)*(sorted_values[hi]-sorted_values[lo]))


def bootstrap(values,repetitions=10000,seed=20261025):
    if not values or not all(math.isfinite(x) for x in values):raise ValueError('Invalid bootstrap observations')
    n=len(values);generator=np.random.default_rng(seed)
    draws=generator.integers(n,size=(repetitions,n))
    counts=np.asarray([np.bincount(row,minlength=n) for row in draws])
    estimates=np.sort((counts@np.asarray(values,dtype=float))/n)
    return math.fsum(values)/n,[percentile(estimates,.025),percentile(estimates,.975)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True)
    parser.add_argument('--grade-audit',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous contrast audits')
    comparison=json.loads(Path(args.comparison).read_text());grade=json.loads(Path(args.grade_audit).read_text())
    if (comparison.get('partition')!='test' or comparison.get('status')!='FROZEN_TEST_COMPARISON'
            or comparison.get('is_primary_comparison') is not True or comparison.get('seed_count')!=3
            or grade.get('test_targets_read') is not True or grade.get('models_invoked') is not False
            or any(row.get('status')!='PASSED' for row in grade['runs'])):
        raise ValueError('Only independently graded, three-seed frozen-test primary comparisons')
    registry=load_registry();config=registry['protocols'][comparison['split_family']]
    plan=json.loads(Path(registry['analysis_plan']).read_text())
    if plan['seeds']!=[17,42,2026] or plan['bootstrap']!={
        'unit':'solute','paired':True,'repetitions':10000,'seed':20261025,
        'interval':'percentile95%; conditional on fixed checkpoints and measured cohort; not universal experimental noise or chemical-space uncertainty'}:
        raise ValueError('Analysis bootstrap or seed specification changed')
    qpath=Path(config['views'])/'test_queries.jsonl'
    queries=[json.loads(x) for x in qpath.read_text().splitlines() if x.strip()]
    roles=pd.read_csv(config['split'],dtype={'solute_id':str}).set_index('solute_id').split.to_dict()
    if any(q.get('split')!='test' or roles.get(q['solute_id'])!='test' for q in queries):raise ValueError('Non-test identity')
    if sha256_file(qpath)!=comparison['cohort_sha256']:raise ValueError('Query cohort differs')
    completed={row['run'] for row in grade['runs']}
    sources={__file__:sha256_file(__file__),args.comparison:sha256_file(args.comparison),
             args.grade_audit:sha256_file(args.grade_audit),str(qpath):sha256_file(qpath),
             config['split']:sha256_file(config['split']),registry['analysis_plan']:sha256_file(registry['analysis_plan'])}
    for path,expected in comparison['inputs'].items():
        if sha256_file(path)!=expected:raise ValueError('Comparison input changed')
        sources[path]=expected
    first,second=[],[]
    for mode,key,frames in [('condition','first_evaluations',first),('structure','second_evaluations',second)]:
        for seed,path in zip(plan['seeds'],comparison[key],strict=True):
            expected=config['run_prefix']+comparison['split_family']+f'_qwen_rag_{seed}_{mode}_test'
            folder=Path(path)
            if folder.parent.name!=expected or expected not in completed:raise ValueError('Wrong or ungraded seed/mode')
            p=folder/'per_query.parquet'
            if grade['evidence_sha256'].get(str(p))!=sha256_file(p):raise ValueError('Per-query scores differ from independent grading')
            frame=pd.read_parquet(p)
            if frame.query_id.duplicated().any() or set(frame.query_id)!={q['query_id'] for q in queries}:
                raise ValueError('Wrong scored question cohort')
            frames.append(frame)
    checked=[]
    for row in comparison['comparisons']:
        task,metric=row['task'],row['metric']
        delta,individual=paired_values(first,second,task,metric)
        point,interval=bootstrap(list(delta.values()))
        actual=[point,*interval,*individual]
        expected=[row['difference_first_minus_second'],*row['ci95'],*row['individual_seed_differences']]
        if row['solutes']!=len(delta) or not all(math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12) for a,b in zip(actual,expected,strict=True)):
            raise ValueError('Independent paired estimate or interval differs')
        checked.append({'task':task,'metric':metric,'solutes':len(delta),'difference_first_minus_second':point,
                        'ci95':interval,'individual_seed_differences':individual,
                        'maximum_absolute_arithmetic_discrepancy':max(abs(a-b) for a,b in zip(actual,expected))})
    pair=next(r for r in checked if r['metric']=='pair_accuracy')
    regret=next(r for r in checked if r['metric']=='regret')
    hypothesis=plan['primary_contrast']
    supported=(pair['difference_first_minus_second']>=hypothesis['minimum_pair_accuracy_gain'] and pair['ci95'][0]>0
               and regret['difference_first_minus_second']<=hypothesis['maximum_rank_regret_difference_dex'] and regret['ci95'][1]<0
               and all(x>0 for x in pair['individual_seed_differences']) and all(x<0 for x in regret['individual_seed_differences']))
    if bool(supported)!=comparison['hypothesis_joint_criteria_supported']:raise ValueError('Independent hypothesis assessment differs')
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'PASSED_INDEPENDENT_FINAL_CONTRAST','comparisons':checked,
        'hypothesis_joint_criteria_supported':bool(supported),'seed_count':3,'bootstrap_repetitions':10000,
        'bootstrap_seed':20261025,'bootstrap_unit':'solute','bootstrap_algorithm':'Count repeated sampled solutes with multinomial counts; independently interpolate sorted percentile endpoints.',
        'test_targets_read':False,'models_invoked':False,'evidence_sha256':sources,
        'scope':'Independent arithmetic audit of fixed-checkpoint frozen-test estimates; no causal, equivalence or publication claim.'})


if __name__=='__main__':main()
