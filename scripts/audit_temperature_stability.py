#!/usr/bin/env python3
"""Recount every training temperature diagnostic using independent arithmetic."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solvent_select.study import load_registry
from solvent_select.temperature_stability import METRICS, read_training
from solvent_select.views import sha256_file, write_json


def weights(frame):
    edges=frame.groupby(['solute_id','source_doi','series_id']).series_id.transform('size').to_numpy(float)
    series=frame.groupby(['solute_id','source_doi']).series_id.transform('nunique').to_numpy(float)
    sources=frame.groupby('solute_id').source_doi.transform('nunique').to_numpy(float)
    return 1/(edges*series*sources*frame.solute_id.nunique())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source,output=Path(args.source),Path(args.output)
    if output.exists():raise FileExistsError('Preserve earlier independent arithmetic audits')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['status']!='COMPLETED_TRAINING_ONLY_TEMPERATURE_DESCRIPTION':raise ValueError('Unknown analysis')
    for path,expected in manifest['inputs'].items():
        if sha256_file(path)!=expected:raise ValueError('Analysis input changed: '+path)
    for name,expected in manifest['outputs'].items():
        if sha256_file(source/name)!=expected:raise ValueError('Analysis output changed: '+name)
    cfg_path=next(p for p in manifest['inputs'] if Path(p).name=='temperature_stability_v1.json')
    cfg=json.loads(Path(cfg_path).read_text())
    summary=pd.read_csv(source/'summary.csv')
    if summary.duplicated(['protocol','relation','gap_bin','metric']).any():raise ValueError('Duplicate summary estimates')
    summary=summary.set_index(['protocol','relation','gap_bin','metric'])
    audits=[]
    for protocol,layout in load_registry()['protocols'].items():
        training=read_training(layout)
        native=training.set_index('record_id')
        times={key:set(part.temperature_k) for key,part in training.groupby(['solute_id','source_doi','solvent_id'])}
        edges=pd.read_parquet(source/f'{protocol}_temperature_comparisons.parquet')
        ys={}
        for level,temp in [('low','temperature_low_k'),('high','temperature_high_k')]:
            for option in ['a','b']:
                measured=native.loc[edges[f'{level}_record_{option}']].reset_index()
                for name,column in [('solute_id','solute_id'),('source_doi','source_doi'),
                                    ('solvent_id',f'solvent_{option}_id'),('temperature_k',temp)]:
                    if not np.array_equal(measured[name].to_numpy(),edges[column].to_numpy()):
                        raise ValueError('Original measurement identity/conditions do not match recorded transition')
                ys[(level,option)]=measured.logx.to_numpy()
        lo=ys[('low','a')]-ys[('low','b')];hi=ys[('high','a')]-ys[('high','b')]
        assert np.array_equal(lo,edges.delta_low_dex) and np.array_equal(hi,edges.delta_high_dex)
        band=cfg['tie_dex']+cfg['tie_float_epsilon']
        a=np.where(abs(lo)<=band,0,np.where(lo>0,1,-1))
        b=np.where(abs(hi)<=band,0,np.where(hi>0,1,-1))
        calculated={'label_changed':a!=b,'decisive_reversal':a*b==-1,
            'near_tie_transition':(a==0)!=(b==0),'both_near_ties':(a==0)&(b==0),
            'both_decisive':(a!=0)&(b!=0),'raw_sign_reversal':lo*hi<0,
            'absolute_delta_change_dex':abs(hi-lo),'delta_change_exceeds_tie_band':abs(hi-lo)>band,
            'reversal_given_both_decisive':np.where((a!=0)&(b!=0),a*b==-1,np.nan)}
        for key,values in calculated.items():
            if not np.allclose(edges[key],values,rtol=0,atol=0,equal_nan=True):raise ValueError('Independent arithmetic mismatch: '+key)
        owner=['solute_id','source_doi','solvent_a_id','solvent_b_id']
        if edges.groupby(owner).series_id.nunique().max()!=1:raise ValueError('Duplicated owner series')
        seen=set()
        for series,group in edges.groupby('series_id',sort=False):
            first=group.iloc[0];sid,doi,sa,sb=(first[k] for k in owner)
            if any(group[k].nunique()!=1 for k in owner):raise ValueError('Mixed source or identities in one series')
            common=sorted(times[(sid,doi,sa)]&times[(sid,doi,sb)])
            expected=set(combinations(common,2));actual=set(zip(group.temperature_low_k,group.temperature_high_k))
            if len(group)!=len(expected) or actual!=expected:raise ValueError('Missing, duplicate or invented temperature comparison')
            adjacent=set(zip(common[:-1],common[1:]))
            if any(flag!=((low,high) in adjacent) for low,high,flag in
                   zip(group.temperature_low_k,group.temperature_high_k,group.adjacent_shared_temperatures)):
                raise ValueError('Incorrect shared-temperature adjacency')
            seen.add((sid,doi,sa,sb))
        expected_owners=set()
        for (sid,doi),group in training.groupby(['solute_id','source_doi']):
            for sa,sb in combinations(sorted(group.solvent_id.unique()),2):
                if len(times[(sid,doi,sa)]&times[(sid,doi,sb)])>=2:expected_owners.add((sid,doi,sa,sb))
        if seen!=expected_owners:raise ValueError('Eligible training series were omitted or added')
        largest=0.;estimates=0
        for relation in cfg['temperature_relations']:
            population=edges if relation=='all_observed_temperature_pairs' else edges[edges.adjacent_shared_temperatures]
            for gap in ['ALL',*cfg['temperature_gap_labels']]:
                part=population if gap=='ALL' else population[population.gap_bin==gap]
                for conditional in [False,True]:
                    metrics=['reversal_given_both_decisive'] if conditional else METRICS[:-1]
                    supported=part[part.reversal_given_both_decisive.notna()] if conditional else part
                    if len(supported):
                        w=weights(supported)
                        assert np.isclose(w.sum(),1.,rtol=1e-12,atol=1e-12)
                        independent=(supported[metrics].to_numpy()*w[:,None]).sum(axis=0)
                    else:independent=[np.nan]*len(metrics)
                    for key,value in zip(metrics,independent):
                        observed=summary.loc[(protocol,relation,gap,key)]
                        if observed.support_solutes!=supported.solute_id.nunique():raise ValueError('Incorrect independent-solute denominator')
                        if not np.isclose(observed.value,value,rtol=1e-12,atol=1e-12,equal_nan=True):
                            raise ValueError('Independent inverse-count weighting disagrees with hierarchical means')
                        if np.isfinite(value):largest=max(largest,abs(value-observed.value))
                        estimates+=1
        audits.append({'protocol':protocol,'temperature_comparisons_checked':len(edges),
            'native_measurement_references_checked':4*len(edges),'complete_source_solvent_series':len(seen),
            'macro_estimates_recounted':estimates,'maximum_macro_arithmetic_difference':largest})
        print(json.dumps(audits[-1]),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'PASSED_FULL_TRAINING_TEMPERATURE_ARITHMETIC_AND_PROVENANCE',
        'audits':audits,'source_manifest':str(source/'manifest.json'),'source_manifest_sha256':sha256_file(source/'manifest.json'),
        'code_sha256':sha256_file(__file__),'heldout_labels_read':False,
        'scope':'Every original four-record condition and all source-series temperature combinations checked. '
                'Labels recomputed with vectorized sign/band arithmetic; all macro point estimates independently recomputed '
                'with inverse edge/series/source/solute counts. Confidence intervals have separate fixed-seed full reconstruction evidence.'})


if __name__=='__main__':main()
