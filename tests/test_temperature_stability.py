"""Hand-computed temperature-series fixtures, not experimental study results."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import solvent_select.temperature_stability as stability


def config():
    return json.loads((Path(__file__).parents[1]/'configs/temperature_stability_v1.json').read_text())


def measured():
    rows=[]
    for t,a,b in [(280.,-1.,-2.),(290.,-1.2,-1.25),(300.,-1.5,-1.1)]:
        for solvent,value in [('a',a),('b',b)]:
            rows.append({'record_id':f'{solvent}-{t}','solute_id':'s','solvent_id':solvent,
                         'source_doi':'10.1234/example','temperature_k':t,'logx':value})
    return pd.DataFrame(rows)


def test_all_observed_pairs_capture_a_reversal_separated_by_a_near_tie():
    frame,counts=stability.transitions(measured(),config())
    assert len(frame)==3
    assert counts['paired_temperature_series']==1
    assert frame.decisive_reversal.sum()==1
    assert frame.near_tie_transition.sum()==2
    assert frame.label_changed.sum()==3
    adjacent=frame[frame.adjacent_shared_temperatures]
    assert len(adjacent)==2
    assert adjacent.decisive_reversal.sum()==0
    assert frame[frame.decisive_reversal==1].iloc[0].gap_k==20.
    assert np.allclose(frame.label_changed,frame.decisive_reversal+frame.near_tie_transition)


def test_shared_temperature_adjacency_does_not_interpolate_a_missing_solvent_measurement():
    source=measured()
    source=source[~((source.solvent_id=='b')&(source.temperature_k==290.))]
    frame,_=stability.transitions(source,config())
    assert len(frame)==1
    assert frame.iloc[0].gap_k==20.
    assert bool(frame.iloc[0].adjacent_shared_temperatures)
    assert 'b-290.0' not in frame.astype(str).to_numpy()


def test_different_sources_and_full_solute_identities_are_never_pooled():
    source=measured()
    source.loc[source.solvent_id=='b','source_doi']='10.1234/another'
    frame,_=stability.transitions(source,config())
    assert frame.empty
    source=measured()
    source.loc[source.solvent_id=='b','solute_id']='different_salt_or_stereoisomer'
    assert stability.transitions(source,config())[0].empty


def test_duplicate_measured_conditions_are_rejected_without_new_label_aggregation():
    source=measured()
    with pytest.raises(ValueError,match='Duplicate measured condition'):
        stability.transitions(pd.concat([source,source.iloc[[0]]]),config())


def test_near_tie_and_gap_boundaries_follow_fixed_rules():
    cfg=config()
    assert stability.gap_label(5.,cfg)=='(0,5]'
    assert stability.gap_label(10.,cfg)=='(5,10]'
    assert stability.gap_label(20.,cfg)=='(10,20]'
    assert stability.gap_label(20.01,cfg)=='>20'
    source=measured()
    source.loc[(source.solvent_id=='b')&(source.temperature_k==290.),'logx']=-1.3
    frame,_=stability.transitions(source,cfg)
    assert set(frame.loc[frame.temperature_high_k==290.,'label_high'])=={'TIE'}


def test_training_predicate_is_applied_before_numerical_labels_are_read(tmp_path,monkeypatch):
    manifest=tmp_path/'split.csv'
    manifest.write_text('solute_id,split\ns,train\nv,val\nt,test\n')
    calls=[]
    def fake(path,columns,filters=None):
        calls.append((columns,filters))
        if 'logx' not in columns:
            return pd.DataFrame({'solute_id':['s','v','t'],'leakage_group':['gs','gv','gt']})
        assert filters==[('solute_id','in',['s'])]
        return measured()[columns]
    monkeypatch.setattr(stability.pd,'read_parquet',fake)
    frame=stability.read_training({'split':manifest,'measurements':'no_unfiltered_label_read.parquet'})
    assert set(frame.solute_id)=={'s'}
    assert len(calls)==2


def test_series_and_solute_macro_means_are_not_dominated_by_dense_temperature_grids():
    # Solute1 has a densely sampled no-reversal series and a sparse reversal
    # series: its series-balanced rate is1/2. Solute2 has rate1, so macro=3/4.
    rows=[]
    for sid,series,values in [('s1','dense',[0.,0.,0.]),('s1','sparse',[1.]),('s2','one',[1.])]:
        for value in values:
            rows.append({'solute_id':sid,'source_doi':'10.1234/source','series_id':series,
                'adjacent_shared_temperatures':True,'gap_bin':'(0,5]',
                **{metric:value for metric in stability.METRICS}})
    cfg=copy.deepcopy(config());cfg['bootstrap']['repetitions']=100
    summary,_=stability.summarize(pd.DataFrame(rows),'identity',cfg)
    row=summary[(summary.relation=='all_observed_temperature_pairs')&(summary.gap_bin=='ALL')
                &(summary.metric=='decisive_reversal')].iloc[0]
    assert row.value==.75
    assert row.support_solutes==2
    empty=summary[(summary.gap_bin=='>20')&(summary.metric=='decisive_reversal')].iloc[0]
    assert empty.support_solutes==0
    assert pd.isna(empty.value)
