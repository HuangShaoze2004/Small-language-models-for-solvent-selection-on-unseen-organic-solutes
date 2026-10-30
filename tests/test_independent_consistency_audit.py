"""Synthetic mapping/identity/invalid-output fixtures, never efficacy results."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('consistency_audit',Path(__file__).parents[1]/'scripts/audit_consistency_outputs.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture(variant='candidate_order'):
    base={'query_id':'q','solute_id':'s','leakage_group':'g','condition_id':'c','task':'pair',
          'split':'val','temperature_k':298.15,'solute_smiles':'C[C@H](O)Cl',
          'candidates':[{'id':'A','smiles':'CCO','record_id':'a'},{'id':'B','smiles':'O','record_id':'b'}]}
    changed=copy.deepcopy(base);changed['query_id']=hashlib.sha256(f'consistency-v1|{variant}|q'.encode()).hexdigest()[:24]
    mapping={'original_query_id':'q','variant':variant,'candidate_to_original':{'A':'A','B':'B'},'changed_smiles':0,'smiles_fields':3}
    if variant=='candidate_order':
        changed['candidates'].reverse()
        for label,c in zip(['A','B'],changed['candidates']):c['id']=label
        mapping['candidate_to_original']={'A':'B','B':'A'}
    return base,changed,mapping


def test_swap_maps_chemical_choice_and_retains_measurement_identity():
    base,changed,mapping=fixture()
    assert audit.verify_transform(base,changed,mapping)==(0,3)
    assert audit.answer({'choice':'A'},base,{'A':'A','B':'B'})==audit.answer({'choice':'B'},changed,mapping['candidate_to_original'])
    changed['candidates'][0]['record_id']='other'
    with pytest.raises(ValueError,match='measurement identity'):audit.verify_transform(base,changed,mapping)


def test_nonbijective_mapping_and_stereochemical_change_are_rejected():
    base,changed,mapping=fixture();mapping['candidate_to_original']={'A':'B','B':'B'}
    with pytest.raises(ValueError,match='Non-bijective'):audit.verify_transform(base,changed,mapping)
    base,changed,mapping=fixture('equivalent_smiles');changed['solute_smiles']='C[C@@H](O)Cl';mapping['changed_smiles']=1
    with pytest.raises(ValueError,match='full chemical identity'):audit.verify_transform(base,changed,mapping)


def test_bad_answers_stay_invalid_and_raw_parsing_is_checked():
    base,_,_=fixture();conversion={'A':'A','B':'B'}
    assert audit.answer({'choice':[]},base,conversion) is None
    assert audit.answer({'choice':'TIE'},base,conversion)==('TIE',)
    assert audit.answer({'raw_output':'bad json','prediction':None},base,conversion) is None
    with pytest.raises(ValueError,match='parsing differs'):
        audit.answer({'raw_output':'{"choice":"A"}','prediction':{'choice':'B'}},base,conversion)
    base['task']='rank'
    assert audit.answer({'ranking':['A','A']},base,conversion) is None


def test_manual_aggregation_balances_physical_conditions_then_solutes():
    rows=[]
    for sid,cid,value in [('s','a',0),('s','a',1),('s','b',1),('t','a',0)]:
        rows.append({'variant':'candidate_order','task':'pair','solute_id':sid,'condition_id':cid,
                     **{key:float(value) for key in audit.METRICS}})
    per_solute,population=audit.recount(rows)
    assert per_solute[('candidate_order','pair','s','exact_agreement')]==.75
    assert population[('candidate_order','pair','exact_agreement')]==.375
