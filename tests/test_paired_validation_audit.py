"""Hand-derived synthetic bootstrap and aggregation checks, not research results."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('paired_audit',Path(__file__).parents[1]/'scripts/audit_paired_validation_contrast.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def frame(values):
    return pd.DataFrame([{'solute_id':s,'condition_id':c,'task':'pair','pair_accuracy':v}
                         for s,c,v in values])


def test_query_pairs_do_not_replace_condition_or_solute_weights():
    data=frame([('s','a',0),('s','a',1),('s','b',1),('t','a',0)])
    assert audit.hierarchical_values(data,'pair','pair_accuracy')=={'s':.75,'t':0}


def test_seed_average_precedes_solute_bootstrap_and_support_must_match():
    first=[frame([('s','c',x),('t','c',0)]) for x in [0,0,1]]
    second=[frame([('s','c',0),('t','c',0)]) for _ in range(3)]
    delta,seeds=audit.paired_values(first,second,'pair','pair_accuracy')
    assert delta=={'s':1/3,'t':0} and seeds==[0,0,.5]
    with pytest.raises(ValueError,match='Mismatched solute support'):
        audit.paired_values(first,[f.iloc[:1] for f in second],'pair','pair_accuracy')


def test_multinomial_bootstrap_preserves_zero_variance_and_percentile_interpolation():
    assert audit.bootstrap([.25,.25,.25],100,17)==(.25,[.25,.25])
    assert audit.percentile([0.,10.],.25)==2.5
    assert audit.percentile([0.,10.],.975)==9.75
