#!/usr/bin/env python3
"""Analyze completed registered validation runs; pending methods stay explicit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from solvent_select.error_analysis import run_analysis
from solvent_select.study import load_registry,property_names,evidence_name
from solvent_select.views import sha256_file
from solvent_select.llm import write_json


def verify_existing(path):
    record=json.loads(path.read_text())
    if record['status']!='VALIDATION_DIAGNOSTICS_ONLY':raise ValueError('Unexpected analysis partition/status')
    for source,expected in record['inputs'].items():
        if sha256_file(source)!=expected:raise ValueError('Previous validation-analysis input changed: '+source)


def main():
    registry=load_registry();analysis=json.loads(Path(registry['analysis_plan']).read_text())
    rows=[]
    for protocol,config in registry['protocols'].items():
        names=property_names(protocol,config)
        names += [evidence_name(protocol,config,mode) for mode in analysis['direct_evidence_modes']]
        names += [name for name in analysis['llm_validation_tasks_to_carry_to_test'] if '_'+protocol+'_' in name]
        for name in names:
            run=Path(config['runs_root'])/name
            output=run/'error_analysis_val_v1'
            metric=run/'evaluation_view10000/per_query.parquet'
            prediction=run/('predictions_val.parquet' if name in property_names(protocol,config) else 'predictions.jsonl')
            if not metric.exists() or not prediction.exists():
                rows.append({'run':name,'status':'PENDING_REGISTERED_VALIDATION'});continue
            if (output/'analysis.json').exists():
                verify_existing(output/'analysis.json');status='VERIFIED_EXISTING'
            else:
                result=run_analysis(Path(config['views'])/'val_queries.jsonl',metric,Path(config['measurements']),
                    Path(config['split']),Path('configs/error_analysis.yaml'),output,
                    Path(config['views'])/'val_targets.jsonl',prediction)
                if result['status']!='VALIDATION_DIAGNOSTICS_ONLY':raise ValueError('Only validation is authorized')
                status='COMPLETED_REAL_VALIDATION_ANALYSIS'
            rows.append({'run':name,'status':status,'analysis':str(output/'analysis.json'),
                         'analysis_sha256':sha256_file(output/'analysis.json')})
            print(name,status,flush=True)
    path=Path('artifacts/audit/dualqa/validation_postprocessing_snapshot.json')
    write_json(path,{'status':'VALIDATION_ONLY_PROGRESS_SNAPSHOT','created_utc':datetime.now(timezone.utc).isoformat(),
        'argv':sys.argv,'runs':rows,'pending':sum(r['status']=='PENDING_REGISTERED_VALIDATION' for r in rows),
        'test_scores_read':False,'code_sha256':sha256_file(__file__)})


if __name__=='__main__':main()
