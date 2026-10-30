#!/usr/bin/env python3
"""Plot registered validation controls; incomplete seed groups remain pending."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.study import load_registry,evidence_name
from solvent_select.report import bootstrap_mean
from solvent_select.llm import write_json
from solvent_select.views import sha256_file


def groups(protocol,config):
    prefix=config['run_prefix']+protocol+'_'
    names={
        'Nearest neighbor':[prefix+'nearest'],
        'LightGBM':[prefix+f'lgbm_{s}' for s in [17,42,2026]],
        'Chemprop (raw input)':[prefix+f'chemprop_{s}' for s in [17,42,2026]],
        'Chemprop + canonicalization':[prefix+f'chemprop_{s}_canonical_control' for s in [17,42,2026]],
        'BM25 evidence (raw input)':[evidence_name(protocol,config,'text')],
        'BM25 + canonicalization':[evidence_name(protocol,config,'text')+'_canonical_control'],
        'Structure evidence':[evidence_name(protocol,config,'structure')],
        'Condition evidence':[evidence_name(protocol,config,'condition')],
        'Condition without temperature':[evidence_name(protocol,config,'no_temperature')],
        'Condition without solvent':[evidence_name(protocol,config,'no_solvent')]}
    return {label:[Path(config['consistency_runs_root'])/protocol/name/'evaluation' for name in values] for label,values in names.items()}


def main():
    registry=load_registry();rows=[];sources={};pending=[]
    for protocol,config in registry['protocols'].items():
        for label,folders in groups(protocol,config).items():
            if not all((folder/'metrics.json').exists() for folder in folders):
                pending.append({'protocol':protocol,'method':label,'required':list(map(str,folders))});continue
            frames=[];manifests=set()
            for folder in folders:
                metric=json.loads((folder/'metrics.json').read_text())
                if metric['partition']!='val' or metric['queries']!=1024:raise ValueError('Unexpected consistency cohort')
                manifests.update(digest for path,digest in metric['inputs'].items() if path.endswith('/manifest.json'))
                frame=pd.read_csv(folder/'per_solute.csv',dtype={'solute_id':str}).set_index(['variant','task','solute_id']).sort_index()
                frames.append(frame)
                for path in [folder/'metrics.json',folder/'per_solute.csv',folder.parent/'run.json']:sources[str(path)]=sha256_file(path)
            if len(manifests)!=1 or any(not frames[0].index.equals(frame.index) for frame in frames):raise ValueError('Unequal control cohorts')
            averaged=(sum(frames)/len(frames)).reset_index()
            for (variant,task),frame in averaged.groupby(['variant','task']):
                for metric in ['exact_agreement','top1_agreement','both_valid','any_smiles_changed']:
                    low,high=bootstrap_mean(frame[metric],10000,20261025)
                    rows.append({'protocol':protocol,'method':label,'variant':variant,'task':task,'metric':metric,
                                 'value':frame[metric].mean(),'ci_low':low,'ci_high':high,'solutes':len(frame),
                                 'runs':len(frames),'cohort_sha256':next(iter(manifests))})
    summary=pd.DataFrame(rows);output=Path('artifacts/reports/dualqa_consistency_validation');output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output/'summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for i,(protocol,config) in enumerate(registry['protocols'].items()):
        for j,task in enumerate(['pair','rank']):
            axis=axes[i,j];methods=list(groups(protocol,config))
            frame=summary[(summary.protocol==protocol)&(summary.task==task)&(summary.variant=='equivalent_smiles')&(summary.metric=='exact_agreement')].set_index('method').reindex(methods)
            present=frame.value.notna();y=np.arange(len(frame));values=frame.loc[present]
            axis.barh(y[present],values.value*100,color=['#aaa' if 'canonicalization' in label else '#31688e' for label in values.index])
            error=np.maximum(0,np.asarray([values.value-values.ci_low,values.ci_high-values.value])*100)
            axis.errorbar(values.value*100,y[present],xerr=error,fmt='none',ecolor='black',capsize=2)
            for position in y[~present]:axis.text(4,position,'PENDING (all 3 seeds required)',va='center',fontsize=9,color='#666')
            axis.set_yticks(y,methods);axis.invert_yaxis();axis.set_xlim(0,103)
            axis.set_title(protocol+' / '+config['data_tag']+': '+task)
            axis.set_xlabel('Exact mapped-answer agreement (%)');axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Registered validation controls; no final-test result\nCanonicalization controls are preprocessing checks; pending groups are not scored')
    for suffix in ['png','pdf']:fig.savefig(output/f'equivalent_smiles.{suffix}',dpi=180)
    plt.close(fig)
    write_json(output/'manifest.json',{'status':'VALIDATION_PROGRESS_ONLY' if pending else 'VALIDATION_CONTROLS_COMPLETED',
        'registry':'configs/study_registry_dualqa_v2.json','registry_sha256':sha256_file('configs/study_registry_dualqa_v2.json'),
        'pending_groups':pending,'sources':sources,'code_sha256':sha256_file(__file__),
        'bootstrap_repetitions':10000,'bootstrap_seed':20261025,
        'estimator':'Seed mean within solute, then bootstrap solutes; fixed512public validation queries per protocol. Incomplete learned-model seed groups excluded and explicitly labelled pending.',
        'test_scores_read':False,'outputs':{p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps({'observed_summary_rows':len(summary),'pending_groups':pending}))


if __name__=='__main__':main()
