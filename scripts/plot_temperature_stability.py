#!/usr/bin/env python3
"""Plot all registered training temperature diagnostics without metric selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solvent_select.temperature_stability import METRICS
from solvent_select.views import sha256_file,write_json

LABELS=['Changed three-way label','Decisive A/B reversal','Enter or leave near-tie band',
        'Near tie at both temperatures','Decisive at both temperatures','Raw difference changes sign',
        'Absolute change of solvent difference','Difference changes by >0.1 dex','Reversal given both decisive']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source,output=Path(args.source),Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous plot outputs')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['status']!='COMPLETED_TRAINING_ONLY_TEMPERATURE_DESCRIPTION' or manifest['heldout_labels_read'] is not False:
        raise ValueError('Only the completed training-only analysis may be plotted')
    for path,expected in manifest['inputs'].items():
        if sha256_file(path)!=expected:raise ValueError('Analysis input changed: '+path)
    for filename,expected in manifest['outputs'].items():
        if sha256_file(source/filename)!=expected:raise ValueError('Analysis output changed: '+filename)
    config_path=next(path for path in manifest['inputs'] if Path(path).name=='temperature_stability_v1.json')
    config=json.loads(Path(config_path).read_text())
    data=pd.read_csv(source/'summary.csv')
    if set(data.metric)!=set(METRICS):raise ValueError('Every registered metric must be retained')
    output.mkdir(parents=True)
    plotted=data[data.gap_bin!='ALL'].copy()
    plotted.to_csv(output/'plotted_values.csv',index=False)
    for protocol in ['identity','scaffold']:
        fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
        for axis,metric,label in zip(axes.flat,METRICS,LABELS):
            factor=1 if metric=='absolute_delta_change_dex' else 100
            for relation,offset,color,title in [
                ('all_observed_temperature_pairs',-.07,'#31688e','All observed temperature pairs'),
                ('adjacent_shared_temperatures',.07,'#b15c22','Adjacent shared temperatures')]:
                group=plotted[(plotted.protocol==protocol)&(plotted.metric==metric)&(plotted.relation==relation)]
                group=group.set_index('gap_bin').reindex(config['temperature_gap_labels'])
                for position,(_,row) in enumerate(group.iterrows()):
                    if pd.isna(row.value):
                        axis.text(position,.02,'empty',transform=axis.get_xaxis_transform(),ha='center',fontsize=7)
                        continue
                    error=None if pd.isna(row.ci_low) else np.maximum(0,[[row.value-row.ci_low],[row.ci_high-row.value]])*factor
                    axis.errorbar(position+offset,row.value*factor,yerr=error,fmt='o',markersize=4,
                        color=color,mfc=color if row.support_solutes>=10 else 'white',capsize=2,
                        label=title if position==0 else None)
            axis.set_xticks(range(4),config['temperature_gap_labels'])
            axis.set_title(label,fontsize=10)
            axis.set_xlabel('Measured temperature gap (K)',fontsize=8)
            axis.set_ylabel('Absolute change (dex)' if factor==1 else 'Solute macro rate (%)',fontsize=8)
            axis.set_ylim(bottom=0)
            axis.spines[['top','right']].set_visible(False)
            axis.grid(axis='y',alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
        fig.suptitle(f'Training-only temperature description: {protocol}\n'
            'All registered metrics; 95% intervals resample solutes; different gap bins can contain different solutes\n'
            'Hollow markers: fewer than 10 solutes. No causal or RAG-effectiveness claim.',fontsize=12)
        for suffix in ['png','pdf']:fig.savefig(output/f'{protocol}_temperature_stability.{suffix}',dpi=180)
        plt.close(fig)
    write_json(output/'manifest.json',{'status':'TRAINING_ONLY_DESCRIPTIVE_FIGURES','source_manifest':str(source/'manifest.json'),
        'source_manifest_sha256':sha256_file(source/'manifest.json'),'code_sha256':sha256_file(__file__),
        'all_registered_metrics_plotted':METRICS,'heldout_labels_read':False,
        'outputs':{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}})


if __name__=='__main__':main()
