#!/usr/bin/env python3
"""Build journal-sized figures/tables from completed, audited frozen results.

No prediction, training, threshold selection or scientific-result replacement.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/reports/dualqa'
WIDTH = 174 / 25.4
CORE = OrderedDict([
    ('Nearest neighbor', 'NN'), ('LightGBM', 'LightGBM'), ('Chemprop', 'Chemprop'),
    ('Base LM / no retrieval', 'Base LM'), ('Ordinary SFT / no retrieval', 'Ordinary SFT'),
    ('Retrieval SFT / structure', 'R-SFT + structure'), ('Retrieval SFT / condition', 'R-SFT + condition'),
])
PROTOCOLS = [('identity', 'Identity'), ('scaffold', 'Scaffold')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def read(path):
    return json.loads(Path(path).read_text())


def interval(row, percentage=True):
    scale = 100 if percentage else 1
    digits = 1 if percentage else 3
    return f"{row['value']*scale:.{digits}f} [{row['ci_low']*scale:.{digits}f}, {row['ci_high']*scale:.{digits}f}]"


def table(path, columns, header, rows, caption=None, label=None):
    lines = []
    if caption:
        lines += [r'\begingroup\footnotesize', r'\setlength{\tabcolsep}{3pt}', '\\begin{longtable}{'+columns+'}',
                  '\\caption{'+caption+'}\\label{'+label+'}\\\\', r'\toprule', header+r' \\', r'\midrule', r'\endfirsthead',
                  '\\multicolumn{'+str(len(header.split('&')))+'}{l}{\\textit{Continued}} \\\\', r'\toprule', header+r' \\', r'\midrule', r'\endhead']
    else:
        lines += [r'\begingroup\footnotesize', r'\setlength{\tabcolsep}{3pt}', '\\begin{tabular}{'+columns+'}', r'\toprule', header+r' \\', r'\midrule']
    lines += [row+r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{longtable}' if caption else r'\end{tabular}', r'\endgroup']
    path.write_text('\n'.join(lines)+'\n')


def save(fig, output, name):
    fig.savefig(output / (name+'.pdf'), metadata={'CreationDate': None, 'ModDate': None}, bbox_inches=None)
    fig.savefig(output / (name+'.png'), dpi=600, facecolor='white')
    plt.close(fig)


def clean_axis(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='x', alpha=.18, linewidth=.4)


def box(axis, x, y, width, height, text, fontsize=8):
    axis.add_patch(FancyBboxPatch((x,y), width,height, boxstyle='round,pad=0.008', facecolor='white', edgecolor='.2', linewidth=.7))
    axis.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.35)


def arrow(axis, a, b):
    axis.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle':'->','lw':.8,'color':'.25'})


def workflow(output):
    fig, axis = plt.subplots(figsize=(WIDTH, 91/25.4));axis.set_axis_off();axis.set_xlim(0,1);axis.set_ylim(0,1)
    box(axis,.02,.78,.96,.18,'BigSolDB 2.0: 103,944 source records\nNative mole fraction; full chemical identity; DOI and exact temperature')
    box(axis,.02,.48,.29,.19,'Protocol-specific training banks\nIdentity: 1,140 solutes\nScaffold: 732 solutes')
    box(axis,.36,.48,.28,.19,'Shared protected test\n144 solute identities\n124 with eligible questions')
    box(axis,.69,.48,.29,.19,'Measured candidate blocks\nSame solute, DOI and T\nAt most eight solvents')
    for x in [.165,.5,.835]:arrow(axis,(x,.78),(x,.68))
    box(axis,.02,.18,.29,.18,'Training-only retrieval\nText / structure / condition\nSelf-group excluded in training')
    box(axis,.36,.18,.28,.18,'Fixed predictors\nNearest neighbor / LightGBM\nChemprop / 4B LM variants')
    box(axis,.69,.18,.29,.18,'Held-out decisions\n3,089 comparisons\n1,457 rankings')
    arrow(axis,(.165,.48),(.165,.37));arrow(axis,(.5,.48),(.5,.37));arrow(axis,(.835,.48),(.835,.37))
    arrow(axis,(.315,.27),(.35,.27));arrow(axis,(.645,.27),(.68,.27))
    axis.text(.5,.055,'Solute-macro scores; matched-seed contrasts; option-order and equivalent-SMILES checks',ha='center',va='center',fontsize=8)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01);save(fig,output,'fig1_workflow')
    fig,axis=plt.subplots(figsize=(80/25.4,30/25.4));axis.set_axis_off();axis.set_xlim(0,1);axis.set_ylim(0,1)
    box(axis,.015,.35,.27,.5,'Unseen solute\n+ measured\nsolvent options',7.5)
    box(axis,.36,.35,.28,.5,'Train-only\nretrieval +\n4B LM',7.5)
    box(axis,.715,.35,.27,.5,'Solvent choice\n+ stability\nchecks',7.5)
    arrow(axis,(.29,.60),(.35,.60));arrow(axis,(.65,.60),(.705,.60))
    axis.text(.5,.12,'Pure solvents; fixed temperature',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.02);save(fig,output,'graphical_abstract')


def performance(data, output):
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,118/25.4),sharey=True)
    for i,(protocol,label) in enumerate(PROTOCOLS):
        for j,(metric,xlabel,scale) in enumerate([('pair_accuracy','Comparison accuracy (%)',100),('top1','Best-solvent hit rate (%)',100),('regret','Selection regret (dex)',1)]):
            a=axes[i,j];subset=data[(data.protocol==protocol)&(data.metric==metric)].set_index('method').loc[list(CORE)]
            for k,(_,row) in enumerate(subset.iterrows()):
                a.errorbar(row.value*scale,k,xerr=np.maximum(0,[[row.value-row.ci_low],[row.ci_high-row.value]])*scale,
                           fmt='o',markersize=3,color='black',ecolor='.35',capsize=1.5,lw=.7)
            a.set_yticks(range(len(CORE)),list(CORE.values()),fontsize=8)
            a.set_ylim(len(CORE)-.5,-.5);a.set_xlabel(xlabel,fontsize=8);clean_axis(a)
            if i==0:a.text(0,1.04,chr(97+j),transform=a.transAxes,fontweight='bold',fontsize=9)
            else:a.text(0,1.04,chr(100+j),transform=a.transAxes,fontweight='bold',fontsize=9)
            if j==0:a.set_ylabel(label,fontsize=9)
            a.set_xlim((15,90) if scale==100 else (0,1.05))
    fig.subplots_adjust(left=.21,right=.99,top=.96,bottom=.11,wspace=.25,hspace=.48);save(fig,output,'fig2_performance')


def contrasts(values, output):
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,52/25.4),sharey=True)
    for j,(metric,xlabel,scale,threshold) in enumerate([('pair_accuracy',r'$\Delta$ comparison accuracy (pp)',100,2),('top1',r'$\Delta$ best-solvent hit (pp)',100,None),('regret',r'$\Delta$ selection regret (dex)',1,-.05)]):
        a=axes[j]
        for i,(protocol,label) in enumerate(PROTOCOLS):
            row=next(r for r in values[protocol]['comparisons'] if r['metric']==metric);v=row['difference_first_minus_second']*scale;lo,hi=[x*scale for x in row['ci95']]
            a.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',markersize=4,color='black',capsize=2,lw=.8)
        a.axvline(0,lw=.7,color='.45')
        if threshold is not None:a.axvline(threshold,lw=.8,color='.3',linestyle='--')
        a.set_yticks([0,1],[x[1] for x in PROTOCOLS],fontsize=8);a.set_ylim(1.55,-.55);a.set_xlabel(xlabel,fontsize=8);clean_axis(a)
        a.text(0,1.04,chr(97+j),transform=a.transAxes,fontweight='bold',fontsize=9)
    fig.subplots_adjust(left=.12,right=.985,top=.91,bottom=.26,wspace=.38);save(fig,output,'fig3_contrasts')


def consistency(data, output):
    methods=OrderedDict(CORE);methods['Chemprop + canonicalization']='Chemprop + canon.'
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,128/25.4),sharey=True)
    # The principal protocol is shown in the main figure; all protocols/methods
    # are retained in the supplementary table and source CSV, not selected by score.
    for i,variant in enumerate(['candidate_order','equivalent_smiles']):
        for j,(task,metric,label) in enumerate([('pair','exact_agreement','Pair answer\nagreement (%)'),('rank','top1_agreement','Ranking first-choice\nagreement (%)'),('rank','exact_agreement','Complete ranking\nagreement (%)')]):
            a=axes[i,j];s=data[(data.protocol=='identity')&(data.variant==variant)&(data.task==task)&(data.metric==metric)].set_index('method').loc[list(methods)]
            for k,(_,row) in enumerate(s.iterrows()):a.errorbar(row.value*100,k,xerr=np.maximum(0,[[row.value-row.ci_low],[row.ci_high-row.value]])*100,fmt='o',markersize=3,color='black',capsize=1.5,lw=.7)
            a.set_yticks(range(len(methods)),list(methods.values()),fontsize=8);a.set_ylim(len(methods)-.5,-.5);a.set_xlim(-3,103);a.set_xlabel(label,fontsize=7.5)
            if j==0:a.set_ylabel('Option order' if i==0 else 'Equivalent SMILES',fontsize=8)
            a.text(0,1.04,chr(97+i*3+j),transform=a.transAxes,fontweight='bold',fontsize=9);clean_axis(a)
    fig.subplots_adjust(left=.23,right=.99,top=.96,bottom=.10,wspace=.27,hspace=.45);save(fig,output,'fig4_consistency')


def molecule_vector(axis,smiles):
    """Render RDKit SVG paths directly into the vector PDF without conversion tools.

    RDKit's supported absolute M/L/Q/C/Z path subset is copied without geometry
    approximation; unknown commands/elements fail instead of silently omitting
    chemical marks. The case molecules are checked by visual inspection too.
    """
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Unparseable case structure')
    draw=rdMolDraw2D.MolDraw2DSVG(800,350);opt=draw.drawOptions();opt.useBWAtomPalette();opt.padding=.10;opt.fixedFontSize=42;draw.DrawMolecule(mol);draw.FinishDrawing()
    tree=ET.fromstring(draw.GetDrawingText());scale=axis.get_position().width*axis.figure.get_figwidth()*72/800
    for element in tree:
        tag=element.tag.rsplit('}',1)[-1]
        if tag=='rect':continue  # the white background is supplied by matplotlib
        if tag!='path':raise ValueError('Unsupported chemical SVG element: '+tag)
        d=element.attrib['d'];tokens=re.findall(r'[MLQCZ]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',d)
        if re.sub(r'[MLQCZ\s,0-9.eE+\-]','',d):raise ValueError('Unsupported chemical SVG path')
        vertices=[];codes=[];i=0;command=None
        while i<len(tokens):
            if tokens[i] in ['M','L','Q','C','Z']:command=tokens[i];i+=1
            if command=='Z':vertices.append((0,0));codes.append(MplPath.CLOSEPOLY);command=None;continue
            count={'M':2,'L':2,'Q':4,'C':6}[command];values=[float(v) for v in tokens[i:i+count]];i+=count
            for j in range(0,count,2):
                vertices.append((values[j],values[j+1]));codes.append({'M':MplPath.MOVETO,'L':MplPath.LINETO,'Q':MplPath.CURVE3,'C':MplPath.CURVE4}[command])
            if command=='M':command='L'
        style=dict(item.split(':',1) for item in element.attrib.get('style','').split(';') if ':' in item)
        fill=style.get('fill',element.attrib.get('fill','black'));stroke=style.get('stroke',element.attrib.get('stroke','none'))
        width=float(style.get('stroke-width','0').removesuffix('px'))*scale
        axis.add_patch(PathPatch(MplPath(vertices,codes),facecolor='none' if fill=='none' else fill,edgecolor='none' if stroke=='none' else stroke,lw=width,
                                 capstyle='butt',joinstyle='round'))
    axis.set_xlim(0,800);axis.set_ylim(350,0);axis.set_aspect('equal')


def case_records(sources):
    p=ROOT/'artifacts/runs_sourceqa/sourceqa_identity_qwen_rag_17_condition_test/error_analysis_test_v1/failure_cases.jsonl';sources[str(p.relative_to(ROOT))]=sha(p)
    raw=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    selected=[next(r for r in raw if r['task']==task and r['selection']=='greatest_regret' and r['selection_rank']==1) for task in ['pair','rank']]
    measurements=ROOT/'data/processed/sourceqa/measurements.parquet';sources[str(measurements.relative_to(ROOT))]=sha(measurements)
    ids=[c['record_id'] for r in selected for c in r['candidates']]
    measured=pd.read_parquet(measurements,columns=['record_id','logx'],filters=[('record_id','in',ids)]).set_index('record_id').logx.to_dict()
    checks=[]
    for r in selected:
        for c in r['candidates']:
            if not math.isclose(c['actual_logx'],measured[c['record_id']],abs_tol=1e-14):raise ValueError('Case label differs from measured record')
        for seed in [17,42,2026]:
            run=ROOT/f'artifacts/runs_sourceqa/sourceqa_identity_qwen_rag_{seed}_condition_test';path=run/'predictions.jsonl';sources[str(path.relative_to(ROOT))]=sha(path)
            row=next(json.loads(x) for x in path.open() if json.loads(x)['query_id']==r['query_id'])
            answer=row['prediction'];chosen=None
            if answer:
                if r['task']=='rank':chosen=answer['ranking'][0]
                elif answer['choice']=='TIE':chosen=min(r['candidates'],key=lambda c:Chem.MolToSmiles(Chem.MolFromSmiles(c['smiles']),isomericSmiles=True))['id']
                else:chosen=answer['choice']
            values={c['id']:c['actual_logx'] for c in r['candidates']};regret=max(values.values())-values[chosen] if chosen else max(values.values())-min(values.values())
            evalpath=run/'evaluation_test_view10000/per_query.parquet';sources[str(evalpath.relative_to(ROOT))]=sha(evalpath)
            score=pd.read_parquet(evalpath,filters=[('query_id','==',r['query_id'])]);assert len(score)==1
            if not math.isclose(regret,float(score.iloc[0].regret),abs_tol=1e-12):raise ValueError('Across-seed case regret differs')
            checks.append({'task':r['task'],'seed':seed,'query_id':r['query_id'],'choice':chosen,'regret':regret,'raw_prediction':answer})
    return selected,checks


def cases(selected, output):
    fig=plt.figure(figsize=(WIDTH,105/25.4));grid=fig.add_gridspec(2,3,left=.02,right=.99,bottom=.12,top=.97,wspace=.08,hspace=.50)
    for i,row in enumerate(selected):
        byid={c['id']:c for c in row['candidates']};chosen=byid[row['model_selected_candidate_id']];best=max(row['candidates'],key=lambda c:c['actual_logx'])
        entries=[(row['solute_smiles'],f"{'a' if i==0 else 'b'}   {row['task'].capitalize()} task; {row['temperature_k']:.2f} K"),
                 (chosen['smiles'],f"Selected: {chosen['id']}\nMeasured log x = {chosen['actual_logx']:.3f}"),
                 (best['smiles'],f"Measured best: {best['id']}\nMeasured log x = {best['actual_logx']:.3f}")]
        for j,(smiles,text) in enumerate(entries):
            a=fig.add_subplot(grid[i,j]);molecule_vector(a,smiles);a.set_axis_off();a.text(.5,-.02,text,ha='center',va='top',transform=a.transAxes,fontsize=8)
        fig.text(.50,.50-i*.43,f"Selection regret: {row['regret']:.3f} dex; {len(row['candidates'])} measured candidates",ha='center',fontsize=8)
    save(fig,output,'fig5_cases')


def write_slices(out, sources):
    """Retain every registered descriptive bin; never filter using its result."""
    config = ROOT / 'configs/error_analysis.yaml'
    sources[str(config.relative_to(ROOT))] = sha(config)
    headings = {
        'all': 'All eligible questions', 'temperature': 'Temperature (K)',
        'train_similarity': 'Maximum training-solute Tanimoto',
        'solute_functional_group': 'Solute functional group',
        'candidate_solvent_class': 'Candidate-solvent class',
    }
    for protocol, title, runroot, prefix, label in [
        ('identity', 'Primary', 'runs_sourceqa', 'sourceqa_identity', 'primary'),
        ('scaffold', 'Strict', 'runs_strictqa', 'strictqa_scaffold', 'strict'),
    ]:
        frames = []
        for seed in [17, 42, 2026]:
            path = ROOT / f'artifacts/{runroot}/{prefix}_qwen_rag_{seed}_condition_test/error_analysis_test_v1/slice_metrics.csv'
            sources[str(path.relative_to(ROOT))] = sha(path)
            frame = pd.read_csv(path)
            if len(frame) != 62 or frame.duplicated(['slice_family', 'slice', 'task']).any():
                raise ValueError('Incomplete or duplicated descriptive slice report')
            frames.append(frame.set_index(['slice_family', 'slice', 'task']))
        reference = frames[0]
        for frame in frames[1:]:
            if not reference.index.equals(frame.index):
                raise ValueError('Slice inventory changed across seeds')
            for count in ['queries', 'physical_conditions', 'solutes', 'pair_accuracy_support_solutes', 'top1_support_solutes', 'regret_support_solutes']:
                if not reference[count].equals(frame[count]):
                    raise ValueError('Slice support changed across seeds')
        rows = []
        for family, heading in headings.items():
            rows.append(r'\multicolumn{6}{l}{\textit{' + heading + '}}')
            names = list(reference.loc[family].index.get_level_values('slice').unique())
            for name in names:
                pairkey = (family, name, 'pair'); rankkey = (family, name, 'rank')
                pairn = int(reference.loc[pairkey, 'pair_accuracy_support_solutes'])
                rankn = int(reference.loc[rankkey, 'top1_support_solutes'])
                if int(reference.loc[rankkey, 'regret_support_solutes']) != rankn:
                    raise ValueError('Different top-one and regret slice supports')
                def metric(key, metric_name, support, percentage):
                    values = np.array([frame.loc[key, metric_name] for frame in frames], dtype=float)
                    if support == 0:
                        if not np.isnan(values).all():
                            raise ValueError('Nonempty zero-support slice metric')
                        return '--'
                    if not np.isfinite(values).all():
                        raise ValueError('Missing positive-support slice metric')
                    return f'{values.mean() * (100 if percentage else 1):.{1 if percentage else 3}f}'
                def support(n):
                    return str(n) + (r'$^{a}$' if n < 10 else '')
                if family in ['temperature', 'train_similarity']:
                    pretty = '$' + str(name).replace('>=', r'\geq ') + '$'
                else:
                    pretty = escape(str(name).replace('_', ' '))
                rows.append(' & '.join([pretty, support(pairn), metric(pairkey, 'pair_accuracy', pairn, True),
                                        support(rankn), metric(rankkey, 'top1', rankn, True), metric(rankkey, 'regret', rankn, False)]))
        caption = (title + ' protocol: all predefined descriptive slices for condition-retrieval SFT. '
                   'Accuracy and best hit are percentages; regret is in dex. Values average the three fixed learning seeds. '
                   '$n_P$ and $n_R$ are independent pair-eligible and ranking-eligible solutes, respectively. '
                   'Superscript $a$ marks fewer than ten eligible solutes; zero-support metrics are undefined. '
                   'Functional and solvent classes overlap. No slice was removed on the basis of its result.')
        table(out / f'slices_{protocol}.tex', 'lrrrrr', r'Slice & $n_P$ & Accuracy & $n_R$ & Best hit & Regret', rows,
              caption, 'tab:si-slices-' + label)


def write_tables(data,con,comp,timing,selected,checks,out,sources):
    def perfrows(frame, core=False):
        rows=[]
        for protocol,label in PROTOCOLS:
            methods=list(CORE) if core else list(frame[frame.protocol==protocol].method.unique())
            for name in methods:
                g=frame[(frame.protocol==protocol)&(frame.method==name)].set_index('metric')
                rows.append(' & '.join([label,escape(CORE.get(name,name)),interval(g.loc['pair_accuracy']),interval(g.loc['top1']),interval(g.loc['regret'],False)]))
        return rows
    header=r'Protocol & Method & Accuracy (\%) & Best hit (\%) & Regret (dex)'
    table(out/'main_performance.tex','llrrr',header,perfrows(data,True))
    table(out/'all_methods.tex',r'lp{.27\linewidth}rrr',header,perfrows(data),
          'All registered final method groups. Values are solute-macro means with 95\\% solute-bootstrap intervals. Learning seeds are averaged within solute; deterministic controls run once.', 'tab:all-methods')
    rows=[]
    metric_labels = {'pair_accuracy': 'Comparison accuracy (pp)',
                     'top1': 'Best-solvent hit (pp)',
                     'regret': 'Selection regret (dex)'}
    for protocol,label in PROTOCOLS:
        for r in comp[protocol]['comparisons']:
            scale=100 if r['metric']!='regret' else 1
            rows.append(f"{label} & {metric_labels[r['metric']]} & {r['difference_first_minus_second']*scale:+.3f} & [{r['ci95'][0]*scale:+.3f}, {r['ci95'][1]*scale:+.3f}]")
    table(out/'contrasts.tex','llrr',r'Protocol & Metric & Difference & 95\% interval',rows)
    rows=[]
    registry=read(ROOT/'configs/study_registry_dualqa_v2.json');audit=read(ROOT/'artifacts/audit/sourceqa/data_audit.json');strict=read(ROOT/'artifacts/audit/strictqa/data_audit.json')
    query_counts = {}
    for protocol,label in PROTOCOLS:
        counts=audit['splits']['identity'] if protocol=='identity' else strict['splits']['scaffold']
        cfg=registry['protocols'][protocol]
        for split in ['train','val','test']:
            p=ROOT/cfg['views']/(split+'_queries.jsonl');sources[str(p.relative_to(ROOT))]=sha(p);q=[json.loads(x) for x in p.read_text().splitlines() if x]
            if split == 'test':
                query_counts[protocol] = {task: sum(x['task'] == task for x in q) for task in ['pair', 'rank']}
            rows.append(f"{label} & {split} & {counts[split]['solutes']:,} & {counts[split]['records']:,} & {sum(x['task']=='pair' for x in q):,} & {sum(x['task']=='rank' for x in q):,}")
    table(out/'cohorts.tex','llrrrr',r'Protocol & Partition & Solutes & Records & Pair view & Rank view',rows,
          'Executed partitions and budgeted instruction views. Record counts refer to measurements. The test identities and query view are shared by both protocols. Training views contain 10,000 questions per protocol; eligible identities can be fewer than manifest identities.', 'tab:cohorts')
    support_path = REPORT / 'final_resources_v1/evidence_support.csv'
    sources[str(support_path.relative_to(ROOT))] = sha(support_path)
    support = pd.read_csv(support_path)
    if len(support) != 20 or support.duplicated(['protocol', 'mode', 'task']).any():
        raise ValueError('Incomplete or duplicated final evidence coverage')
    retrieval_labels = {'text': 'Text', 'structure': 'Structure', 'condition': 'Condition',
                        'no_temperature': 'No temperature', 'no_solvent': 'No solvent'}
    rows = []
    for protocol, label in PROTOCOLS:
        for mode, mode_label in retrieval_labels.items():
            taskrows = support[(support.protocol == protocol) & (support['mode'] == mode)].set_index('task')
            if set(taskrows.index) != {'pair', 'rank'}:
                raise ValueError('Missing evidence task')
            pair, rank = taskrows.loc['pair'], taskrows.loc['rank']
            rows.append(' & '.join([label, mode_label,
                str(query_counts[protocol]['pair']), str(int(pair.solute_count)),
                f'{pair.supported_pair_fraction * 100:.1f}',
                str(query_counts[protocol]['rank']), str(int(rank.solute_count)),
                f'{rank.supported_pair_fraction * 100:.1f}',
                f'{rank.all_candidate_pairs_supported * 100:.1f}']))
    table(out/'evidence_support.tex', 'llrrrrrrr',
          r'Protocol & Retrieval & $N_P$ & $n_P$ & \shortstack{Pair support\\(\%)} & $N_R$ & $n_R$ & \shortstack{Rank pairs\\(\%)} & \shortstack{Rank full\\(\%)}', rows,
          'Complete final direct-evidence coverage: both task rows for all ten protocol/retrieval groups. '
          '$N_P$ and $N_R$ count pair and ranking questions; $n_P$ and $n_R$ count their eligible solutes. '
          'Support fractions use the registered condition-then-solute macro average. Pair support means that at least one retrieved card contains both exact candidate solvents. '
          'Rank pairs is the supported fraction of candidate pairs; rank full means every candidate pair is supported. '
          'All twenty source task rows have 100\\% valid output, which does not establish supported or correct decisions. '
          'These are coverage measurements, not additional model accuracy experiments.', 'tab:evidence-support')
    for protocol,label in PROTOCOLS:
        for variant in ['candidate_order','equivalent_smiles']:
            rows=[]
            s=con[(con.protocol==protocol)&(con.variant==variant)]
            for name in s.method.unique():
                for task in ['pair','rank']:
                    r=s[(s.method==name)&(s.task==task)].set_index('metric')
                    rows.append(' & '.join([escape(name),task,interval(r.loc['exact_agreement']),interval(r.loc['top1_agreement']),interval(r.loc['both_valid'])]))
            filename=out/f'consistency_{protocol}_{variant}.tex'
            table(filename,r'p{.26\linewidth}lrrr',r'Method & Task & Exact (\%) & First choice (\%) & Both valid (\%)',rows,
                  f"{label} protocol, {escape(variant)} transformation. Solute-macro means and 95\\% solute-bootstrap intervals; invalid outputs retain disagreement penalties.", 'tab:consistency-'+protocol+'-'+variant.replace('_','-'))
    (out/'consistency.tex').write_text('\n'.join('\\input{generated/'+f'consistency_{p}_{v}.tex'+'}' for p,_ in PROTOCOLS for v in ['candidate_order','equivalent_smiles'])+'\n')
    rows=[]
    for r in timing.to_dict('records'):
        denominator=int(r['questions']) if pd.notna(r['questions']) else int(r['records']);unit='queries' if pd.notna(r['questions']) else 'records'
        scope={'saved_per_query_retrieval_and_decision_excludes_setup_and_persistence':'E','feature_prediction_and_persistence_model_resident':'P','graph_prediction_and_persistence_excludes_graph_and_checkpoint_preparation':'G','saved_successful_batches_all_segments_excludes_setup_and_persistence':'L'}[r['scope']]
        name=r['run'].removeprefix('sourceqa_identity_').removeprefix('strictqa_scaffold_').removesuffix('_test')
        peak='--' if pd.isna(r.get('peak_allocated_bytes')) else f"{r['peak_allocated_bytes']/2**30:.2f}"
        rows.append(' & '.join([r['protocol'],r'\path{'+name+'}',scope,f"{r['seconds']:.3f}",str(denominator)+' '+unit,peak]))
    table(out/'resources.tex',r'lp{.30\linewidth}lrrr',r'Protocol & Run & Scope & Seconds & Denominator & Peak GiB',rows,
          'Observed final inference costs. E: saved evidence retrieval/decision, excluding setup and persistence; P: model-resident feature/prediction and persistence; G: graph prediction/persistence excluding graph and checkpoint preparation; L: successful saved LM batches, excluding setup and persistence. CPU/GPU and records/queries are distinct scopes; do not interpret quotients as controlled speedups. Shared base-LM inference is a reused timing observation, not additional compute. Peaks are allocated memory where recorded.', 'tab:resources')
    rows=[]
    for r in checks:
        original=next(x for x in selected if x['task']==r['task']);candidate=next((c for c in original['candidates'] if c['id']==r['choice']),None)
        rows.append(' & '.join([r['task'],str(r['seed']),r['choice'] or 'invalid',r'\path{'+candidate['smiles']+'}' if candidate else '--',f"{r['regret']:.6f}"]))
    table(out/'cases.tex','lllr r',r'Task & Seed & Selected ID & Selected solvent SMILES & Regret (dex)',rows,
          'Across-seed outcomes for the two frozen greatest-regret queries selected within primary condition-retrieval seed 17 results. Both task views concern the same solute and condition, not independent case studies. Source DOI: 10.1016/j.molliq.2020.113363; temperature 283.15 K. Case selection is descriptive, not a new confirmation test.', 'tab:cases')
    grade=read(ROOT/'artifacts/audit/dualqa/independent_final_scores_v1.json');rows=[]
    for r in grade['runs']:
        name=r['run'].removeprefix('sourceqa_identity_').removeprefix('strictqa_scaffold_').removesuffix('_test')
        metrics=r['independent_metrics'];rows.append(' & '.join([r['protocol'],r'\path{'+name+'}',f"{metrics['pair']['pair_accuracy']*100:.2f}",f"{metrics['rank']['top1']*100:.2f}",f"{metrics['rank']['regret']:.6f}"]))
    table(out/'per_seed.tex',r'lp{.39\linewidth}rrr',r'Protocol & Run & Accuracy (\%) & Best hit (\%) & Regret (dex)',rows,
          'All 62 completed final method runs. The run IDs identify the fixed learning seed where applicable. Each number is independently checked against raw predictions and measured targets; these rows are not independent datasets.', 'tab:per-seed')
    rows=[]
    def number(value,scale=1,digits=3):
        return '--' if value is None or not math.isfinite(float(value)) else f'{float(value)*scale:.{digits}f}'
    for r in grade['runs']:
        name=r['run'].removeprefix('sourceqa_identity_').removeprefix('strictqa_scaffold_').removesuffix('_test')
        m=r['independent_metrics']
        rows.append(' & '.join([r['protocol'],r'\path{'+name+'}',number(m['pair']['nontie_pair_accuracy'],100,2),
                               number(m['rank']['rank_tau']),number(m['pair']['covered'],100,2),number(m['rank']['covered'],100,2)]))
    table(out/'supplemental_metrics.tex',r'lp{.34\linewidth}rrrr',r'Protocol & Run & \shortstack{Non-tie\\accuracy (\%)} & Rank $\tau_b$ & \shortstack{Pair valid\\(\%)} & \shortstack{Rank valid\\(\%)}',rows,
          'Additional independently audited metrics for all 62 final runs. Non-tie accuracy excludes measured near-ties by the fixed 0.1 dex rule; valid-output rates use all questions. Rank correlation retains the registered invalid-output penalty, while mathematically undefined valid correlations remain undefined rather than zero. Per-run macro estimates are reported without pooling unequal metric supports.', 'tab:supplemental-metrics')
    with (out/'all_methods.tex').open('a') as handle:handle.write('\\input{generated/supplemental_metrics.tex}\n')
    write_slices(out, sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='manuscript/chemical_papers');args=parser.parse_args();output=ROOT/args.output
    figures=output/'figures';generated=output/'generated';figures.mkdir(parents=True,exist_ok=True);generated.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'savefig.dpi':600})
    sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for name in ['final_figures_v1','final_consistency_figures_v1','final_resources_v1']:
        manifest=REPORT/name/'manifest.json';info=read(manifest);sources[str(manifest.relative_to(ROOT))]=sha(manifest)
        for filename,digest in info['outputs'].items():
            if sha(manifest.parent/filename)!=digest:raise ValueError('Audited final source changed')
    data=pd.read_csv(REPORT/'final_figures_v1/summary.csv');con=pd.read_csv(REPORT/'final_consistency_figures_v1/summary.csv');timing=pd.read_csv(REPORT/'final_resources_v1/latency_observations.csv')
    if len(data)!=102 or len(con)!=720 or len(timing)!=62:raise ValueError('Incomplete final source set')
    for p in [REPORT/'final_figures_v1/summary.csv',REPORT/'final_consistency_figures_v1/summary.csv',REPORT/'final_resources_v1/latency_observations.csv',ROOT/'artifacts/audit/dualqa/independent_final_scores_v1.json',ROOT/'artifacts/audit/sourceqa/data_audit.json',ROOT/'artifacts/audit/strictqa/data_audit.json',ROOT/'configs/study_registry_dualqa_v2.json']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    comp={}
    for protocol,_ in PROTOCOLS:
        p=ROOT/f'artifacts/reports/frozen_dualqa/{protocol}_primary_condition_minus_structure.json';comp[protocol]=read(p);sources[str(p.relative_to(ROOT))]=sha(p)
        if comp[protocol]['status']!='FROZEN_TEST_COMPARISON':raise ValueError('Only frozen final contrasts')
    selected,checks=case_records(sources)
    workflow(figures);performance(data,figures);contrasts(comp,figures);consistency(con,figures);cases(selected,figures)
    write_tables(data,con,comp,timing,selected,checks,generated,sources)
    notes={
        'fig1_workflow': 'Real executed protocol. Training banks and the shared test have distinct roles; arrows are conceptual data flow, not chronological audit gates. Property baselines do not use language-model retrieval.',
        'fig2_performance': 'Seven fixed core groups, both protocols, all three principal metrics. All 34 groups and 62 runs remain in all_methods and per_seed. Confidence intervals resample solutes after averaging complete learning-seed groups; they do not quantify training or experimental noise.',
        'fig3_contrasts': 'Condition minus structure on the same seed-specific adapter. Accuracy differences are in percentage points; regret is in dex. Solid vertical line: zero; dashed lines: the prespecified 2 pp pair-accuracy and -0.05 dex regret meaningful thresholds. Top-1 is supplemental. Neither protocol supports both prespecified criteria jointly.',
        'fig4_consistency': 'The main figure shows the primary identity protocol; scaffold and all 36 groups remain in the supplement. Rows: option-order then equivalent-SMILES transforms. Columns: pair mapped-answer, ranking first-choice, and complete-ranking agreement. Invalid outputs retain penalties; consistency does not establish correctness.',
        'fig5_cases': 'Original frozen greatest-regret seed 17 pair and rank queries: both concern the SAME resveratrol solute at 283.15 K, DOI 10.1016/j.molliq.2020.113363. The dataset lacks pH, polymorph and equilibration metadata. Descriptive extreme-case selection is not a representative error rate or causal mechanism. All three seed outcomes are verified against saved per-query scores. RDKit black-and-white SVG paths are copied without geometric approximation into vector PDF; unsupported path commands fail.',
        'graphical_abstract': '80 x 30 mm vector conceptual workflow; no efficacy or chemical mechanism assertion.',
    }
    (figures/'captions_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    manifest={'status':'GENERATED_FROM_AUDITED_FROZEN_RESULTS','sources':sources,'figure_width_mm':174,'graphical_abstract_mm':[80,30],'png_dpi':600,
              'core_methods':list(CORE),'all_method_groups':34,'all_consistency_groups':36,'case_selection':{'rule':'First greatest_regret case (selection_rank == 1) of each task in frozen primary identity condition-retrieval SFT seed 17 results','queries':[r['query_id'] for r in selected],'unique_solutes':len({r['solute_id'] for r in selected}),'across_seed_checks':checks},
              'outputs':{str(p.relative_to(output)):sha(p) for folder in [figures,generated] for p in sorted(folder.iterdir()) if p.is_file()},'models_invoked':False,'new_scientific_tests':False,'notes':notes}
    (output/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':6,'tables':len(list(generated.glob('*.tex'))),'case_seed_checks':len(checks),'output':str(output)}))


if __name__=='__main__':main()
