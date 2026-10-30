#!/usr/bin/env python3
"""Metadata-select strict-training-only sources, then run the fixed numeric audit."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile

import pandas as pd

from download_thermoml_audit import SHA256,URL
from solvent_select.source_audit import compare_training
from solvent_select.views import sha256_file


def main():
    policy_path=Path('configs/strict_source_audit_v1.json')
    policy=json.loads(policy_path.read_text())
    if sha256_file(policy['measurement_path'])!=policy['measurement_sha256'] or sha256_file(policy['split_path'])!=policy['split_sha256']:
        raise ValueError('Registered original strict inputs changed')
    output=Path('artifacts/audit/strictqa/training_sources')
    if output.exists():raise FileExistsError('Preserve existing strict numeric audit')
    output.mkdir(parents=True)
    manifest=pd.read_csv(policy['split_path'],dtype={'solute_id':str})
    identity=pd.read_parquet(policy['measurement_path'],columns=['solute_id','source_doi'])
    identity=identity.merge(manifest[['solute_id','split']],on='solute_id',validate='many_to_one')
    if identity.split.isna().any():raise ValueError('Missing strict assignment')
    training=set(identity.loc[identity.split=='train','source_doi'])
    forbidden=set(identity.loc[identity.split!='train','source_doi'])
    eligible=training-forbidden
    archive=Path('data/raw/source_audit/ThermoML.v2020-09-30.tgz')
    if sha256_file(archive)!=SHA256:raise ValueError('Official fixed archive checksum mismatch')
    available={}
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            match=re.search(r'(10\.\d{4,9}/.+)\.json$',member.name)
            if member.isfile() and match:available[match[1].lower()]=member.name
    selected=sorted(eligible&available.keys())
    selection={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':policy['scope'],'source_dois':selected,
               'strict_train_sources':len(training),'excluded_cross_partition_sources':sorted(training&forbidden),
               'policy_sha256':sha256_file(policy_path),'measurement_numeric_values_read':False,
               'archive_sha256':SHA256,'selection_before_numeric_comparison':True}
    selection_path=output/'selection.json';selection_path.write_text(json.dumps(selection,indent=2)+'\n')
    destination=Path('data/raw/source_audit/nist_strict_training_jsons');destination.mkdir(exist_ok=True)
    files={}
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            match=re.search(r'(10\.\d{4,9}/.+)\.json$',member.name)
            if not member.isfile() or not match or match[1].lower() not in selected:continue
            doi=match[1].lower()
            if member.size>100*1024**2:raise ValueError('Unexpected selected JSON size')
            content=source.extractfile(member).read()
            if json.loads(content)['Citation']['sDOI'].lower()!=doi:raise ValueError('DOI differs from archive metadata')
            path=destination/(hashlib.sha256(doi.encode()).hexdigest()+'.json');path.write_bytes(content)
            files[doi]={'path':str(path),'sha256':sha256_file(path),'archive_member':member.name}
    if set(files)!=set(selected):raise ValueError('Missing selected strict training sources')
    download=output/'download_manifest.json'
    download.write_text(json.dumps({'status':'DOWNLOADED_STRICT_TRAINING_SOURCE_AUDIT_ONLY','files':files,
        'archive_url':URL,'archive_sha256':SHA256,'selection_path':str(selection_path),
        'selection_sha256':sha256_file(selection_path),'scope':policy['scope']},indent=2)+'\n')
    audit=compare_training(policy['measurement_path'],policy['split_path'],download,output/'numeric',
                           threshold=policy['threshold_dex'],selection_rule=policy['scope'])
    print(json.dumps({'selected_sources':len(selected),'matches':audit['exact_unambiguous_training_matches'],
                      'flagged_sources':audit['flagged_sources'],'flagged_records':audit['flagged_records']}))


if __name__=='__main__':main()
