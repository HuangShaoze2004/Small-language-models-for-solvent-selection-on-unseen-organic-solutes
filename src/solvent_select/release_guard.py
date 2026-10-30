"""Verify an actual frozen release before final prediction or outcome access.

Hashes target bytes without deserializing their contents. This read-only guard
does not create a release, execute a model or calculate any test score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .freeze import option, test_prediction_spec
from .llm import read_jsonl, validate_query_manifest
from .views import sha256_file
from .study import load_registry


def verify_assets(assets):
    if not assets:
        raise ValueError('Frozen asset group is empty')
    for path, expected in assets.items():
        if not Path(path).is_file() or sha256_file(path) != expected:
            raise ValueError('Frozen asset missing or modified: ' + str(path))


def verify_release(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get('status') != 'FROZEN':
        raise ValueError('Only an actual FROZEN release may authorize final execution')
    for field in ['protocol', 'query', 'target', 'split', 'consistency_manifest',
                  'decision', 'pretest_review', 'foundation_model_snapshot']:
        verify_assets({data[field + '_path']: data[field + '_sha256']})
    for field in ['property_assets', 'code_sha256', 'scripts_sha256', 'environment_locks',
                  'validation_consistency_sha256']:
        verify_assets(data[field])
    verify_assets({'src/solvent_select/error_analysis.py':data['error_analysis_code_sha256'],
                   'configs/error_analysis.yaml':data['error_analysis_config_sha256']})
    protocol = Path(data['split_path']).stem
    if protocol not in ['identity', 'scaffold']:
        raise ValueError('Unknown frozen split protocol')
    analysis = json.loads(Path(data['protocol_path']).read_text())
    if analysis.get('status') != 'REGISTERED_BEFORE_TEST':
        raise ValueError('Analysis plan is not registered')
    if 'registry_path' in data:
        verify_assets({data['registry_path']:data['registry_sha256']})
        registry=load_registry(data['registry_path'])
        config=registry['protocols'][protocol]
        if data['split_path']!=config['split'] or data['measurement_path']!=config['measurements'] or data['runs_root']!=config['runs_root']:
            raise ValueError('Frozen protocol data paths differ from the registered namespace')
        verify_assets(data['adapter_assets'])
    rows = read_jsonl(data['query_path'])
    validate_query_manifest(rows, data['split_path'])
    if len(rows) != data['query_count'] or {q['split'] for q in rows} != {'test'}:
        raise ValueError('Frozen public query cohort changed')
    consistency = json.loads(Path(data['consistency_manifest_path']).read_text())
    if consistency.get('split') != 'test' or consistency.get('split_sha256') != data['split_sha256']:
        raise ValueError('Wrong consistency partition or split')
    verify_assets({str(Path(data['consistency_manifest_path']).parent / name): digest
                   for name, digest in consistency['files'].items()})
    specifications = data['prediction_specifications']
    if not specifications:
        raise ValueError('No prediction specifications were frozen')
    registered = data['prediction_fingerprints']
    if len(registered) != len(set(registered)) or registered != [s['sha256'] for s in specifications]:
        raise ValueError('Prediction registration is duplicate or inconsistent')
    method_prefix=(config['run_prefix'] if 'registry_path' in data else 'sourceqa_')+protocol+'_'
    expected_methods=[name for name in analysis['llm_validation_tasks_to_carry_to_test']
                      if name.startswith(method_prefix)]
    actual_methods=[option(s['validation_command'],'--run-id') for s in specifications
                    if s['prediction_kind']=='llm' and not s['consistency']]
    if sorted(actual_methods)!=sorted(expected_methods):
        raise ValueError('Frozen LM methods differ from the registered analysis')
    if [m['name'] for m in data['property_models']]!=analysis['property_methods'][protocol]:
        raise ValueError('Frozen property methods differ from the registered analysis')
    expected_transforms=[json.dumps(s['validation_command']) for s in specifications
                         if s['prediction_kind']=='llm' and not s['consistency']
                         and s['retrieval_mode'] in ['none','text','structure','condition']]
    actual_transforms=[json.dumps(s['validation_command']) for s in specifications if s['prediction_kind']=='llm' and s['consistency']]
    if sorted(expected_transforms)!=sorted(actual_transforms):
        raise ValueError('Frozen LM consistency comparisons are incomplete')
    expected_controls=[(m['run'],False) for m in data['property_models']]
    expected_controls += [(m['run'],True) for m in data['property_models'] if m['method']=='chemprop']
    actual_controls=[(c['model_run'],c['canonicalize_query']) for c in data['cpu_consistency_controls'] if c['kind']=='property']
    evidence_controls=[(c.get('retrieval_mode'),c['canonicalize_query']) for c in data['cpu_consistency_controls'] if c['kind']=='evidence']
    if sorted(expected_controls)!=sorted(actual_controls) or evidence_controls!=[('text',True)]:
        raise ValueError('Frozen CPU consistency controls are incomplete or unregistered')
    for consistency in [False,True]:
        actual_modes=[s['retrieval_mode'] for s in specifications if s['prediction_kind']=='evidence' and s['consistency']==consistency]
        if sorted(actual_modes)!=sorted(analysis['direct_evidence_modes']):
            raise ValueError('Frozen evidence methods differ from the registered analysis')
    for spec in specifications:
        fresh = test_prediction_spec(spec['validation_command'], protocol, path.parent, spec['consistency'])
        if fresh['fingerprint'] != spec['fingerprint'] or fresh['sha256'] != spec['sha256']:
            raise ValueError('Frozen model/configuration fingerprint changed')
    for receipt in data['queue_receipts']:
        verify_assets({receipt['plan']: receipt['plan_sha256'], receipt['state']: receipt['state_sha256']})
    # Verify the actual foundation weights as well as the small manifest file.
    snapshot = json.loads(Path(data['foundation_model_snapshot_path']).read_text())
    model_versions={(s['fingerprint']['config']['model_id'],s['fingerprint']['config']['revision']) for s in specifications}
    if model_versions!={(snapshot['model'],snapshot['revision'])}:
        raise ValueError('Foundation snapshot and prediction model versions disagree')
    verify_assets({str(Path(snapshot['path']) / item['name']): item['sha256'] for item in snapshot['files']})
    return {'status': 'VERIFIED_FROZEN_RELEASE', 'release': str(path), 'release_sha256': sha256_file(path),
            'protocol': protocol, 'queries': len(rows), 'prediction_specifications': len(specifications),
            'test_target_values_read': False, 'models_executed': 0, 'test_scores_calculated': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.release), indent=2))


if __name__ == '__main__':
    main()
