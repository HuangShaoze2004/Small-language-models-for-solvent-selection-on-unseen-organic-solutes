"""Explicit data/model namespaces for the two registered evaluation protocols."""
import json
from pathlib import Path

DEFAULT_REGISTRY='configs/study_registry_dualqa_v2.json'


def load_registry(path=DEFAULT_REGISTRY):
    data=json.loads(Path(path).read_text())
    if data.get('status')!='REGISTERED_VALIDATION_ONLY_DUAL_QA' or set(data['protocols'])!={'identity','scaffold'}:
        raise ValueError('Unknown/incomplete study registry')
    if data['protocols']['identity']['data_tag']!='sourceqa' or data['protocols']['scaffold']['data_tag']!='strictqa':
        raise ValueError('The registered primary and repaired strict banks must remain distinct')
    return data


def queue_pairs(registry):
    return [(q['plan'],q['state']) for q in registry['queues']]


def property_names(protocol,config):
    prefix=config['run_prefix']+protocol+'_'
    return [prefix+'nearest']+[f'{prefix}{kind}_{seed}' for kind in ['lgbm','chemprop'] for seed in [17,42,2026]]


def evidence_name(protocol,config,mode):
    return f"{config['run_prefix']}{protocol}_evidence_{mode}_val"


def verify_adapter_boundary(metadata,config,inference_config):
    from .views import sha256_file
    paths=[Path(config['views'])/f'{split}_{kind}.jsonl' for split in ['train','val'] for kind in ['queries','targets']]
    paths.append(Path(config['split']))
    if metadata['config']['retrieval_mode']!='none':paths.append(Path(config['measurements']))
    expected={str(path.resolve()):sha256_file(path) for path in paths}
    recorded={str(Path(path).resolve()):digest for path,digest in metadata['inputs'].items()}
    if recorded!=expected:raise ValueError('Adapter supervised/retrieval inputs do not match its protocol boundary')
    for key in ['model_id','revision','seed']:
        if metadata['config'][key]!=inference_config[key]:raise ValueError('Adapter model/seed differs from inference')
    return expected
