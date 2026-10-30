#!/usr/bin/env python3
"""Rebuild frozen scientific data in isolation, then install missing outputs.

This replays the fixed source policies; it does not re-audit ThermoML or select
new sources. Original scientific modules and thresholds remain unchanged.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, help='New, nonexistent staging directory; default is project .tmp')
    args = parser.parse_args()
    os.chdir(ROOT)
    expected = json.loads((ROOT / 'reproducibility/data_expected.json').read_text())['outputs']
    for name, digest in expected.items():
        destination = ROOT / name
        if destination.exists() and (not destination.is_file() or sha(destination) != digest):
            raise FileExistsError('Existing scientific output differs: ' + name)
    if all((ROOT / name).is_file() for name in expected):
        print(json.dumps({'status': 'ALREADY_VERIFIED', 'scientific_outputs': len(expected)}))
        return
    raw = ROOT / 'data/raw/BigSolDBv2.0.csv'
    if not raw.is_file():
        raise FileNotFoundError('Run python scripts/download_data.py first')
    if sha(raw) != '879f55f39db977e295c705e304d011d89d6e10da1daaa3f069181108c1de38b0':
        raise ValueError('Original archive checksum mismatch')
    (ROOT / '.tmp').mkdir(exist_ok=True)
    if args.stage:
        stage = args.stage.resolve()
        stage.mkdir(parents=True, exist_ok=False)
    else:
        stage = Path(tempfile.mkdtemp(prefix='paper_data_rebuild_', dir=ROOT / '.tmp'))
    for directory in ['configs', 'data/raw']:
        (stage / directory).mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'configs').glob('*'):
        if path.is_file():
            shutil.copy2(path, stage / 'configs' / path.name)
    # The original strict-split amendment records the previous split before
    # rebuilding it. Supply that frozen, label-free metadata in the new stage.
    (stage / 'data/splits').mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'data/splits').glob('*.csv'):
        shutil.copy2(path, stage / 'data/splits' / path.name)
    shutil.copy2(raw, stage / 'data/raw' / raw.name)
    policy = json.loads((ROOT / 'configs/strict_quarantine_v1.json').read_text())
    evidence = policy['source_summary']
    (stage / evidence).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / evidence, stage / evidence)
    start = time.monotonic()
    log = stage / 'rebuild.log'
    print('Rebuilding fixed data in ' + str(stage), flush=True)
    from solvent_select.data import build as initial
    from solvent_select.source_quarantine import build as primary
    from solvent_select.strict_quarantine import build as strict
    from solvent_select.views import build_view
    with log.open('w') as handle, contextlib.redirect_stdout(handle):
        initial(stage, ROOT / 'configs/data.yaml')
        primary(stage, ROOT / 'configs/source_quarantine_v1_reproduce.json')
        strict(stage, stage)
        for tag, protocol in [('', 'identity'), ('', 'scaffold'), ('sourceqa', 'identity'),
                              ('sourceqa', 'scaffold'), ('strictqa', 'scaffold')]:
            build_view(protocol, stage / 'data/processed' / tag,
                       stage / 'data/targets' / tag, stage / 'data/views' / tag)
    checked = {}
    for name, digest in expected.items():
        checked[name] = sha(stage / name)
        if checked[name] != digest:
            raise ValueError('Rebuilt scientific output differs: ' + name + '; see ' + str(log))
    # Check every existing destination before installing any newly generated file.
    for name, digest in expected.items():
        if (ROOT / name).exists() and sha(ROOT / name) != digest:
            raise FileExistsError('Destination changed during reconstruction: ' + name)
    installed = 0
    for name in expected:
        destination = ROOT / name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (stage / name).open('rb') as source, destination.open('xb') as output:
            shutil.copyfileobj(source, output)
        installed += 1
    report = {'status': 'PASSED_BYTE_IDENTICAL', 'scientific_outputs': len(checked),
              'installed': installed, 'source_policy': 'Existing frozen source evidence, not a new ThermoML audit',
              'elapsed_seconds': time.monotonic() - start, 'models_invoked': False,
              'existing_outputs_overwritten': False, 'sha256': checked}
    (ROOT / '.tmp/paper_data_reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sha256'}))


if __name__ == '__main__':
    main()
