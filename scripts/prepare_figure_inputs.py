#!/usr/bin/env python3
"""Install verified local figure inputs without downloads or model execution."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads((ROOT / 'reproducibility/figure_inputs_manifest.json').read_text())
    archive = ROOT / 'reproducibility' / spec['archive']
    if sha(archive) != spec['archive_sha256']:
        raise ValueError('Figure archive checksum mismatch')
    if sha(ROOT / 'scripts/build_chemical_papers_assets.py') != spec['code_sha256']:
        raise ValueError('Figure implementation differs from the recorded version')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if {m.name for m in members} != set(spec['members']) or len(members) != len(spec['members']):
            raise ValueError('Unexpected or duplicate archive member')
        payloads = {}
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError('Unsafe archive member')
            destination = ROOT / member.name
            if destination.resolve() != destination.absolute():
                raise ValueError('Archive path traverses a symlink')
            entry = spec['members'][member.name]
            with stream.extractfile(member) as handle:
                payload = handle.read()
            if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError('Archive member checksum mismatch: ' + member.name)
            if destination.exists() and (not destination.is_file() or sha(destination) != entry['sha256']):
                raise FileExistsError('Existing scientific input differs: ' + member.name)
            payloads[member.name] = payload
        installed = 0
        for name, payload in payloads.items():
            destination = ROOT / name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(payload)
            installed += 1
    print(json.dumps({'status': 'VERIFIED', 'members': len(members), 'installed': installed,
                      'models_invoked': False, 'existing_inputs_overwritten': False}))


if __name__ == '__main__':
    main()
