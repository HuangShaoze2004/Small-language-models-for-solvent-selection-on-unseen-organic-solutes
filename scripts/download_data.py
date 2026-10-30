#!/usr/bin/env python3
"""Acquire immutable public source files with atomic writes and checksum checks."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'BigSolDBv2.0.csv': (
        'https://zenodo.org/api/records/15094979/files/BigSolDBv2.0.csv/content',
        '879f55f39db977e295c705e304d011d89d6e10da1daaa3f069181108c1de38b0'),
    'BigSolDBv2.0_densities.csv': (
        'https://zenodo.org/api/records/15094979/files/BigSolDBv2.0_densities.csv/content', None),
}

def main():
    target = ROOT / 'data/raw'
    target.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, (url, expected) in FILES.items():
        path = target / name
        if not path.exists():
            for attempt in range(4):
                try:
                    print('Downloading', url, flush=True)
                    with urllib.request.urlopen(url, timeout=60) as response, open(str(path)+'.part', 'wb') as f:
                        while block := response.read(1024 * 1024):
                            f.write(block)
                    digest = hashlib.sha256(Path(str(path)+'.part').read_bytes()).hexdigest()
                    if expected and digest != expected:
                        raise ValueError(f'Checksum mismatch: {name}: {digest}')
                    os.replace(str(path)+'.part', path)
                    break
                except Exception as exc:
                    print(f'Attempt {attempt+1} failed: {exc}', flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected and digest != expected:
            raise ValueError(f'Existing file checksum mismatch: {path}')
        provenance.append({'file': str(path.relative_to(ROOT)), 'url': url,
                           'sha256': digest, 'md5': hashlib.md5(payload).hexdigest(),
                           'bytes': len(payload), 'license': 'CC-BY-4.0',
                           'archive': '10.5281/zenodo.15094979',
                           'accessed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print(name, len(payload), digest, flush=True)
    (target / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')

if __name__ == '__main__':
    main()
