#!/usr/bin/env python3
"""Compare rebuilt paper outputs with fixed checksums; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    expected = json.loads((ROOT / 'results/paper_assets/asset_manifest.json').read_text())['outputs']
    bad = [name for name, digest in expected.items()
           if not (args.directory / name).is_file()
           or hashlib.sha256((args.directory / name).read_bytes()).hexdigest() != digest]
    print(json.dumps({'status': 'FAILED' if bad else 'PASSED_BYTE_IDENTICAL',
                      'outputs': len(expected), 'errors': bad}))
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
