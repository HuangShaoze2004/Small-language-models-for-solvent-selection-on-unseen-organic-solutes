#!/usr/bin/env python3
"""Download official fixed NIST archive; extract only preselected training DOIs."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import re
import tarfile
import requests

URL = 'https://data.nist.gov/od/ds/mds2-2422/ThermoML.v2020-09-30.tgz'
SHA256 = '231161b5e443dc1ae0e5da8429d86a88474cb722016e5b790817bb31c58d7ec2'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path('data/raw/source_audit')
    archive = root / 'ThermoML.v2020-09-30.tgz'
    selection_path = Path('artifacts/audit/source_discrepancy/nist_training_source_selection.json')
    selected = set(json.loads(selection_path.read_text())['source_dois'])
    if not archive.exists():
        partial = archive.with_suffix('.tgz.partial')
        # The archive is modest; an incomplete unverified transfer is restarted safely.
        with requests.get(URL, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
        if digest(partial) != SHA256:
            raise ValueError('Official archive SHA256 mismatch; no source is extracted')
        partial.replace(archive)
    assert digest(archive) == SHA256
    destination = root / 'nist_selected_jsons'
    destination.mkdir(exist_ok=True)
    files = {}
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            match = re.search(r'(10\.\d{4,9}/.+)\.json$', member.name)
            if not member.isfile() or not match:
                continue
            doi = match[1].lower()
            if doi not in selected:
                continue
            if member.size > 100 * 1024**2:
                raise ValueError('Unexpectedly large selected JSON')
            content = source.extractfile(member).read()
            document = json.loads(content)
            assert document['Citation']['sDOI'].lower() == doi
            path = destination / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
            path.write_bytes(content)
            files[doi] = {'path': str(path), 'sha256': digest(path), 'archive_member': member.name}
    result = {'status': 'DOWNLOADED_TRAINING_SOURCE_AUDIT_ONLY', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'archive_url': URL, 'archive_sha256': SHA256, 'archive_size_bytes': archive.stat().st_size,
              'selection_path': str(selection_path), 'selection_sha256': digest(selection_path),
              'selected_sources': len(selected), 'extracted_sources': len(files),
              'missing_from_fixed_archive': sorted(selected - files.keys()), 'files': files,
              'scope': 'Only metadata-selected primary-training-only DOI JSONs are extracted/parsed; no heldout target source files enter training or model tools.',
              'limitations': 'NIST states this public archive is not critically evaluated and can omit internal error flags. Archive disagreement is not automatic proof that either value is correct.',
              'official_metadata_page': 'https://data.nist.gov/pdr/lps/ark%3A/88434/mds2-2422/pdr%3Av/1.1.0'}
    out = Path('artifacts/audit/source_discrepancy/nist_download_manifest.json')
    if out.exists():
        previous = json.loads(out.read_text())
        for key in ['archive_sha256', 'selection_sha256', 'files', 'missing_from_fixed_archive']:
            if previous.get(key) != result[key]:
                raise ValueError('Existing immutable source manifest differs: ' + key)
        result = previous
    else:
        out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status','selected_sources','extracted_sources','missing_from_fixed_archive']}))


if __name__ == '__main__':
    main()
