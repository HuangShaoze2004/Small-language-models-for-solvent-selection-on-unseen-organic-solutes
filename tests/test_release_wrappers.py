"""Release helpers must verify bytes and preserve existing scientific files."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figure_fixture(root, name='data/example.txt', payload=b'fixed measurement'):
    (root / 'reproducibility').mkdir()
    (root / 'scripts').mkdir()
    code = root / 'scripts/build_chemical_papers_assets.py'
    code.write_bytes(b'recorded code')
    archive = root / 'reproducibility/figure_inputs.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        entry = tarfile.TarInfo(name)
        entry.size = len(payload)
        stream.addfile(entry, io.BytesIO(payload))
    digest = lambda value: hashlib.sha256(value).hexdigest()
    manifest = {'archive': archive.name, 'archive_sha256': digest(archive.read_bytes()),
                'code_sha256': digest(code.read_bytes()),
                'members': {name: {'bytes': len(payload), 'sha256': digest(payload)}}}
    (root / 'reproducibility/figure_inputs_manifest.json').write_text(json.dumps(manifest))


def test_figure_install_is_verified_and_idempotent(tmp_path, monkeypatch):
    figure_fixture(tmp_path)
    module = load('prepare_figure_inputs')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    module.main()
    module.main()
    assert (tmp_path / 'data/example.txt').read_bytes() == b'fixed measurement'


def test_figure_install_refuses_changed_scientific_input(tmp_path, monkeypatch):
    figure_fixture(tmp_path)
    (tmp_path / 'data').mkdir()
    destination = tmp_path / 'data/example.txt'
    destination.write_bytes(b'different existing observation')
    module = load('prepare_figure_inputs')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(FileExistsError):
        module.main()
    assert destination.read_bytes() == b'different existing observation'


def test_figure_archive_cannot_escape_project(tmp_path, monkeypatch):
    figure_fixture(tmp_path, name='../escaped.txt')
    module = load('prepare_figure_inputs')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Unsafe'):
        module.main()
    assert not (tmp_path.parent / 'escaped.txt').exists()


def data_fixture(root, contents):
    path = root / 'data/scientific.txt'
    path.parent.mkdir()
    path.write_bytes(contents)
    (root / 'reproducibility').mkdir()
    expected = hashlib.sha256(b'original').hexdigest()
    (root / 'reproducibility/data_expected.json').write_text(json.dumps({'outputs': {'data/scientific.txt': expected}}))
    return path


def test_data_wrapper_stops_before_overwriting_existing_data(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    destination = data_fixture(tmp_path, b'changed')
    module = load('reproduce_paper_data')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['reproduce_paper_data.py'])
    with pytest.raises(FileExistsError):
        module.main()
    assert destination.read_bytes() == b'changed'


def test_data_wrapper_reuses_already_verified_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    data_fixture(tmp_path, b'original')
    module = load('reproduce_paper_data')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['reproduce_paper_data.py'])
    module.main()
    assert 'ALREADY_VERIFIED' in capsys.readouterr().out
