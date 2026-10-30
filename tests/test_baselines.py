"""Scientific isolation and predictive-behavior tests for classical baselines."""
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from solvent_select.baselines import (
    BaselineConfig, MolecularFeatures, TanimotoNearestNeighbor,
    load_manifest, load_partition, require_frozen_test, sha256_file,
    validate_group_isolation,
)


def measurements():
    return pd.DataFrame([
        {"record_id": "r1", "solute_id": "s1", "solute_smiles": "CCO", "solvent_id": "v1",
         "solvent_smiles": "O", "temperature_k": 298.15, "logx": -2.0, "leakage_group": "g1"},
        {"record_id": "r2", "solute_id": "s1", "solute_smiles": "CCO", "solvent_id": "v1",
         "solvent_smiles": "O", "temperature_k": 310.15, "logx": -1.0, "leakage_group": "g1"},
        {"record_id": "r3", "solute_id": "s2", "solute_smiles": "CCN", "solvent_id": "v2",
         "solvent_smiles": "CCO", "temperature_k": 298.15, "logx": -0.5, "leakage_group": "g2"},
        {"record_id": "r4", "solute_id": "s3", "solute_smiles": "CCCC", "solvent_id": "v1",
         "solvent_smiles": "O", "temperature_k": 298.15, "logx": -5.0, "leakage_group": "g3"},
    ])


def test_prediction_does_not_consume_query_labels():
    train = measurements().iloc[:3]
    model = TanimotoNearestNeighbor().fit(train)
    query = measurements().iloc[[3]].copy()
    first, audit = model.predict(query)
    query["logx"] = 1e99
    second, second_audit = model.predict(query)
    np.testing.assert_array_equal(first, second)
    pd.testing.assert_frame_equal(audit, second_audit)
    assert audit.nearest_record_id.iloc[0] in set(train.record_id)


def test_neighbor_uses_exact_solvent_then_temperature():
    train = measurements().iloc[:3]
    query = train.iloc[[0]].copy()
    query["temperature_k"] = 309.0
    model = TanimotoNearestNeighbor().fit(train)
    prediction, audit = model.predict(query)
    assert prediction[0] == -1.0
    assert audit.nearest_record_id.iloc[0] == "r2"
    # Altering a competing label must not influence which neighbor is selected.
    train.loc[train.record_id.eq("r1"), "logx"] = 1000.0
    _, changed_audit = TanimotoNearestNeighbor().fit(train).predict(query)
    assert changed_audit.nearest_record_id.iloc[0] == "r2"


def test_imputer_is_fitted_only_on_training_data():
    train = measurements().iloc[:3]
    features = MolecularFeatures(BaselineConfig(fingerprint_bits=128)).fit(train)
    medians = features.medians.copy()
    before = features.transform(train).toarray()
    query = measurements().iloc[[3]].copy()
    query["temperature_k"] = 1e10
    features.transform(query)
    np.testing.assert_array_equal(features.medians, medians)
    np.testing.assert_array_equal(features.transform(train).toarray(), before)
    assert np.isfinite(before).all()
    assert before.shape[1] == 256 + 2 * len(features.descriptor_names) + 1


def test_partition_projection_and_sealed_labels(tmp_path):
    frame = measurements()
    data_path = tmp_path / "data.parquet"
    frame.to_parquet(data_path)
    split_path = tmp_path / "split.csv"
    pd.DataFrame({"solute_id": ["s1", "s2", "s3"], "split": ["train", "val", "test"]}).to_csv(split_path, index=False)
    train = load_partition(data_path, split_path, "train", labels=True)
    assert set(train.record_id) == {"r1", "r2"}
    query = load_partition(data_path, split_path, "test", labels=False)
    assert set(query.record_id) == {"r4"}
    assert "logx" not in query
    with pytest.raises(ValueError, match="never loads test labels"):
        load_partition(data_path, split_path, "test", labels=True)


def test_group_leakage_and_duplicate_split_ids_are_rejected(tmp_path):
    frame = measurements()
    frame.loc[frame.solute_id.eq("s3"), "leakage_group"] = "g1"
    data_path = tmp_path / "data.parquet"
    frame.to_parquet(data_path)
    manifest = pd.DataFrame({"solute_id": ["s1", "s2", "s3"], "split": ["train", "val", "test"]})
    with pytest.raises(ValueError, match="crosses partitions"):
        validate_group_isolation(data_path, manifest)
    path = tmp_path / "split.csv"
    pd.concat([manifest, manifest.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly once"):
        load_manifest(path)


def test_test_release_requires_exact_split_hash(tmp_path):
    split_path = tmp_path / "split.csv"
    split_path.write_text("solute_id,split\ns1,test\n")
    with pytest.raises(ValueError, match="requires"):
        require_frozen_test(None, split_path)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"status": "FROZEN", "split_sha256": "bad"}))
    with pytest.raises(ValueError, match="exact split"):
        require_frozen_test(str(freeze), split_path)
    freeze.write_text(json.dumps({"status": "FROZEN", "split_sha256": sha256_file(split_path)}))
    require_frozen_test(str(freeze), split_path)


def test_chemprop_temperature_and_two_graphs_receive_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from solvent_select.chemprop_runner import ChempropConfig, fit_scalers, make_dataset, make_loader, make_model

    torch.set_num_threads(1)
    torch.manual_seed(17)
    frame = measurements().iloc[:3]
    config = ChempropConfig(hidden_dim=16, ffn_hidden_dim=16, depth=2, dropout=0,
                            batch_size=3, num_workers=0, torch_threads=1)
    scalers = fit_scalers(frame)
    dataset = make_dataset(frame, scalers, labels=True)
    batch = next(iter(make_loader(dataset, config, training=False, seed=17)))
    model = make_model(config)
    prediction = model(batch.bmgs, batch.V_ds, batch.X_d)
    assert prediction.shape == (3, 1)
    ((prediction - batch.Y) ** 2).mean().backward()
    for block in model.message_passing.blocks:
        assert sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None) > 0
    # Temperature is appended after the two graph embeddings, so it is the
    # final input column of the first predictor linear layer.
    first_layer = next(m for m in model.predictor.modules() if isinstance(m, torch.nn.Linear))
    assert first_layer.weight.grad[:, -1].abs().sum() > 0
    prediction_dataset = make_dataset(frame.drop(columns="logx"), scalers, labels=False)
    prediction_batch = next(iter(make_loader(prediction_dataset, config, training=False, seed=17)))
    # Chemprop 2.3's dataset may materialize missing targets as NaN tensors.
    assert prediction_batch.Y is None or not torch.isfinite(prediction_batch.Y).any()


def test_nearest_cli_artifact_is_importable_and_test_remains_sealed(tmp_path):
    frame = measurements()
    data_path = tmp_path / "data.parquet"
    frame.to_parquet(data_path)
    split_path = tmp_path / "identity.csv"
    pd.DataFrame({"solute_id": ["s1", "s2", "s3"], "split": ["train", "val", "test"]}).to_csv(split_path, index=False)
    run = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    command = [sys.executable, "-m", "solvent_select.baselines", "train", "--method", "nearest",
               "--data", str(data_path), "--splits", str(split_path), "--output", str(run)]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    saved = joblib.load(run / "model.joblib")
    assert type(saved["model"]).__module__ == "solvent_select.baselines"
    predictions = pd.read_parquet(run / "predictions_val.parquet")
    assert set(predictions.record_id) == {"r3"}
    assert "logx" not in predictions
    blocked = subprocess.run([sys.executable, "-m", "solvent_select.baselines", "predict", "--run", str(run),
                              "--partition", "test"], env=env, capture_output=True, text=True)
    assert blocked.returncode != 0
    assert "requires --freeze-manifest" in blocked.stderr
    assert not (run / "predictions_test.parquet").exists()


def test_chemprop_sigterm_resume_matches_uninterrupted_training(tmp_path):
    """A real child-process SIGTERM must yield a complete, reproducible epoch."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from solvent_select.chemprop_runner import ChempropConfig
    from dataclasses import asdict

    frame = measurements()
    data_path = tmp_path / "measurements.parquet"
    frame.to_parquet(data_path)
    split_path = tmp_path / "identity.csv"
    pd.DataFrame({"solute_id": ["s1", "s2", "s3"], "split": ["train", "val", "test"]}).to_csv(split_path, index=False)
    config = ChempropConfig(hidden_dim=16, ffn_hidden_dim=16, depth=2, dropout=0.1,
                            ffn_layers=1, batch_size=2, max_epochs=3, warmup_epochs=1,
                            num_workers=0, torch_threads=1)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(asdict(config)))
    script = '''
import argparse,json,os,signal,sys
import lightning.pytorch as pl
from solvent_select import chemprop_runner as runner
if sys.argv[2] == "interrupt":
    original = runner.build_callbacks
    class InterruptAfterFirstBatch(pl.Callback):
        def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
            if trainer.current_epoch == 0 and batch_idx == 0:
                os.kill(os.getpid(),signal.SIGTERM)
    def callbacks(*args,**kwargs):
        callbacks,state,checkpoint=original(*args,**kwargs)
        return [InterruptAfterFirstBatch(),*callbacks],state,checkpoint
    runner.build_callbacks=callbacks
runner.train(argparse.Namespace(**json.loads(sys.argv[1])))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    common = {"data": str(data_path), "splits": str(split_path), "config": str(config_path),
              "seed": 17, "accelerator": "cpu", "purpose": "software_check", "resume": False,
              "limit_train_batches": None, "limit_val_batches": None, "gpu_lock": str(tmp_path / "unused_gpu.lock")}
    full, restored = tmp_path / "full", tmp_path / "restored"
    for output, mode, resume in [(full, "full", False), (restored, "interrupt", False), (restored, "resume", True)]:
        args = dict(common, output=str(output), resume=resume)
        completed = subprocess.run([sys.executable, "-u", "-c", script, json.dumps(args), mode],
                                   env=env, capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["status"] == ("interrupted" if mode == "interrupt" else "completed")
    a = torch.load(full / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    b = torch.load(restored / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    assert a["global_step"] == b["global_step"] == 3
    assert a["epoch"] == b["epoch"] == 2
    assert b["optimizer_states"] and b["lr_schedulers"]
    assert b["solvent_select_rng"]["sampler_next_epoch"] == 3
    assert b["solvent_select_rng"]["cuda"] == []
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    pd.testing.assert_frame_equal(pd.read_parquet(full / "predictions_val.parquet"),
                                  pd.read_parquet(restored / "predictions_val.parquet"))


def test_equivalent_known_solvent_does_not_fall_back_to_fingerprint_collision():
    # Radius2 Morgan fingerprints cannot distinguish these long-chain alcohols.
    # A different spelling of a KNOWN solvent must still use its exact identity.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    assert DataStructs.TanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles('CCCCCCCO')),
                                         generator.GetFingerprint(Chem.MolFromSmiles('CCCCCCO'))) == 1
    train = pd.DataFrame([
        dict(record_id='heptanol', solute_id='train', solute_smiles='CCC', solvent_id='seven',
             solvent_smiles='CCCCCCCO', temperature_k=298.2, logx=-1.),
        dict(record_id='hexanol', solute_id='train', solute_smiles='CCC', solvent_id='six',
             solvent_smiles='CCCCCCO', temperature_k=298.15, logx=-9.),
    ])
    model = TanimotoNearestNeighbor().fit(train)
    q = train.iloc[[0]].copy()
    q['temperature_k'] = 298.15
    first, a = model.predict(q)
    q['solvent_smiles'] = 'OCCCCCCC'
    second, b = model.predict(q)
    np.testing.assert_array_equal(first, second)
    assert first[0] == -1
    assert a.nearest_record_id.iloc[0] == b.nearest_record_id.iloc[0] == 'heptanol'
