"""Controlled, from-scratch dual-molecule Chemprop baseline (no pretrained weights).

Training uses train/validation manifests only. Checkpoints are saved after each
epoch, including optimizer/scheduler and RNG states. SIGTERM requests a clean
stop at the end of the current epoch; an abrupt kill resumes the last full epoch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import pandas as pd

from .baselines import (load_manifest, load_partition, require_frozen_test, sha256_file,
                        validate_group_isolation, write_json)


@dataclass
class ChempropConfig:
    hidden_dim: int = 300
    depth: int = 3
    dropout: float = 0.1
    ffn_hidden_dim: int = 300
    ffn_layers: int = 2
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 8
    warmup_epochs: int = 2
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    num_workers: int = 4
    torch_threads: int = 6
    precision: str = "32-true"


def read_config(path: str | None) -> ChempropConfig:
    if path is None:
        return ChempropConfig()
    import yaml
    with open(path) as handle:
        return ChempropConfig(**yaml.safe_load(handle))


@contextmanager
def gpu_lock(path: str | Path):
    """Shared project GPU lease; refuse rather than queue a duplicate GPU job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another project GPU job holds {path}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "started_unix": time.time(), "job": "chemprop"}) + "\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fit_scalers(train: pd.DataFrame) -> dict[str, float]:
    return {"temperature_mean": float(train.temperature_k.mean()),
            "temperature_std": max(float(train.temperature_k.std(ddof=0)), 1e-8),
            "target_mean": float(train.logx.mean()),
            "target_std": max(float(train.logx.std(ddof=0)), 1e-8)}


def make_dataset(frame: pd.DataFrame, scalers: dict[str, float], *, labels: bool):
    """Extra temperature and targets belong to component zero, per Chemprop API."""
    from chemprop import data
    from rdkit import Chem

    solute_points, solvent_points = [], []
    mols = {s: Chem.MolFromSmiles(s) for s in set(frame.solute_smiles) | set(frame.solvent_smiles)}
    if any(mol is None for mol in mols.values()):
        raise ValueError("Invalid molecule reached Chemprop after data audit")
    columns = ["solute_smiles", "solvent_smiles", "temperature_k"] + (["logx"] if labels else [])
    for row in frame[columns].itertuples(index=False, name=None):
        solute, solvent, temperature = row[:3]
        target = np.asarray([(row[3] - scalers["target_mean"]) / scalers["target_std"]], dtype=np.float32) if labels else None
        extra = np.asarray([(temperature - scalers["temperature_mean"]) / scalers["temperature_std"]], dtype=np.float32)
        solute_points.append(data.MoleculeDatapoint(mols[solute], y=target, x_d=extra))
        solvent_points.append(data.MoleculeDatapoint(mols[solvent]))
    components = [data.MoleculeDataset(solute_points), data.MoleculeDataset(solvent_points)]
    for component in components:
        component.cache = True
    return data.MulticomponentDataset(components)


def make_model(config: ChempropConfig):
    from chemprop import models, nn

    blocks = [nn.BondMessagePassing(d_h=config.hidden_dim, depth=config.depth, dropout=config.dropout)
              for _ in range(2)]
    message_passing = nn.MulticomponentMessagePassing(blocks=blocks, n_components=2)
    predictor = nn.RegressionFFN(input_dim=message_passing.output_dim + 1,
                                 hidden_dim=config.ffn_hidden_dim, n_layers=config.ffn_layers,
                                 dropout=config.dropout)
    return models.MulticomponentMPNN(
        message_passing, nn.MeanAggregation(), predictor, batch_norm=False,
        warmup_epochs=config.warmup_epochs, init_lr=config.init_lr,
        max_lr=config.max_lr, final_lr=config.final_lr)


def build_callbacks(checkpoint_dir: Path, seed: int, train_loader, patience: int):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    import torch

    class ReproducibleEpochState(pl.Callback):
        """Epoch-seeded shuffle is reproducible after an epoch-boundary restart."""
        def __init__(self):
            self.stop_requested = False
            self.loaded_state = None

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            state = np.random.get_state()
            checkpoint["solvent_select_rng"] = {
                "python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if pl_module.device.type == "cuda" else [],
                "numpy": {"name": state[0], "keys": state[1].tolist(), "position": state[2],
                          "has_gauss": state[3], "cached_gaussian": state[4]},
                "sampler_next_epoch": train_loader.sampler.next_epoch,
                "loader_generator": train_loader.generator.get_state(),
            }

        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            self.loaded_state = checkpoint.get("solvent_select_rng")
            if self.loaded_state is not None:
                train_loader.sampler.next_epoch = self.loaded_state["sampler_next_epoch"]
                train_loader.generator.set_state(self.loaded_state["loader_generator"].cpu())

        def on_train_start(self, trainer, pl_module):
            # Lightning installs a SIGTERM notifier during setup that otherwise
            # interrupts mid-epoch. Replace it here for our single-process job.
            def stop_at_epoch_end(signum, frame):
                self.stop_requested = True
                print("SIGTERM received: finishing the current epoch and saving its checkpoint.", flush=True)
            signal.signal(signal.SIGTERM, stop_at_epoch_end)
            state = self.loaded_state
            if state is not None:
                random.setstate(state["python"])
                torch.set_rng_state(state["torch"].cpu())
                n = state["numpy"]
                np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32),
                                     n["position"], n["has_gauss"], n["cached_gaussian"]))
                if state["cuda"] and pl_module.device.type == "cuda":
                    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

        def on_validation_end(self, trainer, pl_module):
            if self.stop_requested and not trainer.sanity_checking:
                trainer.should_stop = True

    state = ReproducibleEpochState()
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename="epoch={epoch:03d}-val={val_loss:.5f}",
                                 auto_insert_metric_name=False, monitor="val_loss", mode="min",
                                 save_top_k=1, save_last=True, every_n_epochs=1,
                                 save_on_train_epoch_end=True, save_weights_only=False)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    return [state, early, checkpoint], state, checkpoint


def make_loader(dataset, config: ChempropConfig, *, training: bool, seed: int):
    from chemprop.data.collate import collate_multicomponent
    import torch
    generator = torch.Generator().manual_seed(seed)
    # The sampler owns an epoch counter. Saving it avoids depending on whether
    # Lightning creates a prefetched iterator before on_train_epoch_start.
    class EpochSampler(torch.utils.data.Sampler):
        def __init__(self, count):
            self.count = count
            self.next_epoch = 0

        def __iter__(self):
            local = torch.Generator().manual_seed(seed + self.next_epoch)
            self.next_epoch += 1
            return iter(torch.randperm(self.count, generator=local).tolist())

        def __len__(self):
            return self.count

    return torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size,
        sampler=EpochSampler(len(dataset)) if training else None,
        num_workers=config.num_workers, drop_last=False, generator=generator,
        collate_fn=collate_multicomponent, persistent_workers=False)


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger
    import torch

    config = read_config(args.config)
    if config.num_workers + config.torch_threads > 12:
        raise ValueError("Chemprop CPU worker plus thread budget must not exceed 12")
    torch.set_num_threads(config.torch_threads)
    pl.seed_everything(args.seed, workers=True)
    output = Path(args.output)
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {output}; use --resume for an incomplete run")
    metadata = {"status": "running", "method": "chemprop", "seed": args.seed,
                "purpose": args.purpose,
                "config": asdict(config), "target": "log10_mole_fraction",
                "measurements": str(Path(args.data).resolve()), "splits": str(Path(args.splits).resolve()),
                "measurement_sha256": sha256_file(args.data), "split_sha256": sha256_file(args.splits),
                "code_sha256": sha256_file(__file__), "started_unix": time.time(),
                "execution_limits": {"train_batches": args.limit_train_batches,
                                     "val_batches": args.limit_val_batches},
                "versions": {name: importlib.metadata.version(name) for name in ["chemprop", "torch", "lightning", "rdkit"]},
                "resume_command": f"PYTHONPATH=src .venv-chemprop/bin/python -m solvent_select.chemprop_runner train --data {args.data} --splits {args.splits} --seed {args.seed} --output {args.output}" + (f" --config {args.config}" if args.config else "") + f" --accelerator {args.accelerator} --purpose {args.purpose}" + (f" --limit-train-batches {args.limit_train_batches}" if args.limit_train_batches else "") + (f" --limit-val-batches {args.limit_val_batches}" if args.limit_val_batches else "") + " --resume"}
    if args.resume:
        previous = json.loads(metadata_path.read_text())
        if previous["status"] == "completed":
            raise ValueError("Training is already complete; refusing to start it again")
        for key in ["measurement_sha256", "split_sha256", "config", "seed", "execution_limits", "purpose"]:
            if metadata[key] != previous[key]:
                raise ValueError(f"Resume mismatch: {key}")
        metadata["prior_attempt"] = {"status": previous["status"], "started_unix": previous["started_unix"]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    old_handler = None
    try:
        manifest = load_manifest(args.splits)
        validate_group_isolation(args.data, manifest)
        if Path(args.splits).stem == "scaffold":
            validate_group_isolation(args.data, manifest, "scaffold_group")
        training = load_partition(args.data, args.splits, "train", labels=True)
        validation = load_partition(args.data, args.splits, "val", labels=True)
        scalers = fit_scalers(training)
        write_json(output / "scalers.json", scalers)
        train_set = make_dataset(training, scalers, labels=True)
        val_set = make_dataset(validation, scalers, labels=True)
        train_loader = make_loader(train_set, config, training=True, seed=args.seed)
        val_loader = make_loader(val_set, config, training=False, seed=args.seed)
        model = make_model(config)
        callbacks, state, checkpoint = build_callbacks(output / "checkpoints", args.seed, train_loader, config.patience)
        old_handler = signal.getsignal(signal.SIGTERM)
        def request_stop(signum, frame):
            state.stop_requested = True
            print("SIGTERM received: completing this epoch, validation and checkpoint before stopping.", flush=True)
        signal.signal(signal.SIGTERM, request_stop)
        trainer = pl.Trainer(
            accelerator=args.accelerator, devices=1, max_epochs=config.max_epochs,
            callbacks=callbacks, logger=CSVLogger(str(output), name="training_logs"),
            deterministic=True, precision=config.precision, enable_progress_bar=False,
            log_every_n_steps=25, num_sanity_val_steps=0,
            limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
            limit_val_batches=args.limit_val_batches if args.limit_val_batches else 1.0)
        resume_path = output / "checkpoints" / "last.ckpt"
        if args.resume and not resume_path.exists():
            raise FileNotFoundError("No complete last.ckpt exists; use a new run ID after diagnosing the failed startup")
        if args.accelerator == "gpu":
            torch.cuda.reset_peak_memory_stats()
        with gpu_lock(args.gpu_lock) if args.accelerator == "gpu" else _nullcontext():
            trainer.fit(model, train_loader, val_loader, ckpt_path=str(resume_path) if args.resume else None)
            best_path = checkpoint.best_model_path
            if not best_path:
                raise RuntimeError("Training produced no validated checkpoint")
            saved = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["state_dict"])
            # Prediction dataset contains no validation labels, even though they
            # were legitimately available to the early-stopping callback.
            prediction_set = make_dataset(validation.drop(columns="logx"), scalers, labels=False)
            prediction_loader = make_loader(prediction_set, config, training=False, seed=args.seed)
            predictions = trainer.predict(model, dataloaders=prediction_loader)
        values = torch.cat(predictions).cpu().numpy().reshape(-1) * scalers["target_std"] + scalers["target_mean"]
        result = validation[["record_id", "split"]].copy()
        result["pred_logx"] = values
        result.to_parquet(output / "predictions_val.parquet", index=False)
        metadata.update(status="interrupted" if state.stop_requested else "completed",
                        elapsed_seconds=time.perf_counter() - started, finished_unix=time.time(),
                        best_checkpoint=str(Path(best_path).resolve()), last_checkpoint=str(resume_path.resolve()),
                        completed_epochs=int(trainer.current_epoch), optimizer_steps=int(trainer.global_step),
                        train_records=len(training), val_records=len(validation),
                        validation_rmse=float(np.sqrt(np.mean((values - validation.logx.to_numpy()) ** 2))),
                        gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if args.accelerator == "gpu" else 0)
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


@contextmanager
def _nullcontext():
    yield


def predict(args: argparse.Namespace) -> None:
    import lightning.pytorch as pl
    import torch

    run = Path(args.run)
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Only completed training runs can be released for prediction")
    config = ChempropConfig(**metadata["config"])
    torch.set_num_threads(config.torch_threads)
    if sha256_file(metadata["measurements"]) != metadata["measurement_sha256"] or sha256_file(metadata["splits"]) != metadata["split_sha256"]:
        raise ValueError("Training data or split manifest changed")
    if args.partition == "test":
        require_frozen_test(args.freeze_manifest, metadata["splits"])
    query = load_partition(metadata["measurements"], metadata["splits"], args.partition, labels=False)
    scalers = json.loads((run / "scalers.json").read_text())
    dataset = make_dataset(query, scalers, labels=False)
    loader = make_loader(dataset, config, training=False, seed=metadata["seed"])
    model = make_model(config)
    checkpoint = torch.load(metadata["best_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    started = time.perf_counter()
    with gpu_lock(args.gpu_lock) if args.accelerator == "gpu" else _nullcontext():
        predicted = trainer.predict(model, dataloaders=loader)
    values = torch.cat(predicted).cpu().numpy().reshape(-1) * scalers["target_std"] + scalers["target_mean"]
    result = query[["record_id", "split"]].copy()
    result["pred_logx"] = values
    path = Path(args.output) if args.output else run / f"predictions_{args.partition}.parquet"
    result.to_parquet(path, index=False)
    write_json(path.with_suffix(".timing.json"), {"records": len(query), "elapsed_seconds": time.perf_counter() - started,
               "partition": args.partition, "checkpoint_sha256": sha256_file(metadata["best_checkpoint"])})
    print(f"Wrote {len(result)} predictions to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train")
    tr.add_argument("--data", default="data/processed/measurements.parquet")
    tr.add_argument("--splits", default="data/splits/identity.csv")
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--config")
    tr.add_argument("--output", required=True)
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--purpose", choices=["formal", "pilot", "software_check"], default="formal")
    tr.add_argument("--limit-train-batches", type=int)
    tr.add_argument("--limit-val-batches", type=int)
    tr.set_defaults(func=train)
    pr = commands.add_parser("predict")
    pr.add_argument("--run", required=True)
    pr.add_argument("--partition", choices=["train", "val", "test"], default="val")
    pr.add_argument("--freeze-manifest")
    pr.add_argument("--output")
    pr.set_defaults(func=predict)
    for sub in [tr, pr]:
        sub.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
        sub.add_argument("--gpu-lock", default="artifacts/gpu.lock")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
