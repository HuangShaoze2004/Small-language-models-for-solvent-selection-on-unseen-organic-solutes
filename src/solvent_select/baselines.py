"""Train-only classical property baselines and label-free prediction.

Examples (run from the repository with PYTHONPATH=src):
  python -m solvent_select.baselines train --method lightgbm --seed 17 \
      --output artifacts/runs/identity_lgbm_17
  python -m solvent_select.baselines predict --run artifacts/runs/identity_lgbm_17

The split manifest is authoritative. No random row splitting is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


FEATURE_COLUMNS = ["record_id", "solute_id", "solute_smiles", "solvent_id",
                   "solvent_smiles", "temperature_k"]
SPLITS = {"train", "val", "test"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"solute_id": str, "split": str})
    if not {"solute_id", "split"}.issubset(manifest.columns):
        raise ValueError("Split manifest must contain solute_id and split")
    if manifest.solute_id.isna().any() or manifest.solute_id.duplicated().any():
        raise ValueError("Each solute_id must occur exactly once in the split manifest")
    if not set(manifest.split).issubset(SPLITS):
        raise ValueError("Unknown split in manifest")
    return manifest[["solute_id", "split"]]


def validate_group_isolation(data_path: str | Path, manifest: pd.DataFrame,
                             group_column: str = "leakage_group") -> None:
    """Read only identity columns; do not inspect held-out target values."""
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(data_path).names)
    if group_column not in schema_names:
        raise ValueError(f"Measurement table lacks required isolation field {group_column}")
    identity = pd.read_parquet(data_path, columns=["solute_id", group_column]).drop_duplicates()
    identity["solute_id"] = identity.solute_id.astype(str)
    joined = identity.merge(manifest, on="solute_id", how="left", validate="many_to_one")
    if joined.split.isna().any():
        raise ValueError("Measurement solutes missing from split manifest")
    if joined[group_column].isna().any():
        raise ValueError(f"Missing {group_column} values")
    counts = joined.groupby(group_column).split.nunique()
    if (counts > 1).any():
        raise ValueError(f"{group_column} crosses partitions")


def load_partition(data_path: str | Path, split_path: str | Path,
                   partition: str, *, labels: bool = False) -> pd.DataFrame:
    """Column projection and parquet filters prevent loading test labels at training time."""
    if partition not in SPLITS:
        raise ValueError(f"Unknown partition {partition}")
    if labels and partition == "test":
        raise ValueError("Baseline model code never loads test labels; use the separate evaluator")
    manifest = load_manifest(split_path)
    ids = manifest.loc[manifest.split.eq(partition), "solute_id"].tolist()
    if not ids:
        raise ValueError(f"No solutes in {partition}")
    columns = FEATURE_COLUMNS + (["logx"] if labels else [])
    result = pd.read_parquet(data_path, columns=columns, filters=[("solute_id", "in", ids)])
    result["solute_id"] = result.solute_id.astype(str)
    if result.empty or not set(result.solute_id).issubset(ids):
        raise ValueError("Empty or inconsistent partition")
    if result.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs")
    result["split"] = partition
    result = result.sort_values("record_id", kind="stable").reset_index(drop=True)
    if not np.isfinite(result.temperature_k.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite temperatures")
    if labels and not np.isfinite(result.logx.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite training/validation targets")
    return result


@dataclass
class BaselineConfig:
    radius: int = 2
    fingerprint_bits: int = 2048
    include_chirality: bool = True
    num_threads: int = 12
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100


def load_config(path: str | None) -> BaselineConfig:
    if not path:
        return BaselineConfig()
    import yaml
    with open(path) as handle:
        payload = yaml.safe_load(handle)
    return BaselineConfig(**payload)


class MolecularFeatures:
    """Sparse Morgan bits plus train-imputed RDKit descriptors and temperature.

    Molecular features are deterministic and may be computed on query structures.
    Only descriptor medians depend on a population, and they are fitted on train.
    """
    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()
        self.descriptor_names = [name for name, _ in Descriptors._descList]
        self.medians: np.ndarray | None = None
        self._fingerprint_cache: dict[str, tuple[int, ...]] = {}
        self._descriptor_cache: dict[str, np.ndarray] = {}

    def _features(self, smiles: str) -> tuple[tuple[int, ...], np.ndarray]:
        if smiles not in self._fingerprint_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid structure passed through data audit: {smiles!r}")
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.config.radius, fpSize=self.config.fingerprint_bits,
                includeChirality=self.config.include_chirality)
            self._fingerprint_cache[smiles] = tuple(generator.GetFingerprint(mol).GetOnBits())
            functions = dict(Descriptors._descList)
            values = []
            for name in self.descriptor_names:
                try:
                    value = float(functions[name](mol))
                    values.append(value if np.isfinite(value) and abs(value) < 1e30 else np.nan)
                except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                    values.append(np.nan)
            self._descriptor_cache[smiles] = np.asarray(values, dtype=np.float32)
        return self._fingerprint_cache[smiles], self._descriptor_cache[smiles]

    def _raw(self, frame: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        descriptors = np.empty((len(frame), 2 * len(self.descriptor_names) + 1), dtype=np.float32)
        for row, (solute, solvent, temperature) in enumerate(
                frame[["solute_smiles", "solvent_smiles", "temperature_k"]].itertuples(index=False, name=None)):
            solute_bits, solute_desc = self._features(solute)
            solvent_bits, solvent_desc = self._features(solvent)
            rows.extend([row] * (len(solute_bits) + len(solvent_bits)))
            cols.extend(solute_bits)
            cols.extend(bit + self.config.fingerprint_bits for bit in solvent_bits)
            descriptors[row] = np.concatenate((solute_desc, solvent_desc, [temperature]))
        fingerprints = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(frame), 2 * self.config.fingerprint_bits), dtype=np.float32)
        return fingerprints, descriptors

    def fit(self, train: pd.DataFrame) -> "MolecularFeatures":
        _, descriptors = self._raw(train)
        # Handle all-missing columns explicitly to avoid nanmedian's empty-slice warning.
        self.medians = np.zeros(descriptors.shape[1], dtype=np.float32)
        for column in range(descriptors.shape[1]):
            valid = descriptors[np.isfinite(descriptors[:, column]), column]
            if len(valid):
                self.medians[column] = np.median(valid)
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if self.medians is None:
            raise RuntimeError("Feature imputation must be fitted on training data first")
        fingerprints, descriptors = self._raw(frame)
        bad_rows, bad_cols = np.where(~np.isfinite(descriptors))
        descriptors[bad_rows, bad_cols] = self.medians[bad_cols]
        return sparse.hstack((fingerprints, sparse.csr_matrix(descriptors)), format="csr", dtype=np.float32)


class TanimotoNearestNeighbor:
    """One measured training neighbor; exact solvent, structure, then temperature.

    Prefer exact candidate solvent identity. For an unseen solvent, use the most
    Tanimoto-similar training solvent (all ties retained), then maximize solute
    Tanimoto and minimize absolute temperature distance. Final ties use record ID.
    No target values participate in neighbor selection.
    """
    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()
        self.train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> "TanimotoNearestNeighbor":
        if train.empty or "logx" not in train:
            raise ValueError("Training measurements with labels are required")
        self.train = train[FEATURE_COLUMNS + ["logx"]].sort_values("record_id").reset_index(drop=True).copy()
        return self

    def predict(self, query: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        if self.train is None:
            raise RuntimeError("Nearest-neighbor model has not been fitted")
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.config.radius, fpSize=self.config.fingerprint_bits,
            includeChirality=self.config.include_chirality)
        solutes = sorted(self.train.solute_smiles.unique())
        solvents = sorted(self.train.solvent_smiles.unique())
        solute_lookup = {s: i for i, s in enumerate(solutes)}
        solvent_lookup = {s: i for i, s in enumerate(solvents)}
        solute_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in solutes]
        solvent_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in solvents]
        row_solute = self.train.solute_smiles.map(solute_lookup).to_numpy()
        row_solvent = self.train.solvent_smiles.map(solvent_lookup).to_numpy()
        temp = self.train.temperature_k.to_numpy(float)
        labels = self.train.logx.to_numpy(float)
        solute_cache: dict[str, np.ndarray] = {}
        solvent_cache: dict[str, tuple[np.ndarray, float]] = {}
        outputs, provenance = [], []
        for q in query[FEATURE_COLUMNS].itertuples(index=False):
            from .retrieval import canonical_smiles
            solute_smiles = canonical_smiles(q.solute_smiles)
            solvent_smiles = canonical_smiles(q.solvent_smiles)
            if solute_smiles not in solute_cache:
                fp = generator.GetFingerprint(Chem.MolFromSmiles(solute_smiles))
                solute_cache[solute_smiles] = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, solute_fps))
            if solvent_smiles not in solvent_cache:
                if solvent_smiles in solvent_lookup:
                    solvent_cache[solvent_smiles] = (
                        np.flatnonzero(row_solvent == solvent_lookup[solvent_smiles]), 1.0)
                else:
                    fp = generator.GetFingerprint(Chem.MolFromSmiles(solvent_smiles))
                    similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, solvent_fps))
                    best_solvents = np.flatnonzero(np.isclose(similarities, similarities.max(), atol=1e-12, rtol=0))
                    solvent_cache[solvent_smiles] = (np.flatnonzero(np.isin(row_solvent, best_solvents)), float(similarities.max()))
            candidates, solvent_similarity = solvent_cache[solvent_smiles]
            similarity = solute_cache[solute_smiles][row_solute[candidates]]
            maximum = similarity.max()
            candidates = candidates[np.isclose(similarity, maximum, atol=1e-12, rtol=0)]
            nearest = candidates[np.argmin(np.abs(temp[candidates] - q.temperature_k))]
            outputs.append(labels[nearest])
            provenance.append({"nearest_record_id": self.train.record_id.iloc[nearest],
                               "solute_tanimoto": float(maximum), "solvent_tanimoto": solvent_similarity,
                               "temperature_distance_k": float(abs(temp[nearest] - q.temperature_k))})
        return np.asarray(outputs), pd.DataFrame(provenance)


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "rdkit", "lightgbm", "joblib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def train_baseline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not 1 <= config.num_threads <= 12:
        raise ValueError("CPU baseline thread count must be between 1 and 12")
    output = Path(args.output)
    if (output / "metadata.json").exists() or (output / "model.joblib").exists():
        raise FileExistsError(f"Run already exists: {output}; use predict or a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "running", "method": args.method, "seed": args.seed,
                "config": asdict(config), "versions": versions(), "target": "log10_mole_fraction",
                "measurements": str(Path(args.data).resolve()), "splits": str(Path(args.splits).resolve()),
                "measurement_sha256": sha256_file(args.data), "split_sha256": sha256_file(args.splits),
                "code_sha256": sha256_file(__file__), "started_unix": time.time()}
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        manifest = load_manifest(args.splits)
        validate_group_isolation(args.data, manifest)
        if Path(args.splits).stem == "scaffold":
            validate_group_isolation(args.data, manifest, "scaffold_group")
        train = load_partition(args.data, args.splits, "train", labels=True)
        val = load_partition(args.data, args.splits, "val", labels=True)
        metadata.update(train_records=len(train), val_records=len(val),
                        train_solutes=int(train.solute_id.nunique()), val_solutes=int(val.solute_id.nunique()))
        if args.method == "nearest":
            model = TanimotoNearestNeighbor(config).fit(train)
            bundle = {"method": "nearest", "model": model}
            predictions, provenance = model.predict(val[FEATURE_COLUMNS])
        else:
            import lightgbm as lgb
            features = MolecularFeatures(config).fit(train)
            x_train, x_val = features.transform(train), features.transform(val)
            model = lgb.LGBMRegressor(
                objective="regression", n_estimators=config.n_estimators,
                learning_rate=config.learning_rate, num_leaves=config.num_leaves,
                min_child_samples=config.min_child_samples, subsample=config.subsample,
                subsample_freq=1, colsample_bytree=config.colsample_bytree,
                reg_lambda=config.reg_lambda, n_jobs=config.num_threads, random_state=args.seed,
                deterministic=True, force_col_wise=True, verbosity=-1)
            history: dict = {}
            model.fit(x_train, train.logx.to_numpy(), eval_set=[(x_val, val.logx.to_numpy())],
                      callbacks=[lgb.early_stopping(config.early_stopping_rounds),
                                 lgb.log_evaluation(100), lgb.record_evaluation(history)])
            predictions, provenance = model.predict(x_val), pd.DataFrame(index=range(len(val)))
            bundle = {"method": "lightgbm", "model": model, "features": features}
            model.booster_.save_model(str(output / "model.txt"))
            write_json(output / "training_history.json", history)
            metadata["best_iteration"] = int(model.best_iteration_)
            metadata["feature_count"] = int(x_train.shape[1])
        joblib.dump(bundle, output / "model.joblib", compress=3)
        result = val[["record_id", "split"]].copy()
        result["pred_logx"] = predictions
        result = pd.concat((result.reset_index(drop=True), provenance.reset_index(drop=True)), axis=1)
        result.to_parquet(output / "predictions_val.parquet", index=False)
        metadata.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                        finished_unix=time.time(), validation_rmse=float(np.sqrt(np.mean((predictions - val.logx.to_numpy()) ** 2))))
    except Exception as error:
        metadata.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                        elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def require_frozen_test(freeze_path: str | None, split_path: str | Path) -> None:
    """Require a preregistered test release referencing the exact split manifest."""
    if not freeze_path:
        raise ValueError("Test prediction requires --freeze-manifest; test labels remain sealed")
    payload = json.loads(Path(freeze_path).read_text())
    expected = sha256_file(split_path)
    allowed = payload.get("split_sha256")
    if expected not in ([allowed] if isinstance(allowed, str) else (allowed or [])):
        raise ValueError("Freeze manifest does not reference this exact split SHA256")
    if payload.get("status") != "FROZEN":
        raise ValueError("Freeze manifest status must be FROZEN")


def predict_baseline(args: argparse.Namespace) -> None:
    run = Path(args.run)
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Only a completed training run can be used for prediction")
    data_path, split_path = metadata["measurements"], metadata["splits"]
    if sha256_file(data_path) != metadata["measurement_sha256"] or sha256_file(split_path) != metadata["split_sha256"]:
        raise ValueError("Training data or split manifest changed since fitting")
    if args.partition == "test":
        require_frozen_test(args.freeze_manifest, split_path)
    query = load_partition(data_path, split_path, args.partition, labels=False)
    bundle = joblib.load(run / "model.joblib")
    started = time.perf_counter()
    if bundle["method"] == "nearest":
        values, provenance = bundle["model"].predict(query)
    else:
        values = bundle["model"].predict(bundle["features"].transform(query))
        provenance = pd.DataFrame(index=range(len(query)))
    result = query[["record_id", "split"]].copy()
    result["pred_logx"] = values
    result = pd.concat((result.reset_index(drop=True), provenance.reset_index(drop=True)), axis=1)
    destination = Path(args.output) if args.output else run / f"predictions_{args.partition}.parquet"
    result.to_parquet(destination, index=False)
    write_json(destination.with_suffix(".timing.json"), {"records": len(query), "elapsed_seconds": time.perf_counter() - started,
               "prediction_code_sha256": sha256_file(__file__),
               "partition": args.partition, "run": str(run), "model_sha256": sha256_file(run / "model.joblib")})
    print(f"Wrote {len(result)} label-free predictions to {destination}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--method", choices=["nearest", "lightgbm"], required=True)
    train.add_argument("--data", default="data/processed/measurements.parquet")
    train.add_argument("--splits", default="data/splits/identity.csv")
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--config")
    train.add_argument("--output", required=True)
    train.set_defaults(func=train_baseline)
    predict = commands.add_parser("predict")
    predict.add_argument("--run", required=True)
    predict.add_argument("--partition", choices=sorted(SPLITS), default="val")
    predict.add_argument("--freeze-manifest")
    predict.add_argument("--output")
    predict.set_defaults(func=predict_baseline)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    # Persist classes under their importable module name rather than __main__,
    # so joblib artifacts can also be loaded by the report/evaluation modules.
    from solvent_select.baselines import main as module_main
    module_main()
