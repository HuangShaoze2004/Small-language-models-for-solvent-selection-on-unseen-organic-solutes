# Reproduction guide

Run commands from the repository root. This guide separates software checks,
rebuilding the saved paper assets, reconstructing the fixed data versions, and
training new models. A successful figure rebuild does not imply that the complete
GPU training campaign has been independently repeated.

The authoritative experiment registry is
[`configs/study_registry_dualqa_v2.json`](../configs/study_registry_dualqa_v2.json).
The main protocol is `sourceqa/identity`; the repaired scaffold protocol is
`strictqa/scaffold`. Earlier `sourceqa/scaffold` runs are superseded diagnostics.

## 1. Isolated environments

The recorded environments use Python 3.11. The principal lock contains data,
classical-model and QLoRA dependencies. Chemprop has a separate lock because its
graph dependency uses a different RDKit version. Do not combine the two
environments or modify the system Python or GPU driver.

```bash
mkdir -p .tmp .cache
source scripts/env.sh
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .

uv venv --python .venv/bin/python .venv-chemprop
uv pip sync --python .venv-chemprop/bin/python requirements-chemprop.lock
uv pip install --python .venv-chemprop/bin/python --no-deps -e .
```

`scripts/env.sh` puts caches and temporary files inside the checkout and sets a
`src/` import path. It must be sourced from the repository root. QLoRA requires a
working CUDA GPU; its implementation does not silently replace this with CPU
training. Software tests and saved-result plotting do not require model weights.

## 2. Software checks without experimental data

The following tests use implementation fixtures. They cover chemical identity,
grouped splitting, measured candidate construction, near ties, retrieval
exclusions, label isolation and metric aggregation. Fixtures are not research
results.

```bash
PYTHON_BIN=.venv/bin/python bash scripts/test_unit.sh
```

Tests requiring Chemprop are skipped in an environment without that package.
Their small graph/temperature and checkpoint-resume checks can be run separately:

```bash
CUDA_VISIBLE_DEVICES='' .venv-chemprop/bin/python -m pytest -q tests/test_baselines.py
```

The full suite also contains real-input and historical execution audits. Their
requirements differ from the fixture tests; inspect skip reasons and missing
artifact errors instead of treating every absent experimental file as a software
failure. The repository's release verification record describes the checks
actually run for this code export.

## 3. Rebuild the saved figures and tables

The static paper assets are in [`results/paper_assets/`](../results/paper_assets/).
Their provenance manifest records 35 direct input files. The reconstruction also
checks the output files listed by three upstream final-report manifests.

The accompanying `reproducibility/figure_inputs.tar.gz` contains the fixed inputs
needed for this operation. It includes processed measurements, query views,
saved result summaries, descriptive slices and the three condition-retrieval
prediction files required to reconstruct the selected chemical cases. It does
not contain the base model, trained adapters, all 62 methods' raw predictions,
or every training checkpoint.

```bash
.venv/bin/python scripts/prepare_figure_inputs.py
.venv/bin/python scripts/build_chemical_papers_assets.py \
  --output .tmp/rebuilt_paper_assets
.venv/bin/python scripts/verify_publication_assets.py .tmp/rebuilt_paper_assets
```

The preparation helper validates the packaged input paths and checksums. It must
not replace a different existing scientific artifact. The plotting command reads
the saved computations and invokes no model. Use a fresh output directory for
each reconstruction. Compare its `asset_manifest.json` and generated files with
`results/paper_assets/`; output counts and values should be assessed together with
the recorded software environment. The original project actually reproduced all
29 figure/table outputs byte-for-byte in its restored environment.

## 4. Reconstruct the fixed scientific data versions

The endpoint is the archive's native decimal-log mole-fraction solubility. No
density-based conversion to mol/L is used. The download script fixes the public
BigSolDB 2.0 archive and verifies the source CSV checksum.

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/reproduce_paper_data.py
```

The reconstruction wrapper builds the initial data and the two source-curated
versions in an isolated stage, checks the saved scientific manifests, and installs
only missing data outputs. Existing different outputs must be inspected rather
than deleted or silently replaced. Its default stage is
an automatically created `.tmp/paper_data_rebuild_*` directory.

This operation applies the frozen source policies and their saved numerical
evidence. It does not download ThermoML or perform a new independent evaluation
of every experimental source. The source lists, threshold, original assignments,
and surviving held-out records remain fixed. The strict protocol's historical
source-curation exposure and subsequent repair remain disclosed in
[`METHODS.md`](METHODS.md).

The resulting namespaces are:

| Protocol | Measurements | Split | Selected query/target views |
|---|---|---|---|
| Main identity | `data/processed/sourceqa/measurements.parquet` | `data/splits/sourceqa/identity.csv` | `data/views/sourceqa/identity/` |
| Strict scaffold | `data/processed/strictqa/measurements_scaffold.parquet` | `data/splits/strictqa/scaffold.csv` | `data/views/strictqa/scaffold/` |

For reference, explicit view construction uses separate roots:

```bash
.venv/bin/python -m solvent_select.views --protocol identity \
  --processed-root data/processed/sourceqa \
  --targets-root data/targets/sourceqa --output-root data/views/sourceqa

.venv/bin/python -m solvent_select.views --protocol scaffold \
  --processed-root data/processed/strictqa \
  --targets-root data/targets/strictqa --output-root data/views/strictqa
```

The legacy `scripts/materialize_views.py` assumes the historical sourceqa roots;
it is not the common entry point for the two current protocols. Likewise, running
the low-level quarantine builders directly in a checkout containing frozen audit
files can trigger their overwrite protections. Use the isolated wrapper above.

Advanced source verification uses `scripts/download_thermoml_audit.py`,
`scripts/reproduce_source_audit.py` and
`scripts/audit_strict_training_sources.py`. These have additional public-archive
and original-data prerequisites, preserve original selection metadata, and refuse
some pre-existing destinations. They are distinct from applying the saved
curation policy and should be run in a separately prepared workspace.

## 5. Fit a new baseline

The following main-protocol example trains only on training records and writes
validation predictions. Choose an unused run identifier; completed experiments
are not overwritten.

```bash
.venv/bin/python scripts/run_experiment.py \
  --run-id reproduction_identity_lgbm_17 --runs-root artifacts/runs_sourceqa \
  --kind lightgbm --protocol identity --seed 17 -- \
  .venv/bin/python -m solvent_select.baselines train \
  --method lightgbm --seed 17 --config configs/baselines.yaml \
  --data data/processed/sourceqa/measurements.parquet \
  --splits data/splits/sourceqa/identity.csv \
  --output artifacts/runs_sourceqa/reproduction_identity_lgbm_17
```

`--method nearest` selects the deterministic Tanimoto baseline. Chemprop is fitted
from scratch in its separate environment:

```bash
.venv-chemprop/bin/python -m solvent_select.chemprop_runner train \
  --config configs/baselines_chemprop.yaml --seed 17 \
  --data data/processed/sourceqa/measurements.parquet \
  --splits data/splits/sourceqa/identity.csv \
  --output artifacts/runs_sourceqa/reproduction_identity_chemprop_17 \
  --purpose formal --accelerator gpu
```

For the strict protocol, replace both measurement and split paths with the
`strictqa/scaffold` entries in the table and use a new output under
`artifacts/runs_strictqa/`. Do not reuse the primary model, scaler or retrieval
bank. Learned methods use seeds 17, 42 and 2026; deterministic methods are not
mechanically repeated.

## 6. Train and validate a language model

First download the fixed public backbone. The model snapshot manifest is required
and all recorded file hashes are checked.

```bash
.venv/bin/python scripts/download_model.py
```

The formal configurations retain their original training budget. This example
creates a new mixed-retrieval SFT run:

```bash
.venv/bin/python scripts/run_experiment.py \
  --run-id reproduction_identity_rag_17 --runs-root artifacts/runs_sourceqa \
  --kind qlora_rag_sft --protocol identity --seed 17 -- \
  .venv/bin/python -m solvent_select.llm train \
  --config configs/llm_formal_rag.yaml --seed 17 \
  --manifest data/splits/sourceqa/identity.csv \
  --measurements data/processed/sourceqa/measurements.parquet \
  --train-queries data/views/sourceqa/identity/train_queries.jsonl \
  --train-targets data/views/sourceqa/identity/train_targets.jsonl \
  --val-queries data/views/sourceqa/identity/val_queries.jsonl \
  --val-targets data/views/sourceqa/identity/val_targets.jsonl \
  --output artifacts/runs_sourceqa/reproduction_identity_rag_17
```

For ordinary SFT, use `configs/llm_formal_sft.yaml`, omit `--measurements`, and
choose an unused output. The formal settings cap training at 625 optimizer
updates; the generic epoch ceiling does not override this cap. The selected
inference adapter is the best recorded validation-loss checkpoint, not
necessarily the last saved training state. Keep optimizer, scheduler and random
state when resuming; inspect actual processes and saved metadata before starting
another job. Run only one principal GPU job at a time.

Compare retrieval modes with the same selected adapter and the same evidence
budget. For example:

```bash
.venv/bin/python -m solvent_select.llm predict \
  --config configs/llm_formal_rag.yaml --seed 17 \
  --retrieval-mode condition \
  --manifest data/splits/sourceqa/identity.csv \
  --measurements data/processed/sourceqa/measurements.parquet \
  --queries data/views/sourceqa/identity/val_queries.jsonl \
  --adapter artifacts/runs_sourceqa/reproduction_identity_rag_17/adapter \
  --output artifacts/runs_sourceqa/reproduction_identity_rag_17_condition_val/predictions.jsonl

.venv/bin/python -m solvent_select.evaluate \
  --queries data/views/sourceqa/identity/val_queries.jsonl \
  --targets data/views/sourceqa/identity/val_targets.jsonl \
  --predictions artifacts/runs_sourceqa/reproduction_identity_rag_17_condition_val/predictions.jsonl \
  --output artifacts/runs_sourceqa/reproduction_identity_rag_17_condition_val/evaluation
```

Use `structure`, `text`, `no_temperature` or `no_solvent` where registered. The
direct evidence rule is exposed through the `evidence` command with the same
query, bank and retrieval arguments. Property predictions are evaluated with
`--property-predictions` and the saved property-prediction Parquet file.

## 7. Frozen testing and interpretation of reproduction

The published final analysis followed development, validation, an explicit
decision record, release checks and a frozen task graph. Test prediction and
grading require their exact release manifests; a renamed validation command is
not a valid test-release procedure. Do not alter prompts or hyperparameters in
response to final-test scores.

The original project completed 294 frozen tasks, 62 final method evaluations and
64 consistency comparisons. Independent numerical audits and restored report
rebuilds were performed. A second complete GPU retraining campaign was not
performed and is not claimed here. Saved summaries and figure inputs permit
their stated reconstruction scope; they are not a substitute for missing raw
outputs or weights.

The executable modules, registered configurations and tests are research code.
Historical queue and freeze files may identify checkpoints or run directories
that are not distributed with this code release. Fresh training uses fresh
outputs and produces new provenance; it must not be presented as the original
run merely because its configuration matches.
