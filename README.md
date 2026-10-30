# Small language models for solvent selection on unseen organic solutes

Research code for **“Small language models for solvent selection on unseen organic solutes: retrieval effects and decision reliability.”**

The study compares a local 4B language model, supervised fine-tuning, training-only retrieval, and dedicated property predictors for **pure-solvent comparisons and candidate rankings at a stated temperature**. The endpoint is the original **log10 mole-fraction solubility**, not mol/L. No new wet experiments were performed.

The recorded experiments are complete. **Neither protocol supports the prespecified joint improvement criterion.** Condition-aware retrieval gave limited ranking improvements on the main test cohort, while the retrained Chemprop baseline retained better point estimates on all three principal metrics. Equivalent SMILES and option order still changed many language-model decisions. The two protocols share the protected test solutes and are not independent external tests. This repository preserves those findings, including negative results.

## Quick start: CPU analysis and software checks

Use Python 3.11 and a project environment. Run commands from the repository root.

```bash
mkdir -p .tmp .cache
source scripts/env.sh
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements-analysis.lock
uv pip install --python .venv/bin/python --no-deps -e .
PYTHON_BIN=.venv/bin/python bash scripts/test_unit.sh
```

The software check uses synthetic fixtures for identity preservation, units, grouped splits, label isolation, retrieval, metrics and consistency. Fixture values are not paper results. The complete `tests/` directory also contains model- and artifact-dependent checks; it is not advertised as an unconditional clean-clone test command.

Actual release checks: **73 software tests passed**, **74 rebuilt data files matched their fixed hashes**, **29 paper assets rebuilt identically**, and **13 CLI help checks passed**. See [the machine-readable validation record](reproducibility/release_validation.json). These checks invoked no trained model.

For the executed QLoRA environment, use `requirements.lock`. Chemprop 2.3.1 requires the separate `requirements-chemprop.lock` environment because its RDKit version differs. See [reproduction instructions](docs/REPRODUCTION.md).

## Repository map

| Path | Purpose |
|---|---|
| `src/solvent_select/` | Data curation, grouping, retrieval, property models, QLoRA, scoring, statistical comparisons and release guards |
| `configs/study_registry_dualqa_v2.json` | Authoritative mapping of the primary and repaired strict protocols |
| `configs/llm_formal_{sft,rag}.yaml` | Executed 625-step configurations; seeds 17, 42 and 2026 |
| `scripts/` | Download, reproduction, experiment, audit and plotting entry points |
| `tests/` | Fixture tests and separately documented integration checks |
| `data/splits/` | Fixed solute-group assignments; measurements are acquired/rebuilt separately |
| `results/` | Recorded final metrics, contrasts and all paper figure/table assets |
| `reproducibility/` | Original code hashes, data checksums and optional frozen figure-input archive |
| `artifacts/audit/` | Selected immutable provenance and arithmetic evidence; not live execution state |
| `references/` | Verified upstream resources, bibliographic metadata and third-party license records |

Small historical configurations remain at their original paths because fixture tests and provenance depend on them. The active protocols are **sourceqa/identity** and **strictqa/scaffold**. Historical sourceqa/scaffold results are development diagnostics. The original scientific modules are retained byte-for-byte; organization adds documentation, dependency separation and safe reconstruction wrappers.

## Rebuild paper figures without a GPU

The optional local archive includes all 35 direct figure inputs and the additional outputs verified by the upstream report manifests. It contains derived data and three case-related prediction files, with checksums and attribution.

```bash
.venv/bin/python scripts/prepare_figure_inputs.py
.venv/bin/python scripts/build_chemical_papers_assets.py \
  --output .tmp/rebuilt_paper_assets
.venv/bin/python scripts/verify_publication_assets.py .tmp/rebuilt_paper_assets
```

The preparation helper checks every member before installing missing files and refuses to overwrite differing scientific inputs. Rebuilding uses saved measurements and predictions; it does not train or query a model. Six vector figures and sixteen LaTeX tables are supplied under `results/paper_assets/`. See [artifact scope and validation](docs/ARTIFACTS.md).

## Data, training and evaluation

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/reproduce_paper_data.py
```

The reconstruction helper stages the original data pipeline, replays the frozen source-curation policies and checks 74 scientific output hashes before installing missing files. It uses existing source-audit evidence; it does not claim to perform a new independent ThermoML audit. Full commands for LightGBM, Chemprop, QLoRA, validation prediction and evaluation are in [REPRODUCTION.md](docs/REPRODUCTION.md). The retrieval policy, tie rules, grouping and invalid-output penalties are in [METHODS.md](docs/METHODS.md).

Final-test prediction is guarded by an immutable release and completed validation evidence. The historical release cannot be reused for newly trained checkpoints. This code release does not contain all 62 methods’ raw predictions, adapters or optimizer checkpoints, and does not claim a second complete GPU training replication.

## Selected recorded results

Primary identity protocol, mean over the three fixed learning seeds within solute:

| Method | Pair accuracy (%) | Best-solvent hit (%) | Ranking regret (dex) |
|---|---:|---:|---:|
| LightGBM | 67.41 | 73.51 | 0.1429 |
| Chemprop | 70.13 | 75.40 | 0.1327 |
| Ordinary SFT, no retrieval | 68.08 | 71.70 | 0.1933 |
| Mixed-retrieval SFT, structure retrieval | 68.57 | 69.76 | 0.1924 |
| Mixed-retrieval SFT, condition retrieval | 69.40 | 73.30 | 0.1655 |

Condition minus structure on the **same seed-specific adapters** changed best-solvent hit rate by +3.55 percentage points (95% paired solute-bootstrap interval +0.98 to +6.49), but the pair-accuracy interval included zero and the joint practically meaningful criterion was not met. Supplementary intervals are not multiplicity-adjusted. Full results, support counts and limitations remain in `results/` and the method documentation; inference-time validity is not chemical correctness.

## Upstream material and citation

BigSolDB 2.0 is obtained from the fixed [Zenodo record 15094979](https://doi.org/10.5281/zenodo.15094979), under its stated CC BY 4.0 license. The backbone is Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, under Apache 2.0; weights are downloaded separately. Third-party software and data keep their own licenses. No blanket license for the author's code has been inferred; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The study manuscript is under preparation. No accepted-paper DOI, final author declaration or acceptance claim is supplied. Verified upstream citations are in `references/bibliography.bib`.
