# Artifact scope and checks

This is a code and numerical-evidence release. It excludes manuscript drafts and author placeholders, credentials, local environments, downloaded model weights, and the original job supervisor state. All source modules in `src/solvent_select` match the recorded project snapshot.

`reproducibility/source_snapshot.json` records the original project commit and per-file hashes. The GitHub packaging commit will differ because it adds documentation, a CPU dependency lock and reconstruction wrappers. Historical metadata can contain original machine paths or timestamps; those are retained provenance, not executable paths to recreate on another machine.

`results/performance.csv` has all 102 principal metric rows for 34 method groups; `results/consistency.csv` has all 720 consistency rows for 36 groups. The two condition-minus-structure comparison JSON files retain the registered failed joint decisions. Point estimates do not establish a statistically confirmed ordering between every pair of methods.

The optional `reproducibility/figure_inputs.tar.gz` contains 43 files: 34 non-code direct inputs plus additional upstream report outputs. The 35th direct input is the figure script already distributed in `scripts/`. Its manifest pins the archive, every member, and that script. It includes processed BigSolDB measurements, derived question views, three condition-retrieval prediction files and case evaluations, and all predefined slice summaries needed by the paper figures. It does not contain every method's raw predictions or any model weights. Data attribution is in `THIRD_PARTY_NOTICES.md` and `reproducibility/upstream_data_provenance.json`.

After preparing the archive, the figure builder verifies the original report outputs, reads the fixed cohorts and summaries and reproduces six figures plus sixteen tables. `scripts/verify_publication_assets.py` compares all 29 output files, including the caption manifest, with the recorded paper asset hashes. PDF date metadata is disabled in this figure builder; whole-manuscript PDF compilation is outside this code release.

The original independent final-score audit covers 62 evaluations and 281,852 question–method rows; the consistency audit covers 64 comparisons and 65,536 transformed rows. Those complete audits were actually executed and reproduced in the original project, as recorded in the included audit reports. Their presence here does not imply that every raw input required to repeat those complete audits is bundled. The release validation report records only checks actually executed in this clean checkout.

The data reconstruction wrapper checks fixed scientific bytes, including labels needed for retrospective reproduction. Newly conducting model selection on these published test labels would not constitute a fresh held-out evaluation. Keep training, retrieval, validation and final-test boundaries as documented.
