# Upstream attribution and release scope

BigSolDB 2.0 is by Lev Krasnov, Dmitry Malikov, Marina Kiseleva, Sergei Tatarin, Sergey Sosnin and Stanislav Bezzubov. Fixed dataset: https://doi.org/10.5281/zenodo.15094979. Dataset article: https://doi.org/10.1038/s41597-025-05559-8. Its stated license is **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/.

Derived artifacts apply structure parsing/canonicalization while retaining full chemical identities, duplicate/conflict handling, training-source quarantine, grouped partitions and measured candidate questions. The endpoint stays native log10 mole fraction. These changes and the study's predictions do not imply upstream endorsement. Download provenance and checksums are preserved in `reproducibility/upstream_data_provenance.json`.

NIST ThermoML source material informed the recorded training-source checks. Its provenance and the limits of source quarantine are documented in the included source audits and `references/`. SoluBench was treated as test-only and was not used for fine-tuning; its upstream license is separately recorded. No official full-benchmark result is claimed.

The Qwen3-4B-Instruct-2507 backbone is Apache 2.0 at the pinned revision. Its license and weight hashes are in the model snapshot metadata; weights are not included. FASTSOLV is MIT-licensed upstream; Solvmate is GPL-2.0 upstream. They are related work and source-review references. Solvmate implementation was not copied into this package. License records are in `references/licenses/` and `references/sources.json`.

Imported scientific libraries retain their upstream licenses. This repository does not assign MIT, GPL or another new blanket license to the author's own research implementation; no such author choice was present in the source project. Publishing source and assigning a reuse license are separate decisions. The repository owner can provide an explicit code license later. No new legal or institutional eligibility determination is asserted here.
