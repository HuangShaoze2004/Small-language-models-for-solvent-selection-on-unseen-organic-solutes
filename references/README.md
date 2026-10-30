# Reference and resource audit

Checked on 2026-10-25. Bibliographic metadata is separated from factual claims
about an article's methods. A DOI resolving through Crossref verifies the
publisher-deposited citation, not the correctness of the paper or an independent
reproduction of its results.

- `bibliography.bib`: verified references used by the manuscript. Chemprop v1's
  online publication was in December 2023, but its cited journal issue is 2024;
  Chemprop v2 is cited as its 2026 journal issue. SoluBench now has a 2026 JCIM
  article and is not cited solely as an older preprint.
- `metadata/*.json`: bibliographic fields, registry response hashes, retrieval
  dates, and source URLs. These files do not contain full article text.
- `resource_checks.json`: direct checks of dataset/model metadata and repository
  license resources. A failure is recorded rather than inferred away.
- `sources.json`: claim-level source list and reuse decisions.
- `licenses/`: upstream license texts for the two surveyed code repositories;
  storing them does not imply copying their implementation into this project.

Refresh registry metadata with:

```bash
python references/fetch_metadata.py --refresh
```

The manuscript distinguishes native mole fractions from the author-derived
mol/L fields. The paper's density-based conversion is not applied to the main
endpoint. No benchmark canary or SoluBench evaluation question may enter SFT.

The unresolved school-specific CAS ranking policy belongs in the journal audit,
not in bibliographic metadata. JCR quartiles, third-party rankings and CAS
divisions must not be used interchangeably.
