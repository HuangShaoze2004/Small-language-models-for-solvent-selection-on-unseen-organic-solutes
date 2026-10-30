# Study methods and interpretation

This document describes the executed `dualqa_v2` study associated with
**Small language models for solvent selection on unseen organic solutes:
retrieval effects and decision reliability**. The configurations, code and saved
artifacts define the implementation. This description does not assert acceptance
of the manuscript or a new physical mechanism.

## Task and endpoint

The scope is organic small solutes in pure solvents at a recorded temperature.
Tasks compare two supplied solvents or rank a supplied block of measured
candidates. They do not propose solvent mixtures, syntheses or an unrestricted
solvent search.

The numerical target is `log10(x)`, where `x` is the archive's native solute mole
fraction. No conversion to mol/L or mixing of logarithm bases is used. Therefore,
these scores are not directly interchangeable with concentration-based benchmark
leaderboards. Solvent selection is relative to the supplied candidate block, not
all possible solvents.

Comparable measurements share the full solute identity, source DOI and exact
recorded temperature: the temperature tolerance is 0 K. Only measured candidate
combinations become questions. Large solvent sets are divided by label-blind
identity hashing into balanced blocks with at most eight candidates. Pair
questions use measured pairs from those blocks; ranking questions require at
least three candidates.

## Data identity, curation and missing conditions

The fixed BigSolDB 2.0 source is Zenodo record 15094979. The original CSV SHA256
is:

```text
879f55f39db977e295c705e304d011d89d6e10da1daaa3f069181108c1de38b0
```

The source has 103,944 rows. Initial processing retains 103,515 measurements.
Full canonical isomeric identities preserve stereochemistry, formal charges and
disconnected fragments. Salt forms are not neutralized and merged into neutral
measurement labels. Largest-organic-parent and scaffold representations serve
only as operational splitting and exclusion safeguards.

Within an exact full-solute/solvent/source/temperature identity, a measurement
span greater than 0.3 dex is quarantined; otherwise the median log value is
retained. The pipeline does not infer missing pH, polymorph, pressure or
equilibration information. Common source and temperature are insufficient to
establish that every other experimental condition was controlled.

Training-source comparisons with a fixed NIST ThermoML archive motivated
conservative whole-source quarantine. The main `sourceqa` policy removes 919
records from 18 training-exclusive sources. The independently defined strict
policy removes 796 records from 16 sources and 11 training identities. Numerical
labels are not substituted, and the source checks do not certify all retained
measurements as correct. The policy reflects unresolved cross-archive
transcription or condition uncertainty.

The fixed source evidence and policies are preserved in
`configs/source_quarantine_v1_reproduce.json`,
`configs/strict_quarantine_v1.json` and the corresponding audit directories.
Reapplying their fixed exclusions is distinct from performing a new independent
source audit.

## Protocols and leakage boundary

Solutes are grouped before instruction generation. All measurements,
temperatures, solvents and derived questions for a solute remain in the same
partition. The split also checks the operational parent group. Identity labels
and grouping keys have different purposes and are not interchangeable.

| Quantity | Main identity (`sourceqa`) | Strict scaffold (`strictqa`) |
|---|---:|---:|
| Curated measurements | 102,596 | 68,829 |
| Training solutes | 1,140 | 732 |
| Validation solutes | 145 | 93 |
| Final-test solutes | 144 | 144 |
| Training measurements | 81,365 | 51,931 |
| Validation measurements | 10,576 | 6,243 |
| Final-test measurements | 10,655 | 10,655 |

The two protocols share the same protected final-test identities and questions.
The strict protocol removes development structures with operational scaffolds
shared with those test identities and uses a separate curated training bank.
Consequently, it is a stricter training restriction on a common test cohort,
not a second independent external dataset. The smaller training bank and source
coverage also differ; performance differences cannot isolate scaffold novelty
causally.

The common final view has 4,546 questions: 3,089 comparisons involving 124
eligible solutes and 1,457 rankings involving 103 eligible solutes. Question
counts are not independent chemical sample sizes. The training view contains
10,000 instructions per protocol, with 7,500 comparisons and 2,500 rankings.
Validation views contain 5,133 main and 3,263 strict questions. View selection
uses metadata and fixed hashes rather than answer values; pair evaluation
questions are capped at 32 per solute and eligible rankings are retained.

Property predictors, retrievers and language-model supervision obey each
protocol's data boundary. Preprocessing is fitted on training data. Retrieval
does not contain held-out solute measurements. For training-query retrieval,
the entire query leakage group is excluded, preventing retrieval of its own
labels. Test targets, experimental test values and answer fields are excluded
from query serialization and retrieval scoring.

The original primary-training source audit also inspected some records belonging
to the original strict-validation partition. Those sourceqa scaffold results
were superseded. Strict curation was rebuilt using strict-training-exclusive
evidence, original strict validation/test bytes were restored, and all affected
strict models were retrained. This repair does not erase historical analyst
exposure. The erratum and its resolution are part of the study's limitations.

SoluBench is treated as test-only and is not used for fine-tuning. Overlapping
solutes and incompatible endpoint conventions prevent an independent full
leaderboard claim. Pretrained FASTSOLV weights are not assumed to be independent
of the study data. Contamination of a public foundation model's pretraining
cannot be completely ruled out.

## Property models and direct evidence

The strong fitted baselines are:

- Tanimoto nearest neighbors with exact-solvent preference, then structural
  solvent matching where necessary, solute similarity and nearest temperature.
  Neighbor selection does not use the query label.
- LightGBM using separate solute/solvent Morgan fingerprints, molecular
  descriptors and temperature, with training-fitted imputation.
- Chemprop v2, retrained from scratch on two molecular graph components plus
  temperature, with training-fitted scaling.

The property models use all available training measurements in their protocol.
The language model sees a fixed 10,000-question view. The systems share an
accessible data boundary but not equal counts of supervised examples.

A deterministic direct-evidence comparator uses the same retrieved cards as the
language model. It aggregates within-training-solute candidate-solvent
differences rather than comparing absolute solubilities of unrelated solutes.
Unsupported pairs use a fixed tie fallback and remain in the denominator.
Evidence support, JSON validity and decision accuracy are separate quantities.

## Retrieval and language-model adaptation

The backbone is `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Its native instruction template and
fixed revision are retained. The public model's own license governs its
separately obtained weights.

Evidence cards group training measurements by solute, DOI and temperature, with
at most eight solvents per card. An inference query receives at most four cards,
no more than one card per training solute, and a 1,536-token evidence budget.
Whole-card trimming preserves the budget. Evidence cards can contain measured
training values; query target values remain inaccessible.

The retrieval policies are a label-free BM25 text index, solute structural
similarity, and the fixed condition-aware score:

```text
score = 0.6 × solute_Tanimoto
      + 0.3 × exact_candidate_solvent_coverage
      + 0.1 × exp(-abs(query_temperature - card_temperature) / 10 K)
```

Retrieval fingerprints use radius 2 and 2,048 bits. The no-temperature and
no-solvent ablations remove the corresponding term without renormalizing the
remaining weights. The temperature scale is an operational heuristic, not an
estimated dissolution enthalpy. These ablations do not establish a causal
temperature mechanism.

Ordinary SFT trains without retrieval. Retrieval SFT cycles text, structural and
condition cards through deterministic within-solute assignments. It produces one
mixed-retrieval adapter per learning seed, subsequently evaluated with different
inference retrieval modes. The principal contrast compares condition and
structural retrieval with the identical seed-specific adapter.

QLoRA uses NF4/BF16, rank 16, alpha 32, dropout 0.05, learning rate `1e-4` and
sequence length 3,072. Effective batch size is 16. Ordinary SFT uses microbatch
4 with accumulation 4; mixed-retrieval SFT uses microbatch 1 with accumulation
16. The resource-based formal cap is 625 optimizer updates, with evaluation and
saving every 125 updates. Best-adapter selection uses teacher-forced loss on
384 fixed validation questions and patience 3. All 12 formal runs reached the
step cap, while the validation-selected adapter can precede the final step.

Learned methods use seeds 17, 42 and 2026; deterministic methods are evaluated
once. Greedy inference permits at most 256 new tokens, starts with batch size 4,
and permits recorded OOM-only reductions to 2 or 1. There is no stronger
external property tool or extra sampling hidden in the language-model method.
The original computations used one RTX 4090 and serialized principal GPU jobs.

## Decisions, uncertainty and consistency

Pair answers are A, B or TIE. The experimental tie threshold is
`abs(logx_A - logx_B) <= 0.1 + 1e-12`. A ranking top-one hit allows the same
0.1-dex tolerance below the measured block maximum. Selection regret is the
unrounded difference between that maximum and the chosen solvent's measured
value; a tolerance-correct hit can therefore still have positive regret.

Invalid outputs remain in the denominator: accuracy and top-one hit are zero,
rank correlation receives the registered negative penalty, and regret is the
measured block's full best-minus-worst range. A pair TIE uses a deterministic
canonical-solvent order when a selected solvent is required for regret.
Mathematically undefined valid rank correlations remain undefined. Coverage is
reported separately.

Metrics average questions within physical condition, then conditions within
solute, then eligible solutes. Learning seeds are averaged within each solute
before paired bootstrap resampling. Confidence intervals use 10,000 solute
resamples with seed 20261025. They are conditional on the measured cohort and
selected checkpoints, not complete experimental-noise or retraining uncertainty.

The prespecified joint criterion requires condition-minus-structure pair accuracy
to improve by at least 0.02 and ranking regret to decrease by at least 0.05 dex,
with beneficial 95% intervals excluding zero and beneficial directions for all
three seeds. The final analysis plan was fixed before final inference after
pilot and development work; this is not whole-study preregistration before any
data inspection.

The consistency subset uses 256 comparison and 256 ranking base questions.
Candidate-order reversal and equivalent isomeric SMILES produce 1,024
transformed questions per comparison. Chemical identities and candidate mappings
are checked before mapped-answer agreement is computed. Pair agreement,
complete-rank agreement, first-choice agreement and both-valid coverage are
distinct outcomes. Raw and canonicalized property/text controls distinguish
removable serialization effects. Consistency is not chemical correctness.

## Supported conclusion and practical limits

Neither protocol supports the prespecified joint-benefit criterion. In the main
cohort, condition retrieval has a supplementary best-solvent hit gain of about
3.55 percentage points over structural retrieval and a 0.0268-dex regret
reduction; the comparison-accuracy interval includes zero. These observations
do not replace the unsuccessful joint hypothesis. The dedicated Chemprop
baseline retains better point estimates on the three principal main metrics.
Equivalent SMILES continue to change many language-model choices despite valid
output schemas.

The study therefore provides a bounded empirical reliability assessment, not
evidence of general superiority, exact equivalence, a new solvation mechanism or
an established quality–cost advantage. Supplementary endpoints and overlapping
chemical slices are descriptive and not multiplicity-adjusted confirmation.

Resource reports distinguish records from questions, CPU from GPU, setup from
successful batches, persistence scope, allocated from reserved memory, and
shared prediction reuse from additional compute. Their ratios are not
controlled cross-device speedups or energy estimates. The original project
performed extensive independent arithmetic audits and restored report rebuilds;
it did not repeat the entire GPU training campaign from scratch in a second
environment.
