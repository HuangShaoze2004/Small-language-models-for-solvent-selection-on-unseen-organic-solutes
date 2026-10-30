"""Local QLoRA SFT and auditable, deterministic inference.

Example (the parent orchestration owns download and GPU scheduling)::

    python -m solvent_select.llm train --config configs/llm_pilot.yaml \
      --train-queries data/queries/identity/train.jsonl \
      --train-targets data/targets/identity/train.jsonl \
      --val-queries data/queries/identity/val.jsonl \
      --val-targets data/targets/identity/val.jsonl \
      --manifest data/splits/identity.csv --output artifacts/runs/pilot

Training uses the Transformers Trainer checkpoint contract: model/adapter,
optimizer, scheduler, RNG, global step and skipped-data progress are restored.
No target file is opened by inference. No GPU job is started on import.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import sys
import time
import traceback
from pathlib import Path

from solvent_select.retrieval import (
    EvidenceBundle, RetrievalIndex, direct_evidence_prediction, public_query, stable_hash,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SYSTEM = (
    "Select pure solvents for an organic solute at the stated temperature. "
    "The target is the highest equilibrium mole-fraction solubility, not mol/L. "
    "Any evidence contains measurements of OTHER training solutes; it is not a measurement "
    "of this query solute. Near equality means an absolute difference in log10(mole fraction) "
    "of at most 0.1. For a pair return only JSON {\"choice\":\"A\"}, "
    "{\"choice\":\"B\"}, or {\"choice\":\"TIE\"}. For ranking return only JSON "
    "{\"ranking\":[candidate IDs from most to least soluble]}, using every candidate once. "
    "Do not output explanations, invented measurements, or additional keys."
)


@dataclasses.dataclass
class LLMConfig:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    local_files_only: bool = True
    seed: int = 17
    retrieval_mode: str = "none"  # none, mixed, text, structure, condition, ablations
    max_sequence_length: int = 3072
    max_evidence_tokens: int = 1536
    max_new_tokens: int = 256
    inference_batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = .05
    num_train_epochs: float = 3.
    max_steps: int = -1
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    warmup_ratio: float = .03
    weight_decay: float = 0.
    early_stopping_patience: int = 3
    max_train_examples: int = 0
    max_val_examples: int = 384
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    cpu_threads: int = 4


def load_config(path: str | Path | None, **overrides) -> LLMConfig:
    values = {}
    if path:
        import yaml
        with open(path, encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
    known = {field.name for field in dataclasses.fields(LLMConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown LLM config fields: {sorted(unknown)}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    config = LLMConfig(**values)
    if config.retrieval_mode not in {"none", "mixed", "text", "structure", "condition", "no_temperature", "no_solvent"}:
        raise ValueError("Invalid retrieval mode")
    if config.save_steps % config.eval_steps != 0:
        raise ValueError("save_steps must be a multiple of eval_steps for best-checkpoint selection")
    if config.inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    return config


def _environment() -> None:
    for name, value in {
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "HF_HUB_CACHE": str(ROOT / ".cache/huggingface/hub"),
        "TMPDIR": str(ROOT / ".tmp"),
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(name, value)
    (ROOT / ".tmp").mkdir(exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate query_id in {path}")
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment() -> dict:
    import torch
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'rdkit', 'numpy'):
        packages[name] = importlib.metadata.version(name)
    return {'python': platform.python_version(), 'packages': packages,
            'torch_cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'locks': {name: file_sha256(ROOT / name) for name in ('requirements.lock',) if (ROOT / name).exists()},
            'environment': {name: os.environ.get(name) for name in
                            ('PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS')}}


def finish_training_segment(metadata, elapsed, initial_step=0):
    """Preserve segment costs on resume; Trainer's resumed rates are not cumulative."""
    segment = {'status': metadata['status'], 'pid': metadata['pid'], 'started': metadata['started'],
               'elapsed_seconds': elapsed, 'initial_step': initial_step,
               'final_step': metadata.get('global_step'),
               'peak_allocated_bytes': metadata.get('peak_allocated_bytes'),
               'peak_reserved_bytes': metadata.get('peak_reserved_bytes')}
    metadata.setdefault('segments', []).append(segment)
    metadata['total_elapsed_seconds'] = sum(s['elapsed_seconds'] for s in metadata['segments'])
    metadata['runtime_definition'] = 'Each segment includes model loading, training, evaluation and saving; '
    metadata['runtime_definition'] += 'data preparation is outside this timer and included by the experiment wrapper. '
    metadata['runtime_definition'] += 'Trainer speed/loss fields apply to the current invocation, not the complete resumed run.'


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def gpu_lock():
    """A project-wide nonblocking GPU lock; the inode is never replaced."""
    import fcntl
    path = ROOT / "artifacts/gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another project GPU job owns {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started": time.time(), "argv": sys.argv}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_partition(queries: list[dict], expected: str) -> None:
    if not queries:
        raise ValueError("Query partition is empty")
    bad = {q.get("split") for q in queries} - {expected}
    if bad:
        raise ValueError(f"Expected only {expected} queries; got {bad}")


def validate_query_manifest(queries: list[dict], manifest: str | Path) -> None:
    import pandas as pd
    split = pd.read_csv(manifest, dtype={"solute_id": str})
    if split["solute_id"].duplicated().any():
        raise ValueError("Split manifest has duplicate identities")
    lookup = dict(zip(split["solute_id"], split["split"]))
    for query in queries:
        if lookup.get(str(query["solute_id"])) != query.get("split"):
            raise ValueError("Query partition does not agree with the frozen split manifest")


def sample_queries(queries: list[dict], limit: int, seed: int) -> list[dict]:
    """Label-blind round-robin sampling by solute and deterministic hash."""
    from collections import defaultdict
    groups = defaultdict(list)
    for query in queries:
        groups[str(query["solute_id"])].append(query)
    solutes = sorted(groups, key=lambda s: stable_hash(f"{seed}:{s}"))
    for solute in solutes:
        groups[solute].sort(key=lambda q: stable_hash(f"{seed}:{q['query_id']}"))
    selected = []
    depth = 0
    while len(selected) < len(queries):
        for solute in solutes:
            if depth < len(groups[solute]):
                selected.append(groups[solute][depth])
                if limit and len(selected) >= limit:
                    return selected
        depth += 1
    return selected


def target_text(query: dict, target: dict) -> str:
    if query["query_id"] != target["query_id"]:
        raise ValueError("Query/target mismatch")
    if query["task"] == "pair":
        if target.get("choice") not in {"A", "B", "TIE"}:
            raise ValueError("Invalid pair training label")
        result = {"choice": target["choice"]}
    else:
        ranking = target.get("ranking")
        expected = {str(c["id"]) for c in query["candidates"]}
        if not isinstance(ranking, list) or len(ranking) != len(expected) or set(map(str, ranking)) != expected:
            raise ValueError("Invalid ranking training label")
        result = {"ranking": list(map(str, ranking))}
    return json.dumps(result, separators=(",", ":"))


def parse_prediction(raw: str, query: dict) -> tuple[dict | None, str | None]:
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        return None, f"invalid_json:{error}"
    if not isinstance(value, dict):
        return None, "output_not_object"
    if query["task"] == "pair":
        if (set(value) != {"choice"} or not isinstance(value["choice"], str)
                or value["choice"] not in {"A", "B", "TIE"}):
            return None, "invalid_choice"
        return value, None
    ranking = value.get("ranking")
    expected = {str(c["id"]) for c in query["candidates"]}
    if set(value) != {"ranking"} or not isinstance(ranking, list):
        return None, "invalid_ranking_schema"
    if not all(isinstance(item, str) for item in ranking):
        return None, "ranking_ids_not_strings"
    if len(ranking) != len(expected) or set(ranking) != expected:
        return None, "ranking_not_a_permutation"
    return value, None


def build_messages(query: dict, evidence: EvidenceBundle | None = None) -> list[dict]:
    content = {"query": public_query(query)}
    if evidence is not None:
        # Serialize payload only; source IDs and scores remain in the audit artifact.
        content["training_evidence"] = evidence.payload
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(content, separators=(",", ":"))}]


def encode_example(query: dict, target: dict, tokenizer, config: LLMConfig,
                   index: RetrievalIndex | None, mode: str) -> tuple[dict, dict]:
    evidence = None
    if mode != "none":
        if index is None:
            raise ValueError("Retrieval mode requires a training index")
        evidence = index.retrieve(query, mode, config.max_evidence_tokens,
                                  lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    answer = target_text(query, target)
    while True:
        messages = build_messages(query, evidence)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                             tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenizer chat template is not prefix-stable; cannot safely mask labels")
        if len(full_ids) <= config.max_sequence_length:
            break
        if evidence is None or not evidence.payload:
            raise ValueError(f"Query {query['query_id']} exceeds context budget without any evidence")
        # Input-budget reconciliation also removes WHOLE cards, uniformly for all methods.
        evidence.payload.pop()
        evidence.audit.pop()
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if not any(value != -100 for value in labels):
        raise ValueError("The assistant target contains no supervised token")
    _refresh_evidence_coverage(evidence, query, tokenizer)
    audit = {"query_id": query["query_id"], "retrieval_mode": mode,
             "input_tokens": len(prompt_ids), "target_tokens": len(full_ids) - len(prompt_ids),
             "evidence": evidence.audit if evidence else [],
             "evidence_tokens": evidence.token_count if evidence else 0,
             "candidate_evidence_coverage": evidence.candidate_coverage if evidence else {}}
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}, audit


class EncodedDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        return self.rows[item]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad = int(pad_token_id)

    def __call__(self, examples: list[dict]):
        import torch
        maximum = max(len(row["input_ids"]) for row in examples)
        return {key: torch.tensor([
            row[key] + [(-100 if key == "labels" else self.pad if key == "input_ids" else 0)] *
            (maximum - len(row[key])) for row in examples], dtype=torch.long)
                for key in ("input_ids", "attention_mask", "labels")}


def make_dataset(queries: list[dict], targets: dict[str, dict], tokenizer,
                 config: LLMConfig, index: RetrievalIndex | None, audit_path: Path):
    from collections import Counter
    rows = []
    per_solute_count = Counter()
    with audit_path.open("w", encoding="utf-8") as audit_handle:
        for i, query in enumerate(queries):
            # Cycle WITHIN each solute, not simply over the globally interleaved
            # list (which could assign one retriever to every example of a solute).
            solute = str(query["solute_id"])
            offset = int(stable_hash(solute)[:8], 16) % 3
            mode = (("text", "structure", "condition")[(per_solute_count[solute] + offset) % 3]
                    if config.retrieval_mode == "mixed" else config.retrieval_mode)
            per_solute_count[solute] += 1
            row, audit = encode_example(query, targets[query["query_id"]], tokenizer, config, index, mode)
            rows.append(row)
            audit_handle.write(json.dumps(audit) + "\n")
            if (i + 1) % 1000 == 0:
                print(json.dumps({"stage": "tokenize", "examples": i + 1, "total": len(queries)}), flush=True)
    return EncodedDataset(rows)


def latest_complete_checkpoint(output: str | Path) -> str | None:
    checkpoints = []
    for path in Path(output).glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if checkpoint_is_complete(path):
            checkpoints.append((step, path))
    return str(max(checkpoints, key=lambda item: item[0])[1]) if checkpoints else None


def checkpoint_is_complete(path: str | Path) -> bool:
    path = Path(path)
    if not (path / "COMPLETE.json").is_file():
        return False
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
            return False
    return any(file.is_file() and file.stat().st_size > 0 for file in path.glob("adapter_model.*"))


def _adapter_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    directory = Path(path)
    files = sorted(p for p in directory.glob("*") if p.is_file() and
                   (p.name == "adapter_config.json" or p.name.startswith("adapter_model.")))
    if not any(p.name.startswith("adapter_model.") for p in files):
        raise ValueError(f"No adapter weights found in {directory}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file_sha256(file).encode())
    return digest.hexdigest()


def prediction_fingerprint(args, config: LLMConfig) -> tuple[dict, str]:
    fingerprint = {"prediction_kind": getattr(args, "prediction_kind", "llm"),
                   "config": dataclasses.asdict(config), "query_sha256": file_sha256(args.queries),
                   "adapter_sha256": _adapter_hash(args.adapter),
                   "manifest_sha256": file_sha256(args.manifest) if args.manifest else None,
                   "measurements_sha256": file_sha256(args.measurements) if args.measurements else None,
                   "implementation_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                              for name in ("llm.py", "retrieval.py")},
                   "limit": args.limit}
    return fingerprint, stable_hash(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")))


def validate_freeze(path: str | Path, fingerprint_hash: str) -> None:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise ValueError("Protocol freeze manifest is not marked FROZEN")
    protocol = Path(freeze["protocol_path"])
    if not protocol.is_absolute():
        protocol = ROOT / protocol
    if file_sha256(protocol) != freeze.get("protocol_sha256"):
        raise ValueError("Protocol document differs from its frozen hash")
    if fingerprint_hash not in freeze.get("prediction_fingerprints", []):
        raise ValueError("This exact prediction configuration is absent from the frozen manifest")


def _refresh_evidence_coverage(evidence, query, tokenizer):
    from .retrieval import canonical_smiles
    if evidence is None:
        return
    evidence.candidate_coverage = {
        str(c["id"]): sum(any(m["solvent_smiles"] == canonical_smiles(c["smiles"]) for m in card["measurements"])
                           for card in evidence.payload) for c in query["candidates"]}
    evidence.token_count = (len(tokenizer.encode(json.dumps(evidence.payload, separators=(",", ":")),
                                                 add_special_tokens=False)) if evidence.payload else 0)


def _tokenizer(config: LLMConfig):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision,
                                              local_files_only=config.local_files_only, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _base_model(config: LLMConfig):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU; no CPU training fallback is silently selected")
    torch.set_num_threads(config.cpu_threads)
    return AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.revision, local_files_only=config.local_files_only,
        trust_remote_code=False, device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16),
        attn_implementation="sdpa",
    )


def _index(args, config):
    if config.retrieval_mode == "none":
        return None
    if not args.measurements or not args.manifest:
        raise ValueError("Retrieval requires --measurements and --manifest")
    return RetrievalIndex.from_files(args.measurements, args.manifest)


def train(args, config: LLMConfig) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists() and not args.resume:
        raise FileExistsError("Existing run found; use --resume or a new output directory")
    train_queries = read_jsonl(args.train_queries)
    val_queries = read_jsonl(args.val_queries)
    assert_partition(train_queries, "train")
    assert_partition(val_queries, "val")
    if not args.manifest:
        raise ValueError("All training runs require an explicit --manifest, including plain SFT")
    validate_query_manifest(train_queries, args.manifest)
    validate_query_manifest(val_queries, args.manifest)
    if {q["leakage_group"] for q in train_queries} & {q["leakage_group"] for q in val_queries}:
        raise ValueError("Training and validation query leakage groups intersect")
    train_targets = {row["query_id"]: row for row in read_jsonl(args.train_targets)}
    val_targets = {row["query_id"]: row for row in read_jsonl(args.val_targets)}
    if {q["query_id"] for q in train_queries} != set(train_targets):
        raise ValueError("Training target IDs must exactly match the training queries")
    if {q["query_id"] for q in val_queries} != set(val_targets):
        raise ValueError("Validation target IDs must exactly match the validation queries")
    # Use a fixed sampling seed independent of learning seeds: identical examples and retrieval assignments.
    train_queries = sample_queries(train_queries, config.max_train_examples, 20261025)
    val_queries = sample_queries(val_queries, config.max_val_examples, 20261025)
    input_paths = [args.train_queries, args.train_targets, args.val_queries, args.val_targets]
    if args.manifest:
        input_paths.append(args.manifest)
    if args.measurements and config.retrieval_mode != "none":
        input_paths.append(args.measurements)
    metadata = {"status": "preparing", "pid": os.getpid(), "started": time.time(),
                "environment": runtime_environment(),
                "config": dataclasses.asdict(config), "argv": sys.argv,
                "inputs": {str(path): file_sha256(path) for path in input_paths},
                "implementation_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                           for name in ("llm.py", "retrieval.py")},
                "train_examples": len(train_queries), "val_examples": len(val_queries),
                "sampling_seed": 20261025}
    if args.resume:
        previous = json.loads((output / "run.json").read_text())
        if (previous["config"] != metadata["config"] or previous["inputs"] != metadata["inputs"] or
                previous.get("implementation_sha256") != metadata["implementation_sha256"]):
            raise ValueError("Resume config or input hashes differ from the original run")
        metadata["resumed_from"] = args.resume
        segment_number = len(previous.get('segments', [])) + 1
        write_json(output / 'segments' / f'prior_run_{segment_number:03d}.json', previous)
        metadata['segments'] = previous.get('segments', [])
    write_json(output / "run.json", metadata)
    set_seed(config.seed)
    tokenizer = _tokenizer(config)
    index = _index(args, config)
    train_data = make_dataset(train_queries, train_targets, tokenizer, config, index, output / "train_evidence.jsonl")
    val_data = make_dataset(val_queries, val_targets, tokenizer, config, index, output / "val_evidence.jsonl")
    stop_requested = {"signal": None}

    class SafeStopCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if stop_requested["signal"] is not None:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(directory / "COMPLETE.json", {"step": state.global_step, "epoch": state.epoch,
                                                      "saved": time.time(), "signal": stop_requested["signal"]})

    old_signals = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_signals[signum] = signal.signal(signum, lambda received, frame: stop_requested.update(signal=received))
    started = time.monotonic()
    try:
        with gpu_lock():
            torch.cuda.reset_peak_memory_stats()
            model = _base_model(config)
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing,
                                                    gradient_checkpointing_kwargs={"use_reentrant": False})
            model = get_peft_model(model, LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
            model.print_trainable_parameters()
            training_args = TrainingArguments(
                output_dir=str(output), seed=config.seed, data_seed=config.seed,
                per_device_train_batch_size=config.micro_batch_size,
                per_device_eval_batch_size=config.micro_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
                max_steps=config.max_steps, eval_strategy="steps", eval_steps=config.eval_steps,
                save_strategy="steps", save_steps=config.save_steps, logging_steps=config.logging_steps,
                warmup_ratio=config.warmup_ratio, weight_decay=config.weight_decay,
                lr_scheduler_type="cosine", optim="adamw_torch", bf16=True,
                gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                save_total_limit=config.save_total_limit, save_only_model=False,
                report_to="none", dataloader_num_workers=0, remove_unused_columns=False,
                prediction_loss_only=True, label_names=["labels"], ignore_data_skip=False,
                restore_callback_states_from_checkpoint=True,
            )
            trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
                              data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                              callbacks=[SafeStopCallback(), EarlyStoppingCallback(
                                  early_stopping_patience=config.early_stopping_patience)])
            checkpoint = None
            initial_step = 0
            if args.resume:
                checkpoint = latest_complete_checkpoint(output) if args.resume == "latest" else args.resume
                if not checkpoint or not checkpoint_is_complete(checkpoint):
                    raise ValueError("Resume requires a complete checkpoint marked by COMPLETE.json")
                initial_step = json.loads((Path(checkpoint) / 'trainer_state.json').read_text())['global_step']
            metadata["status"] = "training"
            write_json(output / "run.json", metadata)
            result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model(str(output / "adapter"))
            tokenizer.save_pretrained(output / "adapter")
            trainer.save_state()
            metadata.update(status="interrupted" if stop_requested["signal"] else "completed",
                            metrics=result.metrics, best_checkpoint=trainer.state.best_model_checkpoint,
                            best_metric=trainer.state.best_metric, global_step=trainer.state.global_step,
                            epoch=trainer.state.epoch,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            elapsed_seconds=time.monotonic() - started,
                            latest_checkpoint=latest_complete_checkpoint(output),
                            stopped_by_signal=stop_requested["signal"])
            finish_training_segment(metadata, metadata['elapsed_seconds'], initial_step)
            write_json(output / "run.json", metadata)
            print(json.dumps(metadata), flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                        elapsed_seconds=time.monotonic() - started,
                        latest_checkpoint=latest_complete_checkpoint(output))
        finish_training_segment(metadata, metadata['elapsed_seconds'], locals().get('initial_step', 0))
        write_json(output / "run.json", metadata)
        raise
    finally:
        for signum, handler in old_signals.items():
            signal.signal(signum, handler)


def _inference_input(query, tokenizer, config, index, mode):
    evidence = None if mode == "none" else index.retrieve(
        query, mode, config.max_evidence_tokens,
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    while True:
        prompt = tokenizer.apply_chat_template(build_messages(query, evidence), tokenize=False,
                                               add_generation_prompt=True)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + config.max_new_tokens <= config.max_sequence_length:
            _refresh_evidence_coverage(evidence, query, tokenizer)
            return prompt, evidence
        if evidence is None or not evidence.payload:
            raise ValueError(f"Query {query['query_id']} exceeds inference context budget")
        evidence.payload.pop()
        evidence.audit.pop()


def trim_generated_tokens(tokens, eos_token_id) -> list[int]:
    """Remove batch-completion padding after the first generated EOS, retaining EOS."""
    values = list(map(int, tokens))
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id])
    for i, token in enumerate(values):
        if token in eos:
            return values[:i + 1]
    return values


def generate_batch(model, tokenizer, prompts: list[str], config: LLMConfig,
                   synchronize=None, inference_context=None) -> dict:
    """Greedy batched decoding; injectable CPU hooks support mock contract tests.

    Decoder-only ``generate`` retains the entire padded input width in every
    output row. Slicing by each unpadded prompt length would leak prompt tokens
    into predictions, so the batch width is the only extraction boundary.
    """
    if not prompts:
        raise ValueError("Cannot generate an empty batch")
    if synchronize is None or inference_context is None:
        import torch
        synchronize = synchronize or torch.cuda.synchronize
        inference_context = inference_context or torch.inference_mode
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    padded_width = int(inputs["input_ids"].shape[1])
    prompt_lengths = list(map(int, inputs["attention_mask"].sum(-1).tolist()))
    synchronize()
    model_started = time.monotonic()
    with inference_context():
        generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                   max_new_tokens=config.max_new_tokens,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
    synchronize()
    model_seconds = time.monotonic() - model_started
    outputs = []
    for row, prompt_length in zip(generated.tolist(), prompt_lengths):
        output_tokens = trim_generated_tokens(row[padded_width:], tokenizer.eos_token_id)
        outputs.append({"raw_output": tokenizer.decode(output_tokens, skip_special_tokens=True),
                        "input_tokens": prompt_length, "output_tokens": len(output_tokens)})
    if len(outputs) != len(prompts):
        raise ValueError("Model returned a different number of sequences than input prompts")
    return {"outputs": outputs, "model_seconds": model_seconds,
            "padded_input_width": padded_width}


def adaptive_batches(items: list, initial_batch_size: int, run_batch, is_oom,
                     clear_oom, on_oom=None):
    """Yield successful batches, lowering concurrency after recorded OOMs only.

    OOM retries never alter inputs, decoding settings, or sampling count. Failed
    attempt time is included in the next successful batch's wall-time cost.
    """
    if initial_batch_size < 1:
        raise ValueError("Batch size must be positive")
    position, batch_size = 0, initial_batch_size
    while position < len(items):
        group_started = time.monotonic()
        failures = 0
        while True:
            batch = items[position:position + batch_size]
            attempt_started = time.monotonic()
            try:
                result = run_batch(batch)
                attempt_seconds = time.monotonic() - attempt_started
                break
            except Exception as error:
                if not is_oom(error):
                    raise
                failed_seconds = time.monotonic() - attempt_started
                next_size = 2 ** ((len(batch) - 1).bit_length() - 1) if len(batch) > 1 else None
                event = {"failed_batch_size": len(batch), "next_batch_size": next_size,
                         "failed_attempt_seconds": failed_seconds, "position": position,
                         "error": repr(error), "time": time.time()}
                if on_oom:
                    on_oom(event)
                if next_size is None:
                    raise
                # Remove traceback references to failed CUDA tensors before cache release.
                error.__traceback__ = None
                clear_oom()
                batch_size = next_size
                failures += 1
        yield batch, result, {"batch_size": len(batch), "active_batch_limit": batch_size,
                              "batch_wall_seconds": time.monotonic() - group_started,
                              "successful_attempt_seconds": attempt_seconds,
                              "preceding_oom_retries": failures}
        position += len(batch)


def predict(args, config: LLMConfig) -> None:
    import torch
    from peft import PeftModel
    from transformers import set_seed
    if config.retrieval_mode == "mixed":
        raise ValueError("Inference requires one explicit retrieval mode, not mixed")
    queries = read_jsonl(args.queries)
    if not args.manifest:
        raise ValueError("All prediction runs require an explicit --manifest")
    validate_query_manifest(queries, args.manifest)
    partitions = {q.get("split") for q in queries}
    if "test" in partitions and (not args.allow_test or not args.freeze_manifest):
        raise ValueError("Test inference requires --allow-test and a matching --freeze-manifest")
    if not partitions.issubset({"train", "val", "test"}):
        raise ValueError("Queries must carry a known split partition")
    queries = sample_queries(queries, args.limit or 0, 20261025)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint, fingerprint_hash = prediction_fingerprint(args, config)
    if "test" in partitions:
        validate_freeze(args.freeze_manifest, fingerprint_hash)
    metadata_path = output.with_suffix(output.suffix + ".run.json")
    completed = set()
    previous = {}
    if output.exists():
        if not args.resume or not metadata_path.exists():
            raise FileExistsError("Existing predictions; use --resume latest with the same config")
        previous = json.loads(metadata_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Prediction resume inputs/config differ")
        completed = {row["query_id"] for row in read_jsonl(output)}
    metadata = {"status": "preparing", "pid": os.getpid(), "fingerprint": fingerprint,
                "environment": runtime_environment(),
                "fingerprint_sha256": fingerprint_hash,
                "freeze_manifest_sha256": file_sha256(args.freeze_manifest) if args.freeze_manifest else None,
                "started": time.time(), "already_complete": len(completed),
                "configured_inference_batch_size": config.inference_batch_size,
                "final_inference_batch_size": min(config.inference_batch_size,
                                                   previous.get("final_inference_batch_size", config.inference_batch_size)),
                "oom_events": previous.get("oom_events", []),
                "batch_wall_seconds_total": previous.get("batch_wall_seconds_total", 0.),
                "batch_model_seconds_total": previous.get("batch_model_seconds_total", 0.),
                "latency_definition": "Batch wall includes preparation, generation, decoding, and preceding OOM retries; "
                                      "excludes output persistence. Amortized latency divides batch wall by batch size; "
                                      "it is not single-query response latency."}
    write_json(metadata_path, metadata)
    set_seed(config.seed)
    tokenizer = _tokenizer(config)
    index = _index(args, config)
    started = time.monotonic()
    with gpu_lock():
        torch.cuda.reset_peak_memory_stats()
        model = _base_model(config)
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        model.config.use_cache = True
        metadata["status"] = "predicting"
        write_json(metadata_path, metadata)

        def run_batch(batch):
            prepared = [_inference_input(q, tokenizer, config, index, config.retrieval_mode) for q in batch]
            generated = generate_batch(model, tokenizer, [p[0] for p in prepared], config)
            rows = []
            for query, (_, evidence), item in zip(batch, prepared, generated["outputs"]):
                prediction, error = parse_prediction(item["raw_output"], query)
                rows.append({"query_id": query["query_id"], "solute_id": query["solute_id"],
                             "task": query["task"], "split": query["split"],
                             "raw_output": item["raw_output"], "prediction": prediction, "parse_error": error,
                             "seed": config.seed, "retrieval_mode": config.retrieval_mode,
                             "evidence": evidence.audit if evidence else [],
                             "candidate_evidence_coverage": evidence.candidate_coverage if evidence else {},
                             "input_tokens": item["input_tokens"], "output_tokens": item["output_tokens"]})
            return {"rows": rows, "model_seconds": generated["model_seconds"],
                    "padded_input_width": generated["padded_input_width"]}

        def is_oom(error):
            return isinstance(error, torch.cuda.OutOfMemoryError) or (
                isinstance(error, RuntimeError) and "cuda out of memory" in str(error).lower())

        def clear_oom():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        def record_oom(event):
            metadata["oom_events"].append(event)
            if event["next_batch_size"] is not None:
                metadata["final_inference_batch_size"] = event["next_batch_size"]
            write_json(metadata_path, metadata)
            print(json.dumps({"event": "inference_oom", **event}), flush=True)

        try:
            with output.open("a", encoding="utf-8") as handle:
                pending = [q for q in queries if q["query_id"] not in completed]
                batches = adaptive_batches(pending, metadata["final_inference_batch_size"], run_batch,
                                           is_oom, clear_oom, record_oom)
                for batch, result, timing in batches:
                    size = timing["batch_size"]
                    batch_id = stable_hash(fingerprint_hash + "|" + "|".join(q["query_id"] for q in batch))[:24]
                    for row in result["rows"]:
                        row.update(batch_id=batch_id, batch_size=size,
                                   configured_batch_size=config.inference_batch_size,
                                   batch_wall_seconds=timing["batch_wall_seconds"],
                                   batch_model_latency_seconds=result["model_seconds"],
                                   amortized_latency_seconds=timing["batch_wall_seconds"] / size,
                                   amortized_model_latency_seconds=result["model_seconds"] / size,
                                   single_query_latency_seconds=timing["batch_wall_seconds"] if size == 1 else None,
                                   padded_input_width=result["padded_input_width"],
                                   preceding_oom_retries=timing["preceding_oom_retries"],
                                   latency_kind="amortized_batch_wall")
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed.add(row["query_id"])
                    handle.flush()
                    metadata["final_inference_batch_size"] = timing["active_batch_limit"]
                    metadata["batch_wall_seconds_total"] += timing["batch_wall_seconds"]
                    metadata["batch_model_seconds_total"] += result["model_seconds"]
                    metadata["completed"] = len(completed)
                    if len(completed) % 100 < size:
                        write_json(metadata_path, metadata)
                        print(json.dumps({"completed": len(completed), "total": len(queries)}), flush=True)
            metadata.update(status="completed", completed=len(completed), elapsed_seconds=time.monotonic() - started,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as error:
            metadata.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                            completed=len(completed), elapsed_seconds=time.monotonic() - started)
            raise
        finally:
            write_json(metadata_path, metadata)


def predict_evidence(args, config: LLMConfig) -> None:
    """CPU baseline using precisely the evidence budget and tokenizer of the LLM."""
    if config.retrieval_mode in {"none", "mixed"}:
        raise ValueError("Direct evidence prediction requires an explicit single retriever")
    if args.adapter:
        raise ValueError("Direct evidence prediction has no language-model adapter")
    queries = read_jsonl(args.queries)
    if not args.manifest:
        raise ValueError("Evidence prediction requires --manifest")
    validate_query_manifest(queries, args.manifest)
    fingerprint, fingerprint_hash = prediction_fingerprint(args, config)
    if any(q.get("split") == "test" for q in queries):
        if not args.allow_test or not args.freeze_manifest:
            raise ValueError("Test evidence prediction requires --allow-test and --freeze-manifest")
        validate_freeze(args.freeze_manifest, fingerprint_hash)
    queries = sample_queries(queries, args.limit or 0, 20261025)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(output.suffix + ".run.json")
    completed = set()
    if output.exists():
        if not args.resume or not metadata_path.exists():
            raise FileExistsError("Existing predictions; use --resume latest with the same config")
        previous = json.loads(metadata_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Prediction resume inputs/config differ")
        completed = {row["query_id"] for row in read_jsonl(output)}
    metadata = {"status": "preparing", "pid": os.getpid(), "fingerprint": fingerprint,
                "fingerprint_sha256": fingerprint_hash,
                "freeze_manifest_sha256": file_sha256(args.freeze_manifest) if args.freeze_manifest else None,
                "started": time.time(), "already_complete": len(completed)}
    write_json(metadata_path, metadata)
    tokenizer = _tokenizer(config)
    index = _index(args, config)
    started = time.monotonic()
    metadata["status"] = "predicting"
    write_json(metadata_path, metadata)
    with output.open("a", encoding="utf-8") as handle:
        for query in queries:
            if query["query_id"] in completed:
                continue
            start = time.monotonic()
            prompt, evidence = _inference_input(query, tokenizer, config, index, config.retrieval_mode)
            direct = direct_evidence_prediction(query, evidence)
            raw = json.dumps(direct["prediction"], separators=(",", ":"))
            prediction, error = parse_prediction(raw, query)
            row = {"query_id": query["query_id"], "solute_id": query["solute_id"],
                   "task": query["task"], "split": query["split"],
                   "raw_output": raw, "prediction": prediction, "parse_error": error,
                   "retrieval_mode": config.retrieval_mode, "method": "direct_evidence",
                   "evidence": evidence.audit, "candidate_evidence_coverage": evidence.candidate_coverage,
                   "pairwise_evidence": direct["pairwise_evidence"], "pair_coverage": direct["pair_coverage"],
                   "input_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                   "output_tokens": len(tokenizer.encode(raw, add_special_tokens=False)),
                   "latency_seconds": time.monotonic() - start}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(query["query_id"])
            if len(completed) % 1000 == 0:
                print(json.dumps({"completed": len(completed), "total": len(queries)}), flush=True)
    metadata.update(status="completed", completed=len(completed), elapsed_seconds=time.monotonic() - started)
    write_json(metadata_path, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "evidence", "fingerprint"])
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--measurements")
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--retrieval-mode")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--train-queries")
    parser.add_argument("--train-targets")
    parser.add_argument("--val-queries")
    parser.add_argument("--val-targets")
    parser.add_argument("--queries")
    parser.add_argument("--adapter")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--prediction-kind", choices=["llm", "evidence"], default="llm",
                        help="For fingerprint: which prediction command will consume the frozen fingerprint")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "evidence":
        args.prediction_kind = "evidence"
    elif args.command == "predict":
        args.prediction_kind = "llm"
    _environment()
    config = load_config(args.config, seed=args.seed, retrieval_mode=args.retrieval_mode,
                         model_id=args.model_id, revision=args.revision, max_steps=args.max_steps,
                         max_train_examples=args.max_train_examples, max_val_examples=args.max_val_examples,
                         inference_batch_size=args.inference_batch_size)
    if args.command != "fingerprint" and not args.output:
        parser.error("Training and prediction require --output")
    try:
        if args.command == "fingerprint":
            if not args.queries:
                parser.error("Fingerprint requires --queries")
            description, digest = prediction_fingerprint(args, config)
            print(json.dumps({"fingerprint": description, "sha256": digest}, indent=2))
        elif args.command == "train":
            if not all([args.train_queries, args.train_targets, args.val_queries, args.val_targets]):
                parser.error("Training requires explicit train and validation queries/targets")
            train(args, config)
        else:
            if not args.queries:
                parser.error("Prediction requires --queries")
            if args.command == "evidence":
                predict_evidence(args, config)
            else:
                predict(args, config)
    except BaseException as error:
        # Preparation and weight-loading failures must not leave a stale
        # 'preparing' status. Never overwrite a previous run that we did not own.
        if args.output:
            output = Path(args.output)
            metadata_path = (output / "run.json" if args.command == "train" else
                             output.with_suffix(output.suffix + ".run.json"))
            if metadata_path.is_file():
                record = json.loads(metadata_path.read_text())
                if record.get("pid") == os.getpid() and record.get("status") != "completed":
                    record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
                    write_json(metadata_path, record)
        raise


if __name__ == "__main__":
    main()
