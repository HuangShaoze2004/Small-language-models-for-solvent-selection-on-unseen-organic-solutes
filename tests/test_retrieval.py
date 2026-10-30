"""Synthetic fixtures verify isolation and scoring, not research performance."""
import dataclasses
import json

import pytest

from solvent_select.llm import (
    CompletionCollator, LLMConfig, build_messages, encode_example, parse_prediction,
    public_query, sample_queries, target_text, latest_complete_checkpoint, file_sha256,
    validate_freeze, validate_query_manifest, generate_batch, adaptive_batches, trim_generated_tokens,
)
from solvent_select.retrieval import (
    Card, EvidenceBundle, Measurement, RetrievalIndex, direct_evidence_prediction,
)


def query(**updates):
    result = {
        "query_id": "query-secret", "solute_id": "test-solute", "solute_smiles": "CCCO",
        "leakage_group": "test-family", "temperature_k": 298.15, "task": "pair", "split": "test",
        "candidates": [{"id": "water", "smiles": "O", "record_id": "target-row-secret-1"},
                       {"id": "ethanol", "smiles": "CCO", "record_id": "target-row-secret-2"}],
        "logx": -987.654, "choice": "ANSWER_SECRET", "source_doi": "TARGET_DOI_SECRET",
    }
    result.update(updates)
    return result


def card(identifier="a", smiles="CCC", group="train-family", temperature=298.15,
         values=(-2., -1.)):
    return Card(identifier, f"train-{identifier}", smiles, group, temperature, "training-doi", (
        Measurement(f"row-{identifier}-1", "water", "O", values[0]),
        Measurement(f"row-{identifier}-2", "ethanol", "CCO", values[1]),
    ))


def index(cards):
    return RetrievalIndex(cards, {c.solute_id for c in cards})


def test_query_whitelist_strips_all_target_metadata():
    text = json.dumps(build_messages(query()))
    for forbidden in ("query-secret", "target-row-secret", "987.654", "ANSWER_SECRET", "TARGET_DOI_SECRET"):
        assert forbidden not in text
    assert "CCCO" in text and "298.15" in text
    assert set(public_query(query())) == {"task", "solute_smiles", "temperature_K", "candidates"}


def test_retrieval_excludes_entire_query_family():
    same_family = card("near", "CCCO", "test-family")
    own_identity = dataclasses.replace(card("self", "CCCO", "test-family"), solute_id="test-solute")
    unrelated = card("far", "c1ccccc1", "other-family")
    retrieval = index([same_family, own_identity, unrelated])
    for mode in ("text", "structure", "condition", "no_temperature", "no_solvent"):
        evidence = retrieval.retrieve(query(), mode)
        assert [item["solute_id"] for item in evidence.audit] == ["train-far"]
        assert evidence.audit[0]["record_ids"] == ["row-far-1", "row-far-2"] or set(
            evidence.audit[0]["record_ids"]) == {"row-far-1", "row-far-2"}


def test_training_label_values_do_not_affect_retrieval_selection():
    cards = [card("a", "CCO", "family-a"), card("b", "CCCC", "family-b", 320.)]
    changed = [dataclasses.replace(c, measurements=tuple(dataclasses.replace(m, logx=-200 - i)
                                                        for i, m in enumerate(c.measurements))) for c in cards]
    first, second = index(cards), index(changed)
    for mode in ("text", "structure", "condition"):
        assert first._scores(query(), mode).tolist() == second._scores(query(), mode).tolist()


def test_temperature_changes_condition_score_but_not_structure_score():
    retrieval = index([card("a", "CCC", "family-a", 350.), card("b", "CCC", "family-b", 298.15)])
    assert retrieval._scores(query(), "structure")[0] == retrieval._scores(query(), "structure")[1]
    assert retrieval._scores(query(), "condition")[1] > retrieval._scores(query(), "condition")[0]
    assert retrieval._scores(query(), "no_temperature")[0] == retrieval._scores(query(), "no_temperature")[1]


def test_single_card_per_training_solute_and_whole_card_budget():
    first = card("a")
    duplicate_solute = dataclasses.replace(card("b", temperature=310.), solute_id=first.solute_id)
    retrieval = index([first, duplicate_solute, card("c", group="other")])
    evidence = retrieval.retrieve(query(), token_counter=len, max_tokens=10000)
    assert len({item["solute_id"] for item in evidence.audit}) == len(evidence.audit) == 2
    no_room = retrieval.retrieve(query(), token_counter=len, max_tokens=1)
    assert no_room.payload == [] and no_room.audit == []
    assert all(value == 0 for value in no_room.candidate_coverage.values())


def test_index_rejects_nontraining_card():
    with pytest.raises(ValueError, match="Non-training"):
        RetrievalIndex([card()], train_solutes={"some-other-solute"})


def test_file_index_filters_test_labels_and_rejects_group_overlap(tmp_path, monkeypatch):
    import pandas as pd
    rows = []
    for compound, group, value in (("train-a", "train-group", -2.), ("test-solute", "test-family", -999.)):
        rows.append(dict(record_id=compound, solute_id=compound, solute_smiles="CCC",
                         solvent_id="water", solvent_smiles="O", temperature_k=298.15,
                         logx=value, source_doi="10.test", leakage_group=group, scaffold_group="s"))
    path, split = tmp_path / "data.parquet", tmp_path / "split.csv"
    pd.DataFrame(rows).to_parquet(path)
    pd.DataFrame({"solute_id": ["train-a", "test-solute"], "split": ["train", "test"]}).to_csv(split, index=False)
    calls = []
    read_parquet = pd.read_parquet
    def checked_read(*args, **kwargs):
        calls.append(kwargs)
        if "logx" in kwargs.get("columns", []):
            assert kwargs["filters"] == [("solute_id", "in", ["train-a"])]
        else:
            assert set(kwargs["columns"]) == {"solute_id", "leakage_group"}
        return read_parquet(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", checked_read)
    retrieval = RetrievalIndex.from_files(path, split)
    assert len(calls) == 2
    assert len(retrieval.cards) == 1
    assert retrieval.cards[0].measurements[0].logx == -2.
    rows[1]["leakage_group"] = "train-group"
    pd.DataFrame(rows).to_parquet(path)
    with pytest.raises(ValueError, match="crosses split"):
        RetrievalIndex.from_files(path, split)


def test_direct_evidence_uses_within_solute_differences():
    # Arbitrary lattice offsets differ by ten orders of magnitude; both cards favor ethanol.
    retrieval = index([card("a", values=(-2., -1.)), card("b", group="other", values=(-12., -11.))])
    evidence = retrieval.retrieve(query())
    prediction = direct_evidence_prediction(query(), evidence)
    assert prediction["prediction"] == {"choice": "B"}
    assert prediction["pairwise_evidence"][0]["delta"] == -1.
    empty = EvidenceBundle([], [], 0, "condition", {})
    assert direct_evidence_prediction(query(), empty)["prediction"] == {"choice": "TIE"}
    assert direct_evidence_prediction(query(), empty)["pair_coverage"] == 0.


def test_prediction_parser_does_not_repair_or_drop_bad_rankings():
    assert parse_prediction('{"choice":"TIE"}', query()) == ({"choice": "TIE"}, None)
    assert parse_prediction('{"choice":"A","reason":"invented"}', query())[0] is None
    ranking = query(task="rank")
    assert parse_prediction('{"ranking":["ethanol","water"]}', ranking)[1] is None
    assert parse_prediction('{"ranking":["water","water"]}', ranking)[0] is None
    assert parse_prediction('I think water', ranking)[0] is None


def test_sampling_is_label_blind_and_solute_balanced():
    rows = [query(query_id=f"{s}-{i}", solute_id=s) for s in ("a", "b", "c") for i in range(10)]
    selected = sample_queries(rows, 6, 17)
    assert {s: sum(q["solute_id"] == s for q in selected) for s in ("a", "b", "c")} == {"a": 2, "b": 2, "c": 2}
    changed = [dict(q, logx=123., choice="OTHER") for q in reversed(rows)]
    assert [q["query_id"] for q in selected] == [q["query_id"] for q in sample_queries(changed, 6, 17)]


class FakeTokenizer:
    """A transparent tokenizer checks masking without loading a model."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        result = "".join(f"<{m['role']}>" + m["content"] + "</end>" for m in messages)
        return result + ("<assistant>" if add_generation_prompt else "")


def test_answer_only_loss_mask_and_no_target_in_prompt():
    q = query()
    target = {"query_id": q["query_id"], "choice": "B"}
    encoded, audit = encode_example(q, target, FakeTokenizer(), LLMConfig(max_sequence_length=4000), None, "none")
    n = audit["input_tokens"]
    assert set(encoded["labels"][:n]) == {-100}
    supervised = "".join(chr(value) for value in encoded["labels"][n:])
    assert supervised == target_text(q, target) + "</end>"
    assert len(encoded["input_ids"]) == len(encoded["labels"])


def test_overlong_query_is_not_silently_truncated():
    q = query()
    with pytest.raises(ValueError, match="exceeds context"):
        encode_example(q, {"query_id": q["query_id"], "choice": "A"}, FakeTokenizer(),
                       LLMConfig(max_sequence_length=10), None, "none")


def test_collator_masks_padding():
    pytest.importorskip("torch")
    collator = CompletionCollator(0)
    batch = collator([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                      {"input_ids": [3], "attention_mask": [1], "labels": [3]}])
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_gpu_lock_is_project_wide_and_nonblocking(tmp_path, monkeypatch):
    import solvent_select.llm as llm
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    with llm.gpu_lock():
        assert (tmp_path / "artifacts/gpu.lock").exists()
        with pytest.raises(RuntimeError, match="Another project GPU job"):
            with llm.gpu_lock():
                pytest.fail("Second lock must not succeed")
    with llm.gpu_lock():
        pass


def test_resume_ignores_incomplete_newer_checkpoint(tmp_path):
    old, new = tmp_path / "checkpoint-50", tmp_path / "checkpoint-100"
    old.mkdir()
    new.mkdir()
    (new / "COMPLETE.json").write_text('{}')
    for name in ("COMPLETE.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                 "trainer_state.json", "adapter_model.safetensors"):
        (old / name).write_text("fixture")
    assert latest_complete_checkpoint(tmp_path) == str(old)


def test_test_freeze_requires_exact_protocol_and_configuration(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen protocol fixture")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"status": "FROZEN", "protocol_path": str(protocol),
                                  "protocol_sha256": file_sha256(protocol),
                                  "prediction_fingerprints": ["approved"]}))
    validate_freeze(freeze, "approved")
    with pytest.raises(ValueError, match="configuration"):
        validate_freeze(freeze, "unapproved")
    protocol.write_text("Changed protocol")
    with pytest.raises(ValueError, match="frozen hash"):
        validate_freeze(freeze, "approved")


def test_query_partition_cannot_override_manifest(tmp_path):
    import pandas as pd
    manifest = tmp_path / "split.csv"
    pd.DataFrame({"solute_id": ["test-solute"], "split": ["test"]}).to_csv(manifest, index=False)
    validate_query_manifest([query()], manifest)
    with pytest.raises(ValueError, match="frozen split manifest"):
        validate_query_manifest([query(split="train")], manifest)


def test_mock_batched_generation_left_padding_and_new_token_boundary():
    """Mock outputs verify batching mechanics, never model performance."""
    import contextlib
    import numpy as np

    class Batch(dict):
        def to(self, device):
            assert device == "cpu-mock"
            return self

    class BatchTokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 999

        def __call__(self, prompts, **kwargs):
            assert self.padding_side == "left"
            assert kwargs == {"padding": True, "return_tensors": "pt", "add_special_tokens": False}
            width = max(map(len, prompts))
            self.batch = Batch(input_ids=np.asarray([[0] * (width - len(text)) + list(map(ord, text))
                                                     for text in prompts]),
                               attention_mask=np.asarray([[0] * (width - len(text)) + [1] * len(text)
                                                          for text in prompts]))
            return self.batch

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens
            return "".join(chr(token) for token in tokens if token not in {0, 999})

    class MockModel:
        device = "cpu-mock"

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs == {"do_sample": False, "num_beams": 1, "max_new_tokens": 19,
                              "pad_token_id": 0, "eos_token_id": 999}
            assert input_ids.tolist()[0] == [0, 0, 0, 0, ord("x"), ord("y")]
            # Different completion lengths cause right padding after the first EOS.
            return np.concatenate([input_ids, np.asarray([[ord("A"), 999, 0],
                                                          [ord("B"), ord("C"), 999]])], axis=1)

    result = generate_batch(MockModel(), BatchTokenizer(), ["xy", "abcdef"],
                            LLMConfig(max_new_tokens=19), synchronize=lambda: None,
                            inference_context=contextlib.nullcontext)
    assert result["padded_input_width"] == 6
    assert result["outputs"] == [{"raw_output": "A", "input_tokens": 2, "output_tokens": 2},
                                  {"raw_output": "BC", "input_tokens": 6, "output_tokens": 3}]
    assert result["model_seconds"] >= 0


def test_adaptive_oom_batches_keep_every_query_and_fixed_generation_contract():
    class MockOOM(Exception):
        pass

    attempts, events, cleared = [], [], []
    def run(batch):
        attempts.append(list(batch))
        if len(batch) > 1:
            raise MockOOM("synthetic capacity failure")
        return list(batch)

    result = list(adaptive_batches(list(range(7)), 4, run, lambda error: isinstance(error, MockOOM),
                                   lambda: cleared.append(True), events.append))
    assert [len(batch) for batch in attempts[:3]] == [4, 2, 1]
    assert [event["next_batch_size"] for event in events] == [2, 1]
    assert len(cleared) == 2
    assert [item for batch, value, timing in result for item in value] == list(range(7))
    assert result[0][2]["preceding_oom_retries"] == 2
    assert all(timing["batch_wall_seconds"] >= timing["successful_attempt_seconds"]
               for batch, value, timing in result)


def test_adaptive_batches_preserve_partial_final_batch_and_propagate_other_errors():
    result = list(adaptive_batches(list(range(6)), 4, lambda batch: batch,
                                   lambda error: False, lambda: None))
    assert [timing["batch_size"] for _, _, timing in result] == [4, 2]
    def broken(batch):
        raise ValueError("unrelated model bug")
    with pytest.raises(ValueError, match="unrelated model bug"):
        list(adaptive_batches([1, 2], 4, broken, lambda error: False, lambda: None))


def test_single_query_oom_stops_instead_of_changing_tokens_or_sampling():
    attempts, events = [], []
    def broken(batch):
        attempts.append(batch)
        raise MemoryError("mock OOM")
    with pytest.raises(MemoryError):
        list(adaptive_batches([1], 4, broken, lambda error: True, lambda: None, events.append))
    assert attempts == [[1]]
    assert events[0]["next_batch_size"] is None


def test_generated_eos_trimming_accepts_multiple_stop_tokens():
    assert trim_generated_tokens([11, 22, 99, 0, 0], [98, 99]) == [11, 22, 99]
    assert trim_generated_tokens([11, 22], 99) == [11, 22]


def test_transformers_cpu_resume_restores_optimizer_rng_and_data_progress(tmp_path):
    """Exercise the real Trainer contract on a tiny random CPU-only model.

    This is a software recovery test with synthetic tokens, never a chemical
    experiment or evidence of Qwen training performance.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    from solvent_select.llm import EncodedDataset

    class StopAtTwo(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_save = True
                control.should_training_stop = True
            return control

    rows = [{"input_ids": [1, 2 + i % 10, 3, 4, 5, 6], "attention_mask": [1] * 6,
             "labels": [-100, -100, 3, 4, 5, 6]} for i in range(12)]
    data = EncodedDataset(rows)

    def make_trainer(path, callback=None):
        transformers.set_seed(17)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=16, n_embd=16, n_layer=1, n_head=2,
            resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1))
        args = transformers.TrainingArguments(
            output_dir=str(path), use_cpu=True, seed=17, data_seed=17,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            max_steps=4, save_steps=2, save_strategy="steps", logging_strategy="no",
            learning_rate=.001, report_to="none", disable_tqdm=True,
            dataloader_num_workers=0, remove_unused_columns=False, ignore_data_skip=False)
        return transformers.Trainer(model=model, args=args, train_dataset=data,
                                    data_collator=CompletionCollator(0), callbacks=([callback] if callback else []))

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        continuous = make_trainer(tmp_path / "continuous")
        continuous.train()
        expected = {name: value.detach().clone() for name, value in continuous.model.state_dict().items()}
        interrupted = make_trainer(tmp_path / "resumed", StopAtTwo())
        interrupted.train()
        assert interrupted.state.global_step == 2
        checkpoint = tmp_path / "resumed/checkpoint-2"
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            assert (checkpoint / name).exists()
        restored = make_trainer(tmp_path / "resumed")
        restored.train(resume_from_checkpoint=str(checkpoint))
        assert restored.state.global_step == continuous.state.global_step == 4
        for name, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(prior_threads)
