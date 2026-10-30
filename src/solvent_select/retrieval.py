"""Train-only, auditable retrieval of condition-specific measurement cards.

No target values are used to score candidates. Query and evidence serialization
are separate allowlists: training measurements are permitted evidence, whereas
query record identifiers, source papers, and target values are never serialized.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

MODES = ("text", "structure", "condition", "no_temperature", "no_solvent")


@lru_cache(maxsize=16384)
def canonical_smiles(smiles: str) -> str:
    """Chemical matching identity; preserve charge, fragments and stereochemistry."""
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES in chemical identity matching")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_query(query: dict) -> dict:
    """Return only the chemically meaningful, label-free task input."""
    task = query["task"]
    if task not in {"pair", "rank"}:
        raise ValueError(f"Unsupported task: {task}")
    candidates = query["candidates"]
    if (task == "pair" and len(candidates) != 2) or not 2 <= len(candidates) <= 8:
        raise ValueError("Pair requires two candidates; supported candidate count is 2..8")
    ids = [str(c["id"]) for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate identifiers must be unique")
    temperature = float(query["temperature_k"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive in kelvin")
    return {
        "task": task,
        "solute_smiles": str(query["solute_smiles"]),
        "temperature_K": temperature,
        "candidates": [
            {"option": chr(65 + i), "id": str(c["id"]), "smiles": str(c["smiles"])}
            for i, c in enumerate(candidates)
        ],
    }


@dataclass(frozen=True)
class Measurement:
    record_id: str
    solvent_id: str
    solvent_smiles: str
    logx: float


@dataclass(frozen=True)
class Card:
    card_id: str
    solute_id: str
    solute_smiles: str
    leakage_group: str
    temperature_k: float
    source_doi: str
    measurements: tuple[Measurement, ...]


@dataclass
class EvidenceBundle:
    payload: list[dict]
    audit: list[dict]
    token_count: int
    mode: str
    candidate_coverage: dict[str, int]

    def as_dict(self) -> dict:
        return asdict(self)


def _text_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text)


def _card_text(card: Card) -> str:
    # Explicitly exclude solubility, record IDs, DOI, and compound names.
    return " ".join([card.solute_smiles, f"{card.temperature_k:g}"] +
                    sorted({m.solvent_smiles for m in card.measurements}))


class RetrievalIndex:
    def __init__(self, cards: Iterable[Card], train_solutes: set[str],
                 max_cards: int = 4, max_solvents: int = 8):
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        self.cards = sorted(cards, key=lambda x: x.card_id)
        if not self.cards:
            raise ValueError("The training retrieval index is empty")
        self.train_solutes = set(map(str, train_solutes))
        self.identity_to_group = {c.solute_id: c.leakage_group for c in self.cards}
        if any(c.solute_id not in self.train_solutes for c in self.cards):
            raise ValueError("Non-training solute found in retrieval index")
        if len({c.card_id for c in self.cards}) != len(self.cards):
            raise ValueError("Duplicate retrieval card ID")
        self.max_cards, self.max_solvents = int(max_cards), int(max_solvents)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self._fingerprints = {}
        for smiles in sorted({c.solute_smiles for c in self.cards}):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid indexed solute SMILES: {smiles}")
            self._fingerprints[smiles] = self.generator.GetFingerprint(mol)
        self._unique_smiles = sorted(self._fingerprints)
        smiles_to_idx = {s: i for i, s in enumerate(self._unique_smiles)}
        self._card_solute_index = np.array([smiles_to_idx[c.solute_smiles] for c in self.cards])
        self._temperature = np.array([c.temperature_k for c in self.cards], dtype=float)
        self._solvent_sets = [{m.solvent_smiles for m in c.measurements} for c in self.cards]
        self._similarity_cache: dict[str, object] = {}
        self._postings = defaultdict(list)
        lengths = []
        for i, card in enumerate(self.cards):
            counts = Counter(_text_tokens(_card_text(card)))
            lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self._postings[token].append((i, count))
        self._lengths = np.asarray(lengths, dtype=float)
        self._average_length = float(self._lengths.mean())
        self._bm25_postings = {}
        for token, postings in self._postings.items():
            indices = np.asarray([p[0] for p in postings], dtype=np.int32)
            frequencies = np.asarray([p[1] for p in postings], dtype=float)
            idf = math.log(1 + (len(self.cards) - len(postings) + .5) / (len(postings) + .5))
            norm = frequencies + 1.5 * (.25 + .75 * self._lengths[indices] / self._average_length)
            self._bm25_postings[token] = (indices, idf * frequencies * 2.5 / norm)
        del self._postings

    @classmethod
    def from_files(cls, measurements: str | Path, manifest: str | Path, **kwargs):
        import pandas as pd

        # The first read has no target column. The second is a filtered projection
        # of training records, so held-out labels never enter the retrieval process.
        identities = pd.read_parquet(measurements, columns=["solute_id", "leakage_group"])
        split = pd.read_csv(manifest, dtype={"solute_id": str})
        if split["solute_id"].duplicated().any():
            raise ValueError("Split manifest must have exactly one row per solute")
        if not set(split["split"]).issubset({"train", "val", "test"}):
            raise ValueError("Unknown split partition")
        identities["solute_id"] = identities["solute_id"].astype(str)
        if set(identities["solute_id"]) - set(split["solute_id"]):
            raise ValueError("Measurements contain a solute missing from the split manifest")
        assignment = identities.drop_duplicates().merge(
            split[["solute_id", "split"]], on="solute_id", validate="many_to_one")
        if (assignment.groupby("solute_id")["leakage_group"].nunique() != 1).any():
            raise ValueError("A solute has inconsistent leakage-group identity")
        if (assignment.groupby("leakage_group")["split"].nunique() > 1).any():
            raise ValueError("A leakage group crosses split boundaries")
        train_solutes = set(split.loc[split["split"] == "train", "solute_id"])
        train_columns = ["record_id", "solute_id", "solute_smiles", "solvent_id", "solvent_smiles",
                         "temperature_k", "logx", "source_doi", "leakage_group"]
        train = pd.read_parquet(measurements, columns=train_columns,
                               filters=[("solute_id", "in", sorted(train_solutes))])
        if not set(train["solute_id"].astype(str)).issubset(train_solutes):
            raise ValueError("Filtered parquet read returned a held-out solute")
        cards = []
        for (solute_id, doi, temperature), group in train.groupby(
                ["solute_id", "source_doi", "temperature_k"], sort=True, dropna=False):
            if group["solute_smiles"].nunique() != 1 or group["leakage_group"].nunique() != 1:
                raise ValueError("Inconsistent structure identity in a measurement card")
            if group["solvent_id"].duplicated().any():
                raise ValueError("Duplicate solvent measurements must be resolved in the audit pipeline")
            values = []
            for row in group.itertuples(index=False):
                value = float(row.logx)
                if not math.isfinite(value) or value >= 0:
                    raise ValueError("Evidence requires finite log10 mole-fraction values below zero")
                values.append(Measurement(str(row.record_id), str(row.solvent_id),
                                          str(row.solvent_smiles), value))
            card_id = stable_hash(json.dumps([str(solute_id), str(doi), float(temperature)]))[:24]
            first = group.iloc[0]
            cards.append(Card(card_id, str(solute_id), str(first.solute_smiles),
                              str(first.leakage_group), float(temperature), str(doi), tuple(values)))
        result = cls(cards, train_solutes, **kwargs)
        result.identity_to_group = dict(zip(assignment["solute_id"].astype(str),
                                            assignment["leakage_group"].astype(str)))
        return result

    def _similarities(self, smiles: str):
        import numpy as np
        from rdkit import Chem, DataStructs

        if smiles not in self._similarity_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid query solute SMILES")
            fp = self.generator.GetFingerprint(mol)
            self._similarity_cache[smiles] = np.asarray(DataStructs.BulkTanimotoSimilarity(
                fp, [self._fingerprints[s] for s in self._unique_smiles]))[self._card_solute_index]
        return self._similarity_cache[smiles]

    def _scores(self, query: dict, mode: str):
        import numpy as np

        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if mode == "text":
            tokens = _text_tokens(" ".join([query["solute_smiles"], f"{query['temperature_k']:g}"] +
                                          [c["smiles"] for c in query["candidates"]]))
            scores = np.zeros(len(self.cards))
            for token, query_count in Counter(tokens).items():
                if token in self._bm25_postings:
                    indices, weights = self._bm25_postings[token]
                    scores[indices] += query_count * weights
            return scores
        similarity = self._similarities(query["solute_smiles"])
        if mode == "structure":
            return similarity
        candidates = {canonical_smiles(c["smiles"]) for c in query["candidates"]}
        coverage = np.asarray([len(candidates & solvents) / len(candidates)
                               for solvents in self._solvent_sets])
        temperature = np.exp(-np.abs(self._temperature - float(query["temperature_k"])) / 10.)
        result = .6 * similarity
        if mode != "no_solvent":
            result = result + .3 * coverage
        if mode != "no_temperature":
            result = result + .1 * temperature
        return result

    def retrieve(self, query: dict, mode: str = "condition", max_tokens: int = 1536,
                 token_counter: Callable[[str], int] | None = None) -> EvidenceBundle:
        import numpy as np

        public_query(query)  # Validate without passing any target fields to the scorer.
        if "solute_id" not in query or "leakage_group" not in query:
            raise ValueError("Retrieval requires identity and leakage group for self-exclusion")
        known_group = self.identity_to_group.get(str(query["solute_id"]))
        if known_group is not None and known_group != str(query["leakage_group"]):
            raise ValueError("Query leakage group disagrees with the audited identity mapping")
        safe_query = {key: query[key] for key in
                      ("solute_smiles", "temperature_k", "candidates")}
        safe_query["candidates"] = [{"smiles": c["smiles"]} for c in query["candidates"]]
        scores = self._scores(safe_query, mode)
        order = np.argsort(-scores, kind="stable")
        selected, audit, used_solutes = [], [], set()
        requested = {canonical_smiles(c["smiles"]) for c in query["candidates"]}
        coverage = {str(c["id"]): 0 for c in query["candidates"]}
        # This fallback is for CPU inspection only; LLM calls always provide its tokenizer.
        count = token_counter or (lambda text: math.ceil(len(text.encode("utf-8")) / 3))
        for index in order:
            card = self.cards[int(index)]
            if (card.solute_id == str(query["solute_id"]) or
                    card.solute_smiles == str(query["solute_smiles"]) or
                    card.leakage_group == str(query["leakage_group"]) or
                    card.solute_id in used_solutes):
                continue
            rows = sorted(card.measurements,
                          key=lambda m: (m.solvent_smiles not in requested,
                                         stable_hash(m.solvent_smiles), m.record_id))[:self.max_solvents]
            payload = {"solute_smiles": card.solute_smiles,
                       "temperature_K": card.temperature_k,
                       "measurements": [{"solvent_smiles": m.solvent_smiles,
                                         "log10_mole_fraction": round(m.logx, 6)} for m in rows]}
            candidate_payload = selected + [payload]
            if count(json.dumps(candidate_payload, separators=(",", ":"))) > max_tokens:
                continue  # Whole cards only; never truncate a structure or value.
            selected.append(payload)
            audit.append({"card_id": card.card_id, "solute_id": card.solute_id,
                          "leakage_group": card.leakage_group, "source_doi": card.source_doi,
                          "record_ids": [m.record_id for m in rows], "score": float(scores[index])})
            used_solutes.add(card.solute_id)
            included = {m.solvent_smiles for m in rows}
            for candidate in query["candidates"]:
                coverage[str(candidate["id"])] += int(canonical_smiles(candidate["smiles"]) in included)
            if len(selected) >= self.max_cards:
                break
        tokens = count(json.dumps(selected, separators=(",", ":"))) if selected else 0
        return EvidenceBundle(selected, audit, tokens, mode, coverage)


def direct_evidence_prediction(query: dict, evidence: EvidenceBundle, tie_threshold: float = .1) -> dict:
    """Rank using within-card differences, never cross-solute absolute differences.

    Uncovered comparisons receive zero difference (a tie), with explicit coverage.
    A fixed candidate-SMILES tie breaker makes final permutations reproducible.
    """
    candidates = query["candidates"]
    scores = {str(c["id"]): 0. for c in candidates}
    pairwise = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            differences = []
            for card in evidence.payload:
                values = {m["solvent_smiles"]: m["log10_mole_fraction"] for m in card["measurements"]}
                first_identity = canonical_smiles(first["smiles"])
                second_identity = canonical_smiles(second["smiles"])
                if first_identity in values and second_identity in values:
                    differences.append(values[first_identity] - values[second_identity])
            delta = sum(differences) / len(differences) if differences else 0.
            win = .5 if abs(delta) <= tie_threshold else float(delta > 0)
            scores[str(first["id"])] += win
            scores[str(second["id"])] += 1 - win
            pairwise.append({"a": str(first["id"]), "b": str(second["id"]),
                             "delta": delta, "support": len(differences)})
    if query["task"] == "pair":
        delta = pairwise[0]["delta"]
        prediction = {"choice": "TIE" if abs(delta) <= tie_threshold else ("A" if delta > 0 else "B")}
    else:
        prediction = {"ranking": [str(c["id"]) for c in sorted(
            candidates, key=lambda c: (-scores[str(c["id"])], canonical_smiles(c["smiles"]), str(c["id"])))]}
    return {"prediction": prediction, "pairwise_evidence": pairwise,
            "pair_coverage": sum(p["support"] > 0 for p in pairwise) / len(pairwise)}
