#!/usr/bin/env python3
"""Refresh bibliographic metadata from primary registries, never article bodies.

Run from any directory. Existing snapshots are overwritten only when --refresh
is supplied. A retrieval failure is recorded and produces a nonzero exit code.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
DOIS = {
    "bigsoldb2025": "10.1038/s41597-025-05559-8",
    "fastsolv2025": "10.1038/s41467-025-62717-7",
    "solvmate2024": "10.1039/D4DD00138A",
    "solubench2026": "10.1021/acs.jcim.6c00677",
    "chemprop2024": "10.1021/acs.jcim.3c01250",
    "chemprop2026": "10.1021/acs.jcim.5c02332",
    "yang2019": "10.1021/acs.jcim.9b00237",
    "deshpande2025": "10.1007/s10953-025-01495-8",
    "ecfp2010": "10.1021/ci100050t",
    "murcko1996": "10.1021/jm9602928",
    "hasolgnn2026": "10.1039/D5DD00407A",
}
FIELDS = (
    "DOI", "title", "author", "container-title", "published", "published-online",
    "published-print", "volume", "issue", "page", "article-number", "license", "URL",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "metadata"
    folder.mkdir(parents=True, exist_ok=True)
    failures = []
    for key, doi in DOIS.items():
        path = folder / f"{key}.json"
        if path.exists() and not args.refresh:
            print(f"cached {key}", flush=True)
            continue
        url = "https://api.crossref.org/works/" + doi
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "solvent-select-research/0.1"})
            with urllib.request.urlopen(req, timeout=45) as response:
                raw = response.read()
            data = json.loads(raw)["message"]
            record = {
                "source_url": url,
                "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "metadata": {field: data.get(field) for field in FIELDS},
                "note": "Publisher-deposited Crossref metadata; not a full-text verification.",
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            print(f"saved {key}: {data['title'][0]}", flush=True)
        except Exception as error:
            failures.append({"key": key, "url": url, "error": repr(error)})
            print(f"FAILED {key}: {error}", flush=True)
    (folder / "fetch_status.json").write_text(json.dumps({
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "failures": failures,
        "requested": len(DOIS),
    }, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
