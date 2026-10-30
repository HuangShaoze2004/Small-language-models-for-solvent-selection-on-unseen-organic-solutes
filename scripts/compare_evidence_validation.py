#!/usr/bin/env python3
"""All fixed-evidence validation contrasts, including unfavorable ablations."""
from pathlib import Path
import json

from solvent_select.compare import compare
from solvent_select.views import sha256_file, write_json
from solvent_select.study import load_registry,evidence_name


def main():
    output = Path('artifacts/reports/dualqa_evidence_validation_comparisons')
    if output.exists():
        raise FileExistsError('Preserve the previous analysis; use its artifacts or a documented new version')
    output.mkdir(parents=True)
    rows, files = [], {}
    for protocol,config in load_registry()['protocols'].items():
        def evaluation(mode):
            return Path(config['runs_root'])/evidence_name(protocol,config,mode)/'evaluation_view10000'
        for reference in ['structure', 'text', 'no_temperature', 'no_solvent']:
            path = output / f'{protocol}_condition_minus_{reference}.json'
            result = compare([evaluation('condition')], [evaluation(reference)], path, expected_seeds=1)
            if result['status'] != 'VALIDATION_ONLY':
                raise ValueError('Only validation contrasts are permitted')
            rows.append({'protocol': protocol, 'first': 'condition', 'second': reference,
                         'comparisons': result['comparisons']})
            files[str(path)] = sha256_file(path)
    write_json(output / 'manifest.json', {'status': 'VALIDATION_ONLY', 'deterministic_runs_per_method': 1,
        'code_sha256': sha256_file(__file__), 'outputs': files,
        'interpretation': 'Descriptive evidence-workflow contrasts, not the preregistered three-seed LM hypothesis. All4comparisons on both protocols retained; no test access or hyperparameter selection.'})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
