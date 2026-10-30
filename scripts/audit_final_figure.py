#!/usr/bin/env python3
"""Recount final figure points and intervals independently of its plotter."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd

from solvent_select.views import sha256_file, write_json

spec = importlib.util.spec_from_file_location('independent_bootstrap', Path(__file__).with_name('audit_paired_validation_contrast.py'))
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', required=True)
    parser.add_argument('--grade-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve earlier audits')
    manifest_path = Path(args.figure) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    grade = json.loads(Path(args.grade_audit).read_text())
    if manifest['status'] != 'FINAL_METHODS_COMPLETE' or len(grade['runs']) != 62:
        raise ValueError('Require complete registered figure and grading')
    for path, digest in manifest['sources'].items():
        if sha256_file(path) != digest:
            raise ValueError('Figure source changed')
    for name, digest in manifest['outputs'].items():
        if sha256_file(Path(args.figure) / name) != digest:
            raise ValueError('Figure output changed')
    sources = {__file__: sha256_file(__file__), args.grade_audit: sha256_file(args.grade_audit),
               str(manifest_path): sha256_file(manifest_path), independent.__file__: sha256_file(independent.__file__)}
    runs = {r['run']: r for r in grade['runs']}
    folders = {Path(p).parent.parent.name: Path(p).parent for p in grade['evidence_sha256'] if p.endswith('/per_solute.csv')}
    table = pd.read_csv(Path(args.figure) / 'summary.csv')
    if len(table) != 102 or table.duplicated(['protocol', 'method', 'task', 'metric']).any():
        raise ValueError('Wrong summary population')
    checked = []
    covered = set()
    for row in table.to_dict('records'):
        names = row['run_ids'].split(';')
        if len(set(names)) != int(row['runs']):
            raise ValueError('Wrong seed count')
        frames = []
        expected = []
        for name in names:
            run = runs[name]
            if run['status'] != 'PASSED' or run['protocol'] != row['protocol']:
                raise ValueError('Ungraded or wrong-protocol method')
            p = folders[name] / 'per_solute.csv'
            if sha256_file(p) != grade['evidence_sha256'][str(p)]:
                raise ValueError('Audited scores changed')
            sources[str(p)] = sha256_file(p)
            frame = pd.read_csv(p, dtype={'solute_id': str}).set_index(['solute_id', 'task']).sort_index()
            frames.append(frame.xs(row['task'], level='task')[row['metric']])
            expected.append(run['independent_metrics'][row['task']][row['metric']])
            covered.add((name, row['task'], row['metric']))
        if any(not frames[0].index.equals(frame.index) for frame in frames):
            raise ValueError('Seed support differs')
        vectors = [frame.to_dict() for frame in frames]
        values = [math.fsum(v[sid] for v in vectors) / len(vectors) for sid in frames[0].index]
        mean, interval = independent.bootstrap(values)
        error = max(abs(mean-row['value']), abs(interval[0]-row['ci_low']), abs(interval[1]-row['ci_high']),
                    abs(mean-math.fsum(expected)/len(expected)))
        if error > 1e-12 or len(values) != row['solute_count']:
            raise ValueError('Independent figure estimate/interval mismatch')
        checked.append({'protocol': row['protocol'], 'method': row['method'], 'task': row['task'],
                        'metric': row['metric'], 'maximum_arithmetic_difference': error})
    if len(covered) != 62 * 3:
        raise ValueError('Not all method/metric combinations represented')
    write_json(output, {'status': 'PASSED_ALL_FINAL_FIGURE_POINTS_AND_INTERVALS', 'rows': checked,
        'registered_method_metric_combinations': len(covered), 'evidence_sha256': sources,
        'scope': 'All102 final summary rows; independent seed/solute mean and count-based paired-index bootstrap arithmetic; no model calls or method selection.'})
    print(json.dumps({'rows': len(checked), 'maximum_error': max(r['maximum_arithmetic_difference'] for r in checked)}))


if __name__ == '__main__':
    main()
