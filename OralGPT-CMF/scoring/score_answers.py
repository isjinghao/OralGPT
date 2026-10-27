#!/usr/bin/env python3
"""Text-only reference scoring. Example: --answer-model qwen3.5-9b-v3 --limit 1 --numworker 3.

Requires the existing PyYAML/httpx environment. GPT credentials default to root/.env
ANSWER_OPENAI_API_KEY and ANSWER_OPENAI_BASE_URL; the judge model is explicitly gpt-5-mini.
A flat thread pool spans patients AND questions. Rerun the same command to resume;
changed inputs/specification require a new --run-name. Never edits original answers.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.question_prompts import load_question_prompts, prompt_for_question
from scoring.judge import Judge, validate
from scoring.utils import aggregate, api_endpoint, average, calculate, check, env_file, read, run_lock, write


def run_job(job, judge, definitions):
    path = job['path']
    if path.exists():
        saved = read(path)
        if saved['status'] == 'completed':
            check(saved['input'] == job['payload'], 'Cached scoring input changed; use a new run-name')
            validate(saved['judge'], job['payload'], definitions)
            metrics = calculate(saved['judge'], job['payload'], definitions)
            if saved['metrics'] != metrics:
                saved['metrics'] = metrics
                write(path, saved)
            return saved, True
    result = dict(status='failed', task_id=job['task_id'], task=job['task'],
                  category=job['payload']['category'], input=job['payload'])
    try:
        if 'reference_error' in job:
            raise job['reference_error']
        result.update(judge.score(job['payload'], definitions))
        result['metrics'] = calculate(result['judge'], job['payload'], definitions)
        result['status'] = 'completed'
    except Exception as error:
        result['error'] = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
    write(path, result)
    return result, False


def freeze_requirements(jobs, judge, folder, workers):
    for category, key in [('treatment', 'applicability'), ('perception', 'reference_facts')]:
        selected = [j for j in jobs if j['payload']['category'] == category]
        if not selected:
            continue
        definitions = judge.spec['categories'][category]['criteria']
        shared = folder/category
        with run_lock(shared/'reference.lock', blocking=True):
            pending = []
            for job in selected:
                path = shared/job['case_id']/f'{job["task"]}.json'
                payload = {k: job['payload'][k] for k in ('category', 'question', 'reference_answer')}
                if path.exists():
                    saved = read(path)
                    check(saved['input'] == payload and saved['criteria'] == definitions and saved['judge_model'] == judge.model,
                          'Shared reference inputs changed; use a new run-name')
                    job['payload'][key] = saved[key]
                else:
                    pending.append(job)

            def decide(job):
                payload = {k: job['payload'][k] for k in ('category', 'question', 'reference_answer')}
                try:
                    result = judge.score(payload, definitions, applicability_only=category == 'treatment', reference_only=category == 'perception')
                except RuntimeError as error:
                    job['reference_error'] = error
                    print(f'Reference {job["task"]} failed: {error}; continuing other questions', flush=True)
                    return
                items = result['judge'][key]
                path = shared/job['case_id']/f'{job["task"]}.json'
                write(path, dict(input=payload, criteria=definitions, judge_model=judge.model, **{key: items}))
                job['payload'][key] = items

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(decide, pending))


def prepare(args, spec, routing, endpoint):
    files = sorted(args.output_root.glob(f'*/*/evaluation/{args.answer_model}/answers.json'))
    requested = set(args.case_id)
    case_id = lambda p: p.parents[3].name+'__'+p.parents[2].name
    if requested:
        files = [p for p in files if case_id(p) in requested]
        check({case_id(p) for p in files} == requested, 'Requested patient not found')
    if args.limit is not None:
        files = files[:args.limit]
    check(bool(files), 'No answer files selected')
    task_category = {t: c for c, s in spec['categories'].items() for t in s['tasks']}
    config = dict(judge_model=args.judge_model, answer_model=args.answer_model, spec=spec,
                  endpoint=endpoint, tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
                  cases=[case_id(p) for p in files])
    jobs = []
    for path in files:
        patient = path.parents[2].name
        folder = path.parent/'scoring'/args.judge_model/args.run_name
        for row in read(path):
            task, _ = prompt_for_question(routing, row['question'])
            category = task_category[task]
            redact = lambda s: s.replace('<image>', '').replace(patient, '[PATIENT]').strip()
            payload = dict(category=category, question=redact(row['question']), reference_answer=redact(row['gold_answer']),
                           candidate_answer=redact(row['model_answer']))
            jobs.append(dict(case_id=case_id(path), task_id=row['task_id'], task=task, payload=payload,
                             path=folder/f'turn_{row["turn_index"]:02d}.json'))
    return config, jobs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--answer-model', default='qwen3.5-9b-v3')
    parser.add_argument('--judge-model', default='gpt-5-mini')
    parser.add_argument('--run-name', default='all77')
    parser.add_argument('--case-id', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--category', choices=['perception', 'diagnosis', 'treatment'])
    parser.add_argument('--numworker', '--num-workers', dest='numworker', type=int, default=4)
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--routing', type=Path)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--api-key-env', default='ANSWER_OPENAI_API_KEY')
    parser.add_argument('--base-url-env', default='ANSWER_OPENAI_BASE_URL')
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high', 'omit'], default='low')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-http', action='store_true', help='Explicitly permit unencrypted transmission to the configured API')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root or args.root/'output'
    spec_path = args.prompts or args.root/'scoring/prompts/oral_scoring.yaml'
    spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    routing = load_question_prompts(args.routing or args.root/'prompts/oral_question_prompts.yaml')
    env_file(args.env_file or args.root/'.env')
    base = os.environ.get(args.base_url_env, '')
    endpoint = api_endpoint(base, allow_http=args.allow_http or args.dry_run)
    config, jobs = prepare(args, spec, routing, endpoint)
    if args.category:
        jobs = [j for j in jobs if j['payload']['category'] == args.category]
        config['category'] = args.category
    print(json.dumps(dict(patients=len(config['cases']), questions=len(jobs), numworker=args.numworker,
                          categories=dict(Counter(j['payload']['category'] for j in jobs)), judge_model=args.judge_model,
                          dry_run=args.dry_run), ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    key = os.environ.get(args.api_key_env)
    check(bool(key), 'API credential missing')
    folder = args.output_root/'scoring'/args.answer_model/args.judge_model/args.run_name
    with run_lock(folder/'run.lock'):
        manifest = folder/'run.json'
        if manifest.exists():
            check(read(manifest) == config, 'Run configuration changed; use a new run-name')
        else:
            write(manifest, config)
        results, cached = defaultdict(list), 0
        expected = defaultdict(Counter)
        for job in jobs:
            expected[job['case_id']][job['payload']['category']] += 1
        patients = {}
        with httpx.Client(headers={'Authorization': 'Bearer '+key}, timeout=args.timeout, follow_redirects=False,
                          limits=httpx.Limits(max_connections=args.numworker, max_keepalive_connections=args.numworker)) as client:
            judge = Judge(client, endpoint, args.judge_model, spec, args.max_tokens, args.max_attempts, args.reasoning_effort)
            freeze_requirements(jobs, judge, args.output_root/'scoring'/'reference'/args.judge_model/args.run_name, args.numworker)
            with ThreadPoolExecutor(max_workers=args.numworker) as pool:
                futures = {pool.submit(run_job, j, judge, spec['categories'][j['payload']['category']]['criteria']): j
                           for j in sorted(jobs, key=lambda job: job['path'].name)}
                for index, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    result, reused = future.result()
                    cached += reused
                    case = job['case_id']
                    results[case].append(result)
                    if len(results[case]) == sum(expected[case].values()):
                        summary = aggregate(results[case], expected[case])
                        patients[case] = summary
                        group, patient = case.split('__', 1)
                        write(job['path'].parent/'summary.json',
                              dict(case_id=case, group=group, patient_name=patient,
                                   answer_model=args.answer_model, judge_model=args.judge_model, **summary))
                    print(f'[{index}/{len(jobs)}] {job["task"]} {result["status"]} cached={reused}', flush=True)
        all_records = [r for records in results.values() for r in records]
        total = aggregate(all_records, Counter(j['payload']['category'] for j in jobs))
        report = dict(cached_questions=cached, patients=len(patients),
                      patient_macro_percent=average([p['category_macro_percent'] for p in patients.values()]),
                      by_patient=patients, **total)
        write(folder/'summary.json', report)
        print(json.dumps(dict(complete=report['complete'], scored_questions=report['scored_questions'],
                              failed_questions=report['failed_questions'], cached_questions=cached,
                              summary=str(folder/'summary.json')), ensure_ascii=False), flush=True)
        return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
