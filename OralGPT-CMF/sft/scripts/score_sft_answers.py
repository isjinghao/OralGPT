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
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx
import yaml

from oral_question_prompts import load_question_prompts, prompt_for_question

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def api_endpoint(base, allow_http=False):
    base = base.rstrip('/')
    parsed = urlsplit(base)
    check(parsed.scheme in ('http', 'https') and bool(parsed.hostname), 'Invalid API base URL')
    check(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
          'API base URL must not contain credentials, query or fragment')
    check(parsed.scheme == 'https' or parsed.hostname in ('127.0.0.1', 'localhost', '::1') or allow_http,
          'Nonlocal HTTP exposes credentials and text in transit; explicit --allow-http approval required')
    return base if base.endswith('/chat/completions') else base+'/chat/completions'


def ratio(a, b):
    return a/b if b else None


def average(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def env_file(path):
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.removeprefix('export ').split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def quotes(items):
    check(isinstance(items, list) and all(isinstance(text, str) for text in items), 'quotes must be a list of strings')


def validate(data, payload, definitions):
    required = {'criteria', 'main_errors', 'unscorable_reason', 'reference_unmatched',
                'needs_review', 'review_reason', 'fact_comparison'}
    check(isinstance(data, dict) and set(data) == required, 'Incorrect response fields')
    items = data['criteria']
    check(isinstance(items, list) and len(items) == len(definitions), 'Incorrect criterion count')
    check({i['id'] for i in items} == set(definitions), 'Incorrect/duplicate criterion IDs')
    for item in items:
        check(type(item['applicable']) is bool, 'Criterion applicability must be boolean')
        check(isinstance(item['reason'], str) and bool(item['reason'].strip()), 'Criterion needs a reason')
        score = item['awarded']
        if item['applicable']:
            check(type(score) in (int, float) and math.isfinite(score) and
                  0 <= score <= definitions[item['id']]['max_score'], 'Criterion score out of range')
        else:
            check(score is None, 'N/A criterion must have null score')
        quotes(item['reference_quotes'])
        quotes(item['candidate_quotes'])
    applicable = {i['id'] for i in items if i['applicable']}
    check(isinstance(data['unscorable_reason'], str), 'Unscorable reason must be a string')
    check(bool(data['unscorable_reason'].strip()) == (not applicable), 'Explain all-N/A only when no criterion applies')
    check(isinstance(data['main_errors'], list), 'Main errors must be a list')
    for error in data['main_errors']:
        check(isinstance(error, dict) and set(error) == {'criterion_id', 'reason'}, 'Invalid main error fields')
        check(error['criterion_id'] in applicable and isinstance(error['reason'], str) and bool(error['reason'].strip()),
              'Main error needs an applicable criterion and reason')
    check(isinstance(data['reference_unmatched'], list), 'Unmatched claims must be a list')
    for extra in data['reference_unmatched']:
        quotes([extra['candidate_quote']])
        check(isinstance(extra['reason'], str) and bool(extra['reason']), 'Unmatched claim needs a reason')
    check(type(data['needs_review']) is bool and isinstance(data['review_reason'], str), 'Invalid review flag')
    check(not data['needs_review'] or bool(data['review_reason'].strip()), 'Review flag needs a reason')
    if payload['category'] == 'perception':
        validate_comparison(data['fact_comparison'])
    else:
        check(data['fact_comparison'] is None, 'Non-perception fact_comparison must be null')
    return data


def comparison_counts(comparison):
    comparison = [row for row in comparison if row['scored']]
    tp = sum(row['matched'] for row in comparison)
    ref = sum(row['reference'] is not None for row in comparison)
    pred = sum(row['candidate'] is not None for row in comparison)
    return dict(reference_total=ref, candidate_total=pred, tp=tp, fn=ref-tp, fp_reference_unmatched=pred-tp)


def validate_comparison(comparison):
    check(isinstance(comparison, list), 'fact_comparison must be a list')
    for index, row in enumerate(comparison):
        path = f'fact_comparison[{index}]'
        check(isinstance(row, dict) and set(row) == {'field', 'reference', 'candidate', 'matched', 'scored', 'reason'}, 'Invalid comparison row: '+path)
        check(all(isinstance(row[k], str) for k in ('field', 'reason')) and bool(row['reason'].strip()), 'Field and nonempty reason must be strings: '+path)
        check(type(row['matched']) is bool, 'matched must be boolean: '+path)
        check(type(row['scored']) is bool, 'scored must be boolean: '+path)
        check(row['reference'] is not None or row['candidate'] is not None, 'Both sides missing: '+path)
        check(not row['matched'] or (row['reference'] is not None and row['candidate'] is not None), 'Match needs both sides: '+path)
        for side in ('reference', 'candidate'):
            fact = row[side]
            if fact is None:
                continue
            check(isinstance(fact, dict) and set(fact) == {'text', 'quotes', 'values', 'unit'}, 'Invalid fact fields: '+path+'.'+side)
            check(isinstance(fact['text'], str) and isinstance(fact['unit'], str), 'Fact text and unit must be strings: '+path+'.'+side)
            quotes(fact['quotes'])
            values = fact['values']
            check(isinstance(values, list) and len(values) in (0, 1, 2) and
                  all(type(v) in (int, float) and math.isfinite(v) for v in values),
                  'values must be [] (qualitative), [number] or [low,high] (ONE range): '+path+'.'+side)
    return comparison


def prf(tp, fp, fn):
    return dict(tp=tp, fp_reference_unmatched=fp, fn=fn, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn), jaccard=ratio(tp, tp+fp+fn))


def fact_metrics(comparison):
    counts = comparison_counts(comparison)
    result = prf(counts['tp'], counts['fp_reference_unmatched'], counts['fn'])
    result.update(reference_count=counts['reference_total'], candidate_count=counts['candidate_total'])
    return result


def numeric_summary(comparisons):
    pairs = []
    for row in (row for comparison in comparisons for row in comparison):
        a, b = row['reference'], row['candidate']
        if a is None or not a['values']:
            continue
        x = a['values'][0] if len(a['values']) == 1 else None
        y = b['values'][0] if b is not None and len(b['values']) == 1 else None
        unit_match = b is not None and a['unit'] == b['unit'] and a['unit'] not in ('', 'unspecified')
        comparable = x is not None and y is not None and unit_match
        pairs.append(dict(reference_value=x, candidate_value=y, reference_unit=a['unit'],
                          unit_match=unit_match, comparable=comparable,
                          absolute_error=abs(x-y) if comparable else None,
                          value_match=x == y if comparable else None,
                          sign_match=((x > 0)-(x < 0)) == ((y > 0)-(y < 0)) if comparable else None))
    def summarize(rows):
        eligible = [p for p in rows if p['comparable']]
        return dict(reference_fields=len(rows), comparable_fields=len(eligible),
                    strict_value_accuracy=average([p['value_match'] for p in eligible]),
                    sign_accuracy=average([p['sign_match'] for p in eligible]),
                    mae=average([p['absolute_error'] for p in eligible]))
    unit_eligible = [p for p in pairs if p['reference_value'] is not None and p['candidate_value'] is not None
                     and p['reference_unit'] not in ('', 'unspecified')]
    comparable = sum(p['comparable'] for p in pairs)
    return dict(reference_fields=len(pairs), comparable_fields=comparable,
                comparison_coverage=ratio(comparable, len(pairs)), noncomparable_fields=len(pairs)-comparable,
                unit_accuracy=average([p['unit_match'] for p in unit_eligible]), unit_denominator=len(unit_eligible),
                by_unit={u: summarize([p for p in pairs if p['reference_unit'] == u])
                         for u in sorted({p['reference_unit'] for p in pairs})})


def calculate(data, payload, definitions):
    items = [i for i in data['criteria'] if i['applicable']]
    awarded = sum(i['awarded'] for i in items)
    possible = sum(definitions[i['id']]['max_score'] for i in items)
    result = dict(rubric_percent=round(100*awarded/possible, 1) if possible else None,
                  rubric_awarded=awarded, rubric_applicable_max=possible,
                  needs_review=data['needs_review'],
                  dimensions={i['id']: dict(awarded=i['awarded'], max_score=definitions[i['id']]['max_score'],
                              percent=100*i['awarded']/definitions[i['id']]['max_score'] if i['applicable'] else None)
                              for i in data['criteria']})
    if payload['category'] == 'perception':
        result['facts'] = fact_metrics(data['fact_comparison'])
    return result


def aggregate(records, expected):
    completed = [r for r in records if r['status'] == 'completed']
    categories = {}
    for category in ('perception', 'diagnosis', 'treatment'):
        selected = [r for r in completed if r['category'] == category]
        metrics = [r['metrics'] for r in selected]
        facts = [m['facts'] for m in metrics if 'facts' in m]
        scored = sum(m['rubric_percent'] is not None for m in metrics)
        item = dict(expected_questions=expected.get(category, 0), completed_questions=len(selected),
                    scored_questions=scored, unscorable_questions=len(selected)-scored,
                    rubric_percent=average([m['rubric_percent'] for m in metrics]),
                    review_questions=sum(m['needs_review'] for m in metrics),
                    dimensions={k: dict(applicable_questions=sum(m['dimensions'][k]['percent'] is not None for m in metrics),
                                percent=average([m['dimensions'][k]['percent'] for m in metrics]))
                                for k in (metrics[0]['dimensions'] if metrics else [])})
        if category == 'perception':
            fact_scored = sum(m['reference_count']+m['candidate_count'] > 0 for m in facts)
            item.update(fact_scored_questions=fact_scored, fact_question_coverage=ratio(fact_scored, expected.get(category, 0)),
                        fact_micro=prf(sum(m['tp'] for m in facts), sum(m['fp_reference_unmatched'] for m in facts), sum(m['fn'] for m in facts)),
                        fact_macro_f1=average([m['f1'] for m in facts]),
                        numeric=numeric_summary([r['judge']['fact_comparison'] for r in selected]))
        categories[category] = item
    scored = sum(c['scored_questions'] for c in categories.values())
    category_scores = [c['rubric_percent'] for c in categories.values()]
    return dict(expected_questions=sum(expected.values()), completed_questions=len(completed),
                scored_questions=scored, unscorable_questions=len(completed)-scored,
                failed_questions=sum(r['status'] == 'failed' for r in records), complete=len(completed) == sum(expected.values()),
                question_macro_percent=average([r['metrics']['rubric_percent'] for r in completed]),
                category_macro_percent=average(category_scores) if all(s is not None for s in category_scores) else None,
                categories_with_scores=sum(s is not None for s in category_scores), categories=categories)


class JudgeFailure(RuntimeError):
    def __init__(self, message, attempts, usage):
        super().__init__(message)
        self.attempts, self.usage = attempts, dict(usage)


class Judge:
    def __init__(self, client, endpoint, model, spec, tokens, attempts, reasoning):
        self.client, self.endpoint, self.model, self.spec = client, endpoint, model, spec
        self.tokens, self.attempts, self.reasoning = tokens, attempts, reasoning

    def score(self, payload, definitions):
        system = self.spec['judge_system']+'\n'+self.spec['output_contract']
        if payload['category'] == 'perception':
            system += '\n'+self.spec['matching_system']
        messages = [dict(role='system', content=system),
                    dict(role='user', content=json.dumps(dict(**payload, criteria=definitions), ensure_ascii=False))]
        budget, usage = self.tokens, Counter()
        for attempt in range(self.attempts):
            content = None
            try:
                body = dict(model=self.model, messages=messages, response_format=dict(type='json_object'), max_completion_tokens=budget)
                if self.reasoning != 'omit':
                    body['reasoning_effort'] = self.reasoning
                response = self.client.post(self.endpoint, json=body)
                if response.status_code >= 400:
                    last_error = f'HTTP {response.status_code}'
                    if response.status_code not in (408, 409, 429) and response.status_code < 500:
                        raise RuntimeError(last_error+'; non-retryable API error (response body not logged)')
                    raise httpx.TransportError(last_error)
                obj = response.json()
                usage.update({k: obj.get('usage', {}).get(k, 0) or 0
                              for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
                choice = obj['choices'][0]
                content = choice['message']['content']
                if choice['finish_reason'] == 'length':
                    budget = min(budget*2, 32768)
                    raise ValueError('Judge output truncated')
                check(choice['finish_reason'] == 'stop', 'Judge response did not finish normally')
                data = validate(json.loads(content), payload, definitions)
                return dict(judge=data, usage=dict(usage), attempts=attempt+1)
            except (ValueError, KeyError, TypeError, httpx.TransportError) as error:
                last_error = str(error) if isinstance(error, ValueError) else type(error).__name__
                if isinstance(error, (ValueError, KeyError, TypeError)):
                    previous = [dict(role='assistant', content=content)] if content else []
                    messages = messages[:2]+previous+[dict(role='user', content=json.dumps(dict(
                        validation_error=last_error,
                        instruction='Fix the stated JSON format or field type error and return the whole object. Preserve your scoring and matching decisions and all findings. Do not return counts; software computes them.'), ensure_ascii=False))]
                if attempt+1 < self.attempts:
                    time.sleep(min(30, 2**(attempt+1)))
        raise JudgeFailure(f'Judge failed after {self.attempts} attempts: {last_error}', self.attempts, usage)


def run_job(job, judge, definitions):
    path = job['path']
    if path.exists():
        saved = read(path)
        check(saved['input'] == job['payload'], 'Cached scoring input changed; use a new run-name')
        if saved['status'] == 'completed':
            saved['metrics'] = calculate(saved['judge'], job['payload'], definitions)
            write(path, saved)
            return saved, True
    result = dict(status='failed', task_id=job['task_id'], task=job['task'],
                  category=job['payload']['category'], input=job['payload'])
    started = time.monotonic()
    try:
        result.update(judge.score(job['payload'], definitions))
        result['metrics'] = calculate(result['judge'], job['payload'], definitions)
        result['status'] = 'completed'
    except Exception as error:
        result['error'] = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
        if isinstance(error, JudgeFailure):
            result.update(usage=error.usage, attempts=error.attempts)
    result['seconds'] = round(time.monotonic()-started, 2)
    write(path, result)
    return result, False


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(' ')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def prepare(args, spec, routing, endpoint):
    for name in (args.answer_model, args.judge_model, args.run_name):
        check(bool(re.fullmatch(r'[A-Za-z0-9_.-]+', name)) and name not in ('.', '..'), 'Invalid output directory name')
    files = sorted(args.output_root.glob(f'*/*/evaluation/{args.answer_model}/answers.json'))
    requested = set(args.case_id)
    case_id = lambda p: p.parents[3].name+'__'+p.parents[2].name
    if requested:
        files = [p for p in files if case_id(p) in requested]
        check({case_id(p) for p in files} == requested, 'Requested patient not found')
    if args.limit is not None:
        files = files[:args.limit]
    check(bool(files), 'No answer files selected')
    check(all(sum(c['max_score'] for c in s['criteria'].values()) == 100 for s in spec['categories'].values()), 'Each rubric must total 100')
    task_category = {t: c for c, s in spec['categories'].items() for t in s['tasks']}
    config = dict(judge_model=args.judge_model, answer_model=args.answer_model, spec=spec,
                  endpoint=endpoint, tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
                  cases=[case_id(p) for p in files])
    jobs = []
    for path in files:
        rows = read(path)
        check(isinstance(rows, list) and bool(rows), 'Empty or invalid answer file')
        check(len({r['task_id'] for r in rows}) == len(rows), 'Duplicate task_id')
        check(len({r['turn_index'] for r in rows}) == len(rows), 'Duplicate turn_index')
        patient = path.parents[2].name
        for row in rows:
            check(bool(row['gold_answer'].strip()), 'Reference answer is empty')
            task, _ = prompt_for_question(routing, row['question'])
            category = task_category[task]
            redact = lambda s: s.replace('<image>', '').replace(patient, '[PATIENT]').strip()
            payload = dict(category=category, question=redact(row['question']), reference_answer=redact(row['gold_answer']),
                           candidate_answer=redact(row['model_answer']))
            folder = path.parent/'scoring'/args.judge_model/args.run_name
            jobs.append(dict(case_id=case_id(path), task_id=row['task_id'], task=task, payload=payload,
                             path=folder/f'turn_{row["turn_index"]:02d}.json'))
    return config, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--answer-model', default='qwen3.5-9b-v3')
    parser.add_argument('--judge-model', default='gpt-5-mini')
    parser.add_argument('--run-name', default='reference_v2_1_20261020')
    parser.add_argument('--case-id', action='append', default=[])
    parser.add_argument('--limit', type=int)
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
    args = parser.parse_args()
    check(args.numworker > 0 and args.max_attempts > 0 and 0 < args.max_tokens <= 32768 and args.timeout > 0, 'Invalid runtime limits')
    check(args.limit is None or args.limit > 0, 'Limit must be positive')
    args.output_root = args.output_root or args.root/'output'
    spec_path = args.prompts or args.root/'prompts/oral_scoring.yaml'
    spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    routing = load_question_prompts(args.routing or args.root/'sft_data_v3_question_prompts/question_prompts.yaml')
    env_file(args.env_file or args.root/'.env')
    base = os.environ.get(args.base_url_env, '')
    endpoint = api_endpoint(base, allow_http=args.allow_http or args.dry_run)
    config, jobs = prepare(args, spec, routing, endpoint)
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
        with httpx.Client(headers={'Authorization': 'Bearer '+key}, timeout=args.timeout, follow_redirects=False,
                          limits=httpx.Limits(max_connections=args.numworker, max_keepalive_connections=args.numworker)) as client:
            judge = Judge(client, endpoint, args.judge_model, spec, args.max_tokens, args.max_attempts, args.reasoning_effort)
            with ThreadPoolExecutor(max_workers=args.numworker) as pool:
                futures = {pool.submit(run_job, j, judge, spec['categories'][j['payload']['category']]['criteria']): j for j in jobs}
                for index, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    result, reused = future.result()
                    cached += reused
                    results[job['case_id']].append(result)
                    print(f'[{index}/{len(jobs)}] {job["task"]} {result["status"]} cached={reused}', flush=True)
        all_records = [r for records in results.values() for r in records]
        patients = {case: aggregate(records, Counter(j['payload']['category'] for j in jobs if j['case_id'] == case))
                    for case, records in results.items()}
        total = aggregate(all_records, Counter(j['payload']['category'] for j in jobs))
        report = dict(cached_questions=cached, patients=len(patients),
                      patient_macro_percent=average([p['category_macro_percent'] for p in patients.values()]),
                      by_patient=patients, **total)
        usage = Counter()
        for record in all_records:
            usage.update(record.get('usage', {}))
        report['usage_totals'] = dict(usage)
        write(folder/'summary.json', report)
        print(json.dumps(dict(complete=report['complete'], scored_questions=report['scored_questions'],
                              failed_questions=report['failed_questions'], cached_questions=cached,
                              summary=str(folder/'summary.json')), ensure_ascii=False), flush=True)
        return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
