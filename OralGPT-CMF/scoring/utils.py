"""File operations, locking and arithmetic for reference scoring."""
from contextlib import contextmanager
import fcntl
import json
import os
from urllib.parse import urlsplit


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


def env_file(path):
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.removeprefix('export ').split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@contextmanager
def run_lock(path, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def ratio(a, b):
    return a/b if b else None


def average(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def prf(tp, fp, fn):
    return dict(tp=tp, fp_reference_unmatched=fp, fn=fn, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn), jaccard=ratio(tp, tp+fp+fn))


def fact_metrics(comparison):
    rows = [row for row in comparison if row['scored']]
    tp = sum(row['matched'] for row in rows)
    ref = sum(row['reference'] is not None for row in rows)
    pred = sum(row['candidate'] is not None for row in rows)
    return dict(prf(tp, pred-tp, ref-tp), reference_count=ref, candidate_count=pred)


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
    clinical = 100*awarded/possible if possible else None
    penalty = data['expression']['penalty']
    result = dict(rubric_percent=round(max(0, clinical-penalty), 1) if clinical is not None else None,
                  clinical_percent=round(clinical, 1) if clinical is not None else None, expression_penalty=penalty,
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
                    clinical_percent=average([m['clinical_percent'] for m in metrics]),
                    expression_penalty=average([m['expression_penalty'] for m in metrics]),
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
