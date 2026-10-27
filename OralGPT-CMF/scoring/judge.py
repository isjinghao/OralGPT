"""Judge requests, evidence assembly and validation of model-generated results."""
import json
import math
from pathlib import Path
import re
import time

import httpx
import yaml

from scoring.utils import check

RUNTIME_PROMPTS = yaml.safe_load(
    (Path(__file__).resolve().parent/'prompts/scoring_runtime.yaml').read_text(encoding='utf-8'))


def quotes(items, source=None):
    check(isinstance(items, list) and all(isinstance(text, str) for text in items), 'quotes must be a list of strings')
    if source is not None:
        invalid = [text for text in items if not text.strip() or text not in source]
        check(not invalid, f'Quotes must be exact contiguous source substrings: {invalid!r}')


def validate(data, payload, definitions):
    required = {'criteria', 'main_errors', 'unscorable_reason', 'reference_unmatched',
                'expression', 'needs_review', 'review_reason', 'fact_comparison'}
    check(isinstance(data, dict) and set(data) == required, 'Incorrect response fields')
    items = data['criteria']
    check(isinstance(items, list) and len(items) == len(definitions), 'Incorrect criterion count')
    check({i['id'] for i in items} == set(definitions), 'Incorrect/duplicate criterion IDs')
    fixed = {i['id']: i['applicable'] for i in payload.get('applicability', [])}
    for item in items:
        check(type(item['applicable']) is bool, 'Criterion applicability must be boolean')
        if fixed:
            check(item['applicable'] == fixed[item['id']],
                  f'Frozen applicability mismatch for {item["id"]}: expected {fixed[item["id"]]}')
        check(isinstance(item['reason'], str) and bool(item['reason'].strip()), 'Criterion needs a reason')
        score = item['awarded']
        if item['applicable']:
            check(type(score) in (int, float) and math.isfinite(score) and
                  0 <= score <= definitions[item['id']]['max_score'], 'Criterion score out of range')
        else:
            check(score is None, 'N/A criterion must have null score')
        quotes(item['reference_quotes'], payload['reference_answer'])
        quotes(item['candidate_quotes'], payload['candidate_answer'])
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
        quotes([extra['candidate_quote']], payload['candidate_answer'])
        check(isinstance(extra['reason'], str) and bool(extra['reason']), 'Unmatched claim needs a reason')
    expression = data['expression']
    check(isinstance(expression, dict) and set(expression) == {'penalty', 'reason', 'quotes', 'reference_quotes'}, 'Invalid expression fields')
    penalty = expression['penalty']
    check(type(penalty) in (int, float) and math.isfinite(penalty) and 0 <= penalty <= 20, 'Expression penalty out of range')
    check(isinstance(expression['reason'], str) and bool(expression['reason'].strip()), 'Expression needs a reason')
    quotes(expression['quotes'])
    invalid = [(i, text) for i, text in enumerate(expression['quotes']) if not text.strip() or text not in payload['candidate_answer']]
    check(not invalid, f'Expression quotes must occur in the candidate; invalid expression.quotes entries: {invalid!r}')
    quotes(expression['reference_quotes'])
    invalid = [(i, text) for i, text in enumerate(expression['reference_quotes']) if not text.strip() or text not in payload['reference_answer']]
    check(not invalid, f'Omission quotes must occur in the reference; invalid expression.reference_quotes entries: {invalid!r}')
    check(not penalty or bool(expression['quotes'] or expression['reference_quotes']), 'Expression penalty needs quoted evidence')
    check(bool(applicable) or penalty == 0, 'All-N/A cannot have an expression penalty')
    check(type(data['needs_review']) is bool and isinstance(data['review_reason'], str), 'Invalid review flag')
    check(not data['needs_review'] or bool(data['review_reason'].strip()), 'Review flag needs a reason')
    if payload['category'] == 'perception':
        validate_comparison(data['fact_comparison'], payload)
        if 'reference_facts' in payload:
            rows = [r for r in data['fact_comparison'] if r['reference'] is not None]
            expected = payload['reference_facts']
            check(len(rows) == len(expected) and len({r['field'] for r in rows}) == len(expected) and
                  {r['field'] for r in rows} == {r['field'] for r in expected}, 'Frozen reference facts missing or duplicated')
            fixed = {r['field']: r for r in expected}
            for row in rows:
                check(row['reference'] == fixed[row['field']]['reference'] and row['scored'] == fixed[row['field']]['scored'],
                      'Frozen reference fact changed: '+row['field'])
    else:
        check(data['fact_comparison'] is None, 'Non-perception fact_comparison must be null')
    return data


def validate_comparison(comparison, payload):
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
            check(bool(fact['text'].strip()) and bool(fact['quotes']), 'Empty fact side must be null: '+path+'.'+side)
            quotes(fact['quotes'], payload[side+'_answer'])
            values = fact['values']
            check(isinstance(values, list) and len(values) in (0, 1, 2) and
                  all(type(v) in (int, float) and math.isfinite(v) for v in values),
                  'values must be [] (qualitative), [number] or [low,high] (ONE range): '+path+'.'+side)
    return comparison


def assemble_comparison(data, payload):
    fixed = {f['field']: f for f in payload['reference_facts']}
    rows = data['fact_comparison']
    fields = [r['field'] for r in rows]
    check(len(fields) == len(set(fields)) and set(fixed) <= set(fields), 'Frozen reference facts missing or duplicated')
    for row in rows:
        if row['field'] in fixed:
            fact = fixed[row['field']]
            check(set(row) == {'field', 'candidate', 'matched', 'reason'}, 'Use compact rows without reference/scored for frozen fields')
            row.update(reference=fact['reference'], scored=fact['scored'])
        else:
            check(set(row) == {'field', 'candidate', 'matched', 'scored', 'reason'}, 'Invalid compact addition row')
            row['reference'] = None
    return data


def assemble_quotes(data, passages):
    def resolve(value, side):
        if type(value) is int:
            check(0 <= value < len(passages[side]), 'Invalid quote passage index for '+side)
            return passages[side][value]
        return value

    def visit(value, side='candidate'):
        if isinstance(value, list):
            for item in value:
                visit(item, side)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ('quotes', 'candidate_quotes', 'reference_quotes'):
                    source = 'reference' if key == 'reference_quotes' else side if key == 'quotes' else 'candidate'
                    check(isinstance(item, list), 'quotes must be a list')
                    value[key] = [resolve(q, source) for q in item]
                elif key == 'candidate_quote':
                    value[key] = resolve(item, 'candidate')
                else:
                    visit(item, key if key in ('candidate', 'reference') else side)

    visit(data)
    return data


def system_prompt(spec, payload, applicability_only=False, reference_only=False):
    category = payload['category']
    contract = spec['output_contract']
    if 'reference_facts' in payload:
        contract = contract.split('fact_comparison must be null')[0]
    system = spec['judge_system']+'\n'+spec['judge_common']+'\n'+contract
    if category in ('diagnosis', 'treatment'):
        system += '\n'+spec[category+'_system']
    if category != 'diagnosis':
        system += '\n'+RUNTIME_PROMPTS['non_diagnosis']
    if 'reference_facts' in payload:
        system += '\n'+RUNTIME_PROMPTS['frozen_reference']
    if payload.get('applicability'):
        system += '\n'+RUNTIME_PROMPTS['frozen_applicability']
    if applicability_only:
        system = RUNTIME_PROMPTS['applicability_only']
    if category == 'perception':
        system += '\n'+spec['matching_system']
    if reference_only:
        system = RUNTIME_PROMPTS['reference_only']+' '+spec['matching_system']
    if not applicability_only and not reference_only:
        system += '\n'+RUNTIME_PROMPTS['evidence_format']
    if not applicability_only:
        system += '\n'+RUNTIME_PROMPTS['evidence_selection']
    return system


class Judge:
    def __init__(self, client, endpoint, model, spec, tokens, attempts, reasoning):
        self.client, self.endpoint, self.model, self.spec = client, endpoint, model, spec
        self.tokens, self.attempts, self.reasoning = tokens, attempts, reasoning

    def score(self, payload, definitions, applicability_only=False, reference_only=False):
        system = system_prompt(self.spec, payload, applicability_only, reference_only)
        passages = {side: [s.strip() for s in re.split(r'\n|(?<=[.;!?])\s+', payload[side+'_answer']) if s.strip()]
                    for side in ('reference', 'candidate') if side+'_answer' in payload}
        messages = [dict(role='system', content=system),
                    dict(role='user', content=json.dumps(dict(**payload, criteria=definitions, source_passages=passages), ensure_ascii=False))]
        budget = self.tokens
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
                choice = obj['choices'][0]
                content = choice['message']['content']
                if choice['finish_reason'] == 'length':
                    budget = min(budget*2, 32768)
                    raise ValueError('Judge output truncated')
                check(choice['finish_reason'] == 'stop', 'Judge response did not finish normally')
                data = json.loads(content)
                if not applicability_only:
                    data = assemble_quotes(data, passages)
                if applicability_only:
                    items = data['applicability']
                    check(len(items) == len(definitions) and {i['id'] for i in items} == set(definitions),
                          'Incorrect applicability criterion IDs')
                    for item in items:
                        check(type(item['applicable']) is bool and isinstance(item['reason'], str) and bool(item['reason'].strip()),
                              'Applicability needs a boolean and reference-based reason')
                        check(isinstance(item['required_points'], list) and
                              all(isinstance(p, str) and bool(p.strip()) for p in item['required_points']) and
                              bool(item['required_points']) == item['applicable'], 'Required points must agree with applicability')
                elif reference_only:
                    facts = data['reference_facts']
                    check(len({f['field'] for f in facts}) == len(facts), 'Duplicate reference fact fields')
                    validate_comparison([dict(f, candidate=None, matched=False, reason='Reference extraction') for f in facts], payload)
                else:
                    if 'reference_facts' in payload:
                        data = assemble_comparison(data, payload)
                    data = validate(data, payload, definitions)
                return dict(judge=data)
            except (ValueError, KeyError, TypeError, httpx.TransportError) as error:
                last_error = str(error) if isinstance(error, ValueError) else f'{type(error).__name__}: {error}'
                if isinstance(error, (ValueError, KeyError, TypeError)):
                    previous = [dict(role='assistant', content=content)] if content else []
                    if reference_only or 'reference_facts' in payload:
                        instruction = RUNTIME_PROMPTS['retry_reference']
                    elif applicability_only or last_error.startswith('Frozen applicability mismatch'):
                        instruction = RUNTIME_PROMPTS['retry_applicability']
                    else:
                        instruction = RUNTIME_PROMPTS['retry_format']
                    correction = dict(validation_error=last_error, instruction=instruction)
                    messages = messages[:2]+previous+[dict(role='user', content=json.dumps(correction, ensure_ascii=False))]
                if attempt+1 < self.attempts:
                    time.sleep(min(30, 2**(attempt+1)))
        raise RuntimeError(f'Judge failed after {self.attempts} attempts: {last_error}')
