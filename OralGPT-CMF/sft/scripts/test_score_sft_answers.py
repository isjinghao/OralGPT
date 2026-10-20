"""Offline regression tests; no API credentials or medical data required."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import yaml

from score_sft_answers import (Judge, JudgeFailure, aggregate, api_endpoint, calculate, comparison_counts,
                               fact_metrics, numeric_summary, prepare, prf, quotes, read, run_job,
                               run_lock, validate)

SPEC = yaml.safe_load((Path(__file__).resolve().parents[1]/'prompts/oral_scoring.yaml').read_text(encoding='utf-8'))
DEFS = SPEC['categories']['perception']['criteria']
PAYLOAD = dict(category='perception', question='What do the images show?',
               reference_answer='Tooth 26 has a filling. No gummy smile.',
               candidate_answer='Tooth 27 has a filling. No gingival display on smiling.')


def fact(text, values=None, unit=''):
    return dict(text=text, quotes=[text], values=[] if values is None else values, unit=unit)


def response(payload=PAYLOAD, definitions=DEFS):
    comparison = [dict(field='gummy smile', reference=fact('No gummy smile.'),
                       candidate=fact('No gingival display on smiling.'), matched=True, scored=True, reason='同义'),
                  dict(field='tooth 26 filling', reference=fact('Tooth 26 has a filling.'),
                       candidate=None, matched=False, scored=True, reason='牙位未匹配'),
                  dict(field='tooth 27 filling', reference=None, candidate=fact('Tooth 27 has a filling.'),
                       matched=False, scored=True, reason='不同牙位')]
    return dict(criteria=[dict(id=k, applicable=True, awarded=v['max_score']/2, reason='部分对应',
                               reference_quotes=[payload['reference_answer']], candidate_quotes=[payload['candidate_answer']])
                          for k, v in definitions.items()], main_errors=[], unscorable_reason='', reference_unmatched=[],
                needs_review=False, review_reason='', fact_comparison=comparison if payload['category']=='perception' else None)


class Tests(unittest.TestCase):
    def test_endpoint_consent(self):
        self.assertEqual(api_endpoint('https://example.invalid/v1/'), 'https://example.invalid/v1/chat/completions')
        with self.assertRaises(ValueError):
            api_endpoint('http://example.invalid/v1')
        self.assertEqual(api_endpoint('http://example.invalid/v1', True), 'http://example.invalid/v1/chat/completions')
        for url in ['http://127.0.0.1.evil/v1', 'https://user:secret@example.invalid/v1', 'https://example.invalid/v1?key=secret']:
            with self.assertRaises(ValueError):
                api_endpoint(url)

    def test_model_matches_and_counts(self):
        data = validate(response(), PAYLOAD, DEFS)
        metrics = calculate(data, PAYLOAD, DEFS)
        self.assertEqual(metrics['rubric_percent'], 50)
        self.assertEqual(metrics['facts']['f1'], 0.5)
        self.assertEqual(metrics['facts']['tp'], 1)
        self.assertEqual(set(metrics['facts']), {'tp', 'fn', 'fp_reference_unmatched', 'precision', 'recall',
                                                  'f1', 'jaccard', 'reference_count', 'candidate_count'})
        self.assertEqual(comparison_counts(data['fact_comparison']),
                         dict(reference_total=2, candidate_total=2, tp=1, fn=1, fp_reference_unmatched=1))
        repeated = response()
        repeated['fact_comparison'].append(deepcopy(repeated['fact_comparison'][0]))
        metrics = calculate(validate(repeated, PAYLOAD, DEFS), PAYLOAD, DEFS)
        self.assertEqual(metrics['facts']['tp'], 2)
        self.assertFalse(metrics['needs_review'])
        reviewed = response()
        reviewed.update(needs_review=True, review_reason='原文歧义，匹配需复核')
        metrics = calculate(validate(reviewed, PAYLOAD, DEFS), PAYLOAD, DEFS)
        self.assertTrue(metrics['needs_review'])
        self.assertIn('facts', metrics)
        self.assertIsNone(prf(0, 0, 0)['f1'])
        self.assertEqual(prf(0, 0, 2)['recall'], 0)

    def test_quotes_and_rubric(self):
        quotes(['Overjet -4 mm'])
        quotes([])
        for invalid in ['not an array', [123], None]:
            with self.assertRaises(ValueError):
                quotes(invalid)
        for modification in ['score', 'nan', 'quote', 'duplicate_id', 'na_score', 'null_comparison']:
            bad = response()
            if modification == 'score':
                bad['criteria'][0]['awarded'] = 101
            elif modification == 'nan':
                bad['criteria'][0]['awarded'] = float('nan')
            elif modification == 'quote':
                bad['fact_comparison'][0]['reference']['quotes'] = 123
            elif modification == 'duplicate_id':
                bad['criteria'][1]['id'] = bad['criteria'][0]['id']
            elif modification == 'na_score':
                bad['criteria'][0].update(applicable=False, awarded=0)
            else:
                bad['fact_comparison'] = None
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                validate(bad, PAYLOAD, DEFS)
        good = response()
        good['criteria'][3].update(applicable=False, awarded=None)
        self.assertEqual(calculate(validate(good, PAYLOAD, DEFS), PAYLOAD, DEFS)['rubric_percent'], 50)
        metrics = calculate(good, PAYLOAD, DEFS)
        self.assertEqual(metrics['rubric_percent'], 50)
        summary = aggregate([dict(status='completed', category='perception', judge=good, metrics=metrics)], {'perception': 1})
        output = json.dumps(dict(judge=good, metrics=metrics, summary=summary))
        self.assertNotIn('major_error', output)
        for field in ['scope', 'version', 'source_digest', 'fact_method', 'usage_note', 'metrics_note', 'metrics_code_digest']:
            self.assertNotIn('"'+field+'":', output)
        self.assertEqual(summary['question_macro_percent'], 50)
        self.assertEqual(summary['categories']['perception']['fact_micro']['f1'], 0.5)

    def test_numeric_and_range(self):
        payload = dict(PAYLOAD, reference_answer='Overjet −4 mm. Ratio 56–57%.', candidate_answer='Overjet 2 mm. Ratio 56.7%.')
        data = response(payload)
        comparison = [dict(field='overjet', reference=fact('Overjet −4 mm.', [-4], 'mm'),
                           candidate=fact('Overjet 2 mm.', [2], 'mm'), matched=True, scored=True, reason='模拟裁判误判'),
                      dict(field='ratio', reference=fact('Ratio 56–57%.', [56, 57], 'percent'),
                           candidate=fact('Ratio 56.7%.', [56.7], 'percent'), matched=True, scored=True, reason='模拟区间误判')]
        data['fact_comparison'] = comparison
        result = calculate(validate(data, payload, DEFS), payload, DEFS)
        metrics = result['facts']
        self.assertEqual(metrics['f1'], 1)
        self.assertNotIn('numeric_match_overrides', metrics)
        self.assertFalse(result['needs_review'])
        self.assertEqual(result['rubric_percent'], 50)
        summary = numeric_summary([comparison])
        self.assertEqual(summary['by_unit']['mm']['mae'], 6)
        self.assertEqual(summary['by_unit']['mm']['sign_accuracy'], 0)
        self.assertEqual(summary['comparison_coverage'], 0.5)
        self.assertIsNone(summary['by_unit']['percent']['mae'])
        comparison[0]['candidate']['unit'] = 'degrees'
        self.assertEqual(numeric_summary([comparison])['unit_accuracy'], 0)
        for values in [[float('nan')], [True], ['2'], [1, 2, 3]]:
            comparison[0]['candidate']['values'] = values
            with self.assertRaises(ValueError):
                validate(data, payload, DEFS)
        row = comparison[1]
        for values in [[56], [57], [56.5], [56.7], [56, 58], [57, 56]]:
            row['candidate']['values'] = values
            self.assertEqual(comparison_counts([row])['tp'], 1)
        row['candidate']['values'] = [56, 57]
        self.assertEqual(comparison_counts([row])['tp'], 1)
        row['reference']['values'], row['candidate']['values'] = [82.0], [82]
        self.assertEqual(comparison_counts([row])['tp'], 1)
        row['candidate'] = None
        row['matched'] = False
        self.assertEqual(fact_metrics([row])['recall'], 0)
        self.assertEqual(numeric_summary([[row]])['noncomparable_fields'], 1)

    def test_numeric_unit_groups_do_not_overwrite(self):
        rows = [dict(field='ratio '+str(i), reference=fact('Ratio 1.5', [1.5], unit),
                     candidate=fact('Ratio 1.5', [1.5], unit), matched=True, scored=True, reason='模型配对')
                for i, unit in enumerate(['', 'unspecified', 'ratio'])]
        summary = numeric_summary([rows])
        self.assertEqual(set(summary['by_unit']), {'', 'unspecified', 'ratio'})
        for key in ('reference_fields', 'comparable_fields'):
            self.assertEqual(sum(v[key] for v in summary['by_unit'].values()), summary[key])

    def test_no_secondary_content_review(self):
        data = response()
        data['criteria'][0]['reference_quotes'] = ['model-provided quotation']
        data['fact_comparison'][0]['candidate']['quotes'] = []
        original = deepcopy(data)
        self.assertEqual(validate(data, PAYLOAD, DEFS), original)
        self.assertFalse(calculate(data, PAYLOAD, DEFS)['needs_review'])
        data['fact_comparison'][0]['reference'] = None
        with self.assertRaisesRegex(ValueError, 'Match needs both sides'):
            validate(data, PAYLOAD, DEFS)

    def test_single_request_and_retry(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body['model'], 'gpt-5-mini')
            self.assertNotIn('temperature', body)
            self.assertEqual(set(json.loads(body['messages'][1]['content'])), set(PAYLOAD)|{'criteria'})
            return httpx.Response(200, json=dict(model='gpt-5-mini', usage={'total_tokens': 7}, choices=[dict(
                finish_reason='stop', message=dict(content=json.dumps(response())))]))
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = Judge(client, 'https://example.invalid', 'gpt-5-mini', SPEC, 1024, 2, 'low').score(PAYLOAD, DEFS)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result['attempts'], 1)
        self.assertNotIn('stages', result)
        calls = []
        def retry(request):
            body = json.loads(request.content)
            calls.append(body)
            data = response()
            if len(calls) == 1:
                data['fact_comparison'][0]['matched'] = 'true'
            else:
                self.assertEqual(body['messages'][2]['role'], 'assistant')
                self.assertIn('matched must be boolean', body['messages'][-1]['content'])
            return httpx.Response(200, json=dict(usage={'total_tokens': 7}, choices=[dict(
                finish_reason='stop', message=dict(content=json.dumps(data)))]))
        with httpx.Client(transport=httpx.MockTransport(retry)) as client, patch('score_sft_answers.time.sleep'):
            result = Judge(client, 'https://example.invalid', 'gpt-5-mini', SPEC, 1024, 2, 'low').score(PAYLOAD, DEFS)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['usage']['total_tokens'], 14)
        self.assertEqual(set(result), {'judge', 'usage', 'attempts'})

    def test_api_failure_and_truncation(self):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text='secret'))) as client:
            with self.assertRaisesRegex(RuntimeError, '^HTTP 401;'):
                Judge(client, 'https://example.invalid', 'gpt-5-mini', SPEC, 1024, 3, 'low').score(PAYLOAD, DEFS)
        budgets = []
        def truncated(request):
            budgets.append(json.loads(request.content)['max_completion_tokens'])
            return httpx.Response(200, json=dict(choices=[dict(finish_reason='length', message=dict(content=''))]))
        with httpx.Client(transport=httpx.MockTransport(truncated)) as client, patch('score_sft_answers.time.sleep'):
            with self.assertRaisesRegex(JudgeFailure, 'truncated'):
                Judge(client, 'https://example.invalid', 'gpt-5-mini', SPEC, 1024, 2, 'low').score(PAYLOAD, DEFS)
        self.assertEqual(budgets, [1024, 2048])

    def test_diagnosis_percent(self):
        defs = SPEC['categories']['diagnosis']['criteria']
        self.assertEqual([d['max_score'] for d in defs.values()], [60, 20, 5, 15])
        payload = dict(PAYLOAD, category='diagnosis')
        data = validate(response(payload, defs), payload, defs)
        self.assertEqual(calculate(data, payload, defs)['rubric_percent'], 50)
        data['criteria'][2].update(applicable=False, awarded=None)
        self.assertEqual(calculate(validate(data, payload, defs), payload, defs)['rubric_percent'], 50)
        self.assertNotIn('facts', calculate(data, payload, defs))

    def test_cache_and_parallelism(self):
        class MockJudge:
            def __init__(self):
                self.count = self.active = self.peak = 0
                self.lock = threading.Lock()
            def score(self, payload, defs):
                with self.lock:
                    self.count += 1
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.03)
                with self.lock:
                    self.active -= 1
                return dict(judge=validate(response(), payload, defs), usage={})
        with TemporaryDirectory() as folder:
            judge = MockJudge()
            jobs = [dict(path=Path(folder)/f'patient{i%2}'/f'turn_{i}.json', payload=PAYLOAD,
                         task_id=str(i), task='FACE') for i in range(6)]
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(lambda j: run_job(j, judge, DEFS)[0], jobs))
            self.assertEqual(judge.peak, 3)
            for job in jobs:
                self.assertTrue(run_job(job, judge, DEFS)[1])
                self.assertNotIn('input_hash', read(job['path']))
            self.assertEqual(judge.count, 6)
            with self.assertRaises(ValueError):
                run_job(dict(jobs[0], payload=dict(PAYLOAD, candidate_answer='changed')), judge, DEFS)
            summary = aggregate(results, {'perception': 6})
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['categories']['perception']['fact_micro']['f1'], 0.5)
            self.assertEqual(summary['categories']['perception']['fact_question_coverage'], 1)
            self.assertFalse(aggregate(results, {'perception': 7})['complete'])
            with run_lock(Path(folder)/'run.lock'):
                with self.assertRaises(OSError):
                    with run_lock(Path(folder)/'run.lock'):
                        pass

    def test_selection_and_redaction(self):
        routing = dict(tasks={'FACE': dict(match=['facial photographs'], instruction='none', blocks=[])}, common='x', blocks={})
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'group'/'NAME'/'evaluation'/'v3'/'answers.json'
            path.parent.mkdir(parents=True)
            row = dict(task_id='group__NAME__turn_01', turn_index=1, question='<image> facial photographs NAME',
                       gold_answer='Reference NAME', model_answer='Answer NAME', images=['private.jpg'], history='PRIVATE')
            path.write_text(json.dumps([row]))
            args = SimpleNamespace(answer_model='v3', judge_model='gpt-5-mini', run_name='test', output_root=root,
                                   case_id=[], limit=1, max_tokens=1024, reasoning_effort='low')
            config, jobs = prepare(args, SPEC, routing, 'https://example.invalid')
            self.assertEqual(len(jobs), 1)
            self.assertEqual(set(config), {'judge_model', 'answer_model', 'spec', 'endpoint', 'tokens', 'reasoning_effort', 'cases'})
            self.assertEqual(config['cases'], ['group__NAME'])
            self.assertNotIn('input_hash', jobs[0])
            transmitted = json.dumps(jobs[0]['payload'])
            for text in ['NAME', 'PRIVATE', 'private.jpg']:
                self.assertNotIn(text, transmitted)
            self.assertEqual(read(path), [row])
            args.case_id = ['missing']
            with self.assertRaises(ValueError):
                prepare(args, SPEC, routing, 'https://example.invalid')

    def test_dentist_prompt_contract(self):
        zh = yaml.safe_load((Path(__file__).resolve().parents[1]/'prompts/oral_scoring_zh.yaml').read_text(encoding='utf-8'))
        self.assertEqual(SPEC, zh)
        for category, weights in [('perception', [40, 25, 25, 10]), ('diagnosis', [60, 20, 5, 15]),
                                  ('treatment', [50, 25, 15, 10])]:
            self.assertEqual([v['max_score'] for v in SPEC['categories'][category]['criteria'].values()], weights)
        text = SPEC['judge_system']+SPEC['matching_system']
        for old_rule in ['0.5 mm', '1 degree', '1 percentage point', '可接受最多', '至少首要维度必须适用']:
            self.assertNotIn(old_rule, text)
        for rule in ['不设固定', '同一错误只在主要所属维度扣分', 'TMJ', 'PAOO', '待核实', '全部维度无法适用']:
            self.assertIn(rule, text)
        decoder = json.JSONDecoder()
        example, end = decoder.raw_decode(SPEC['output_contract'])
        self.assertEqual(set(example), set(response()))
        rows, _ = decoder.raw_decode(SPEC['output_contract'][SPEC['output_contract'].index('\n[', end)+1:])
        data = response()
        data['fact_comparison'] = rows
        validate(data, PAYLOAD, DEFS)

    def test_all_na_and_score_coverage(self):
        data = response()
        for item in data['criteria']:
            item.update(applicable=False, awarded=None, reason='本题无可适用要求')
        data['fact_comparison'] = []
        with self.assertRaisesRegex(ValueError, 'all-N/A'):
            validate(data, PAYLOAD, DEFS)
        data['unscorable_reason'] = '本题与参考均无可适用评价内容'
        metrics = calculate(validate(data, PAYLOAD, DEFS), PAYLOAD, DEFS)
        self.assertIsNone(metrics['rubric_percent'])
        self.assertIsNone(metrics['facts']['f1'])
        missing = dict(status='completed', category='perception', judge=data, metrics=metrics)
        report = aggregate([missing], {'perception': 1})
        self.assertTrue(report['complete'])
        self.assertEqual(report['completed_questions'], 1)
        self.assertEqual(report['scored_questions'], 0)
        self.assertEqual(report['unscorable_questions'], 1)
        self.assertEqual(report['categories_with_scores'], 0)
        self.assertEqual(report['categories']['perception']['fact_question_coverage'], 0)
        self.assertIsNone(report['question_macro_percent'])
        self.assertIsNone(report['category_macro_percent'])
        good = response()
        report = aggregate([missing, dict(status='completed', category='perception', judge=good,
                                        metrics=calculate(good, PAYLOAD, DEFS))], {'perception': 2})
        self.assertEqual(report['question_macro_percent'], 50)
        self.assertEqual(report['scored_questions'], 1)
        good['criteria'][0].update(applicable=False, awarded=None)
        self.assertEqual(calculate(validate(good, PAYLOAD, DEFS), PAYLOAD, DEFS)['rubric_percent'], 50)

    def test_rounding_and_equal_category_weights(self):
        records = []
        for category, percent in [('perception', 100), ('diagnosis', 50), ('treatment', 0), ('treatment', 0)]:
            defs = SPEC['categories'][category]['criteria']
            payload = dict(PAYLOAD, category=category)
            data = response(payload, defs)
            for item in data['criteria']:
                item['awarded'] = defs[item['id']]['max_score']*percent/100
            records.append(dict(status='completed', category=category, judge=data,
                                metrics=calculate(validate(data, payload, defs), payload, defs)))
        summary = aggregate(records, dict(perception=1, diagnosis=1, treatment=2))
        self.assertEqual(summary['category_macro_percent'], 50)
        self.assertEqual(summary['question_macro_percent'], 37.5)
        defs = SPEC['categories']['diagnosis']['criteria']
        payload = dict(PAYLOAD, category='diagnosis')
        data = response(payload, defs)
        data['criteria'][2].update(applicable=False, awarded=None)
        data['criteria'][0]['awarded'] = 31
        metrics = calculate(validate(data, payload, defs), payload, defs)
        self.assertEqual(metrics['rubric_applicable_max'], 95)
        self.assertEqual(metrics['rubric_percent'], 51.1)

    def test_main_errors_do_not_deduct_again(self):
        data = response()
        before = calculate(data, PAYLOAD, DEFS)
        data['main_errors'] = [dict(criterion_id='entity_binding', reason='牙位绑定错误，仅此维度扣分')]
        self.assertEqual(calculate(validate(data, PAYLOAD, DEFS), PAYLOAD, DEFS), before)
        data['criteria'][1].update(applicable=False, awarded=None)
        with self.assertRaisesRegex(ValueError, 'applicable criterion'):
            validate(data, PAYLOAD, DEFS)

    def test_optional_facts_and_auxiliary_numbers(self):
        data = response()
        data['fact_comparison'] = [data['fact_comparison'][0],
            dict(field='optional angle', reference=fact('Angle 80', [80], 'degrees'),
                 candidate=fact('Angle 82', [82], 'degrees'), matched=False, scored=False,
                 reason='非必需附加数值，不改变主要判断，只保留辅助统计'),
            dict(field='optional measurement', reference=fact('Distance 2', [2], 'mm'),
                 candidate=None, matched=False, scored=False, reason='未要求具体数值，不因省略扣分'),
            dict(field='additional claim', reference=None, candidate=fact('Unverified addition'),
                 matched=False, scored=False, reason='参考未记载且不冲突，待核实')]
        data['reference_unmatched'] = [dict(candidate_quote='Unverified addition', reason='待核实')]
        facts = calculate(validate(data, PAYLOAD, DEFS), PAYLOAD, DEFS)['facts']
        self.assertEqual((facts['tp'], facts['fn'], facts['fp_reference_unmatched']), (1, 0, 0))
        self.assertNotIn('excluded', facts)
        self.assertEqual(sum(not row['scored'] for row in data['fact_comparison']), 3)
        numeric = numeric_summary([data['fact_comparison']])
        self.assertEqual(numeric['reference_fields'], 2)
        self.assertEqual(numeric['by_unit']['degrees']['mae'], 2)
        data['fact_comparison'][1]['scored'] = True
        self.assertEqual(calculate(validate(data, PAYLOAD, DEFS), PAYLOAD, DEFS)['facts']['f1'], 0.5)
        data['fact_comparison'][1]['scored'] = 'true'
        with self.assertRaisesRegex(ValueError, 'scored must be boolean'):
            validate(data, PAYLOAD, DEFS)

    def test_unknown_units_and_qualitative_match(self):
        rows = [dict(field='measurement '+str(i), reference=fact('Reference', [2], unit),
                     candidate=fact('Candidate', [2], unit), matched=True, scored=True, reason='模型判断一致')
                for i, unit in enumerate(['', 'unspecified', 'mm'])]
        numeric = numeric_summary([rows])
        self.assertEqual(numeric['comparable_fields'], 1)
        self.assertEqual(numeric['unit_denominator'], 1)
        rows[2]['candidate'] = fact('Qualitative only')
        self.assertEqual(comparison_counts(rows)['tp'], 3)
        self.assertEqual(numeric_summary([rows])['comparable_fields'], 0)


if __name__ == '__main__':
    unittest.main()
