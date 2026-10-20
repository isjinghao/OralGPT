#!/usr/bin/env python3
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import yaml

from oral_question_prompts import load_question_prompts, prompt_for_question
import run_sft_test_rollout as rollout


def fails(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected ValueError')


def main():
    bundle = load_question_prompts(Path(__file__).resolve().parents[1]/'prompts/oral_question_prompts.yaml')
    for task, spec in bundle['tasks'].items():
        for pattern in spec['match']:
            assert prompt_for_question(bundle, '<image> ' + pattern)[0] == task
    fails(lambda: prompt_for_question(bundle, 'unknown question'))
    fails(lambda: prompt_for_question(bundle, 'Frontal facial photographs and intraoral photographs'))
    _, frontal = prompt_for_question(bundle, 'Frontal facial photographs')
    assert bundle['blocks']['frontal_face'] in frontal
    assert bundle['blocks']['reference_table'] not in frontal and bundle['blocks']['pathway'] not in frontal
    _, ceph = prompt_for_question(bundle, 'cephalometric radiographs and measurements')
    assert bundle['blocks']['reference_table'] in ceph and bundle['blocks']['reference_exceptions'] in ceph

    calls = []
    class MockModel:
        def __init__(self, args):
            self.engine = SimpleNamespace(generating_args={})

        def chat(self, messages, system, images, **kwargs):
            calls.append(dict(messages=messages, system=system, images=images))
            assert 'GOLD_ONLY' not in system + messages[0]['content']
            return [SimpleNamespace(response_text=f'ANSWER_{len(calls)}', finish_reason='stop',
                                    response_length=4, prompt_length=20)]

    fake = ModuleType('llamafactory.chat')
    fake.ChatModel = MockModel
    with TemporaryDirectory() as folder, patch.dict(sys.modules, {'llamafactory.chat': fake}):
        root = Path(folder)
        fixture = deepcopy(bundle)
        fixture['common'] = 'COMMON_RULE'
        fixture['blocks'] = {key: f'RULE_{key}_ONLY' for key in fixture['blocks']}
        prompt_path = root/'prompts.yaml'
        prompt_path.write_text(yaml.safe_dump(fixture), encoding='utf-8')
        questions = ['Frontal facial photographs', 'intraoral photographs',
                     'cephalometric radiographs and measurements', 'What is the overall treatment pathway?']
        messages = [{'from': 'system', 'value': 'PATIENT_RECORD'}]
        for question in questions:
            messages.extend([{'from': 'human', 'value': question}, {'from': 'gpt', 'value': 'GOLD_ONLY'}])
        case = dict(id='group__test', conversations=messages, images=[])
        dataset = root/'cases.json'
        dataset.write_text(json.dumps([case]))
        base = ['rollout', '--dataset', str(dataset), '--model', str(root), '--no-adapter',
                '--output-root', str(root/'output'), '--output-model', 'routed',
                '--question-prompts', str(prompt_path)]

        def run(argv):
            with patch.object(sys, 'argv', argv), redirect_stdout(StringIO()):
                rollout.main()

        run(base)
        assert len(calls) == 4
        for index, call in enumerate(calls):
            task, _ = prompt_for_question(fixture, questions[index])
            selected = set(fixture['tasks'][task]['blocks'])
            assert call['messages'] == [{'role': 'user', 'content': questions[index]}]
            assert 'COMMON_RULE' in call['system'] and 'PATIENT_RECORD' in call['system']
            for key, text in fixture['blocks'].items():
                assert (text in call['system']) == (key in selected), (index, key)
            for previous in range(index):
                assert questions[previous] in call['system'] and f'ANSWER_{previous+1}' in call['system']
        from prepare_sft_v3 import augment_rows
        source_rows, history = [], []
        for index, question in enumerate(questions):
            answer = f'ANSWER_{index+1}' if index < 3 else 'CURRENT_TARGET_ONLY'
            source_rows.append(dict(id=f'case__turn_{index+1:02d}', source_patient_id='case',
                turn_index=index+1, history_qa_pairs=index, images=[], conversations=[
                    dict(from_='system', value=rollout.rollout_system_context('PATIENT_RECORD', history)),
                    dict(from_='human', value=question), dict(from_='gpt', value=answer)]))
            for message in source_rows[-1]['conversations']:
                message['from'] = message.pop('from_')
            history.append((question, answer))
        original_rows = deepcopy(source_rows)
        augmented = augment_rows(source_rows, fixture)
        assert source_rows == original_rows
        for index, row in enumerate(augmented):
            assert row['conversations'][0]['value'] == calls[index]['system']
            assert row['conversations'][1:] == source_rows[index]['conversations'][1:]
            assert row['images'] == source_rows[index]['images']
            assert 'CURRENT_TARGET_ONLY' not in row['conversations'][0]['value']
        state_path, _ = rollout.patient_paths(root/'output', 'routed', case['id'])
        state = json.loads(state_path.read_text(encoding='utf-8'))
        assert [t['question'] for t in state['turns']] == questions
        assert all('RULE_' not in json.dumps(turn) for turn in state['turns'])
        run(base)
        assert len(calls) == 4
        state['turns'].pop()
        state['status'] = 'in_progress'
        state_path.write_text(json.dumps(state))
        run(base)
        assert len(calls) == 5 and 'ANSWER_3' in calls[-1]['system']
        assert 'RULE_reference_table_ONLY' not in calls[-1]['system']
        fixture['common'] = 'CHANGED_RULE'
        prompt_path.write_text(yaml.safe_dump(fixture), encoding='utf-8')
        fails(lambda: run(base))
        fails(lambda: run(base[:-2]))
        assert len(calls) == 5
        baseline = base[:-2]
        baseline[baseline.index('--output-model')+1] = 'baseline'
        run(baseline)
        assert calls[5]['system'] == 'PATIENT_RECORD'
        assert calls[6]['system'] == rollout.rollout_system_context('PATIENT_RECORD', [(questions[0], 'ANSWER_6')])
    print('Passed: task routing, current-only rules, original Q/A history, baseline unchanged, resume and prompt-change rejection.')


if __name__ == '__main__':
    main()
