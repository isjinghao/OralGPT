#!/usr/bin/env python3
"""Add current-question expert rules to existing v1 samples without changing targets or splits."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.question_prompts import load_question_prompts, prompt_for_question, question_system
from common.conversation import rollout_system_context


def augment_rows(rows, bundle):
    contexts, histories, result = {}, {}, []
    assert len({row['id'] for row in rows}) == len(rows)
    for row in rows:
        patient = row['source_patient_id']
        messages = row['conversations']
        assert [m['from'] for m in messages] == ['system', 'human', 'gpt'], row['id']
        if patient not in histories:
            assert row['turn_index'] == 1 and row['history_qa_pairs'] == 0, row['id']
            histories[patient] = []
            contexts[patient] = messages[0]['value']
        history = histories[patient]
        assert row['turn_index'] == len(history)+1 and row['history_qa_pairs'] == len(history), row['id']
        assert messages[0]['value'] == rollout_system_context(contexts[patient], history), row['id']
        assert messages[1]['value'].count('<image>') == len(row['images']), row['id']
        new = deepcopy(row)
        context = rollout_system_context(contexts[patient], history, bundle['history_heading'])
        new['conversations'][0]['value'] = question_system(bundle, messages[1]['value'], context)
        result.append(new)
        history.append((messages[1]['value'], messages[2]['value']))
    return result


def prepare(source, output, prompt_path):
    bundle = load_question_prompts(prompt_path)
    info = json.loads((source/'dataset_info.json').read_text(encoding='utf-8'))
    transformed, patients, statistics = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        name = f'oralcmf_sft_{split}'
        rows = json.loads((source/info[name]['file_name']).read_text(encoding='utf-8'))
        transformed[split] = augment_rows(rows, bundle)
        patients[split] = {row['source_patient_id'] for row in rows}
        statistics[split] = dict(patients=len(patients[split]), samples=len(rows),
                                tasks=dict(Counter(prompt_for_question(bundle, row['conversations'][1]['value'])[0]
                                                   for row in rows)))
        for row in rows:
            for image in row['images']:
                assert (source/image).is_file(), image
    assert not (patients['train'] & patients['validation'] or patients['train'] & patients['test']
                or patients['validation'] & patients['test'])
    rollout_path = source/'oralcmf_sft_test_rollout.json'
    cases = json.loads(rollout_path.read_text(encoding='utf-8'))
    assert {case['id'] for case in cases} == patients['test']
    test_rows = {row['id']: row for row in transformed['test']}
    for case in cases:
        context, *messages = case['conversations']
        history, image_cursor = [], 0
        for index in range(0, len(messages), 2):
            question, answer = messages[index:index+2]
            row = test_rows[f"{case['id']}__turn_{index//2+1:02d}"]
            assert row['conversations'][1:] == [question, answer]
            count = question['value'].count('<image>')
            assert row['images'] == case['images'][image_cursor:image_cursor+count]
            image_cursor += count
            expected = question_system(bundle, question['value'],
                rollout_system_context(context['value'], history, bundle['history_heading']))
            assert row['conversations'][0]['value'] == expected
            history.append((question['value'], answer['value']))
        assert image_cursor == len(case['images'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'media').symlink_to((source/'media').resolve(), target_is_directory=True)
    for split, rows in transformed.items():
        (output/f'oralcmf_sft_{split}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    selected = {name: spec for name, spec in info.items() if name in {f'oralcmf_sft_{s}' for s in transformed}}
    (output/'dataset_info.json').write_text(json.dumps(selected,indent=2)+'\n')
    shutil.copyfile(prompt_path, output/'question_prompts.yaml')
    shutil.copyfile(rollout_path, output/rollout_path.name)
    manifest = dict(source=str(source.resolve()), prompt_snapshot='question_prompts.yaml',
                    splits=statistics, unchanged='IDs, patient splits, question/answer text, image paths/order and history Q/A',
                    changed='System: common + current question rules, with existing routed-prompt history heading',
                    rollout=dict(file=rollout_path.name, required_args=['--question-prompts',str(output/'question_prompts.yaml')],
                                 history='Actual model predictions, never reference answers',
                                 note='Unchanged case records; rules are injected dynamically per question, not accumulated.'),
                    checks=dict(original_rows_preserved=True, patient_splits_disjoint=True, test_prompt_alignment=True))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--question-prompts', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.question_prompts)


if __name__ == '__main__':
    main()
