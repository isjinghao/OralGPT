import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_answers import answer_patient, clear_truncated_suffix, complete


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self.chunks)


def chunk(text=None, finish=None, usage=None):
    return NS(choices=[NS(delta=NS(content=text), finish_reason=finish)], usage=usage)


def test_complete():
    usage = NS(prompt_tokens=100, completion_tokens=10, completion_tokens_details=None)
    stream = Stream([chunk('Hel'), chunk('lo'), chunk(finish='stop'), NS(choices=[], usage=usage)])
    client = Mock()
    client.chat.completions.create.return_value = stream
    result = complete(client, 'model', [], 4096, 1, '[test]', stream=True)
    assert result['answer'] == 'Hello'
    assert result['finish_reason'] == 'stop'
    assert result['usage']['prompt_tokens'] == 100
    assert stream.closed
    assert client.chat.completions.create.call_args.kwargs['stream_options'] == {'include_usage': True}

    client.chat.completions.create.return_value = Stream([chunk('partial')])
    try:
        complete(client, 'model', [], 4096, 1, '[test]', stream=True)
    except RuntimeError as error:
        assert 'without a finish reason' in str(error)
    else:
        raise AssertionError('An incomplete stream must not count as success')

    client.chat.completions.create.return_value = NS(choices=[NS(message=NS(content='Answer'), finish_reason='stop')], usage=usage)
    assert complete(client, 'model', [], 4096, 1, '[test]')['answer'] == 'Answer'
    assert 'stream' not in client.chat.completions.create.call_args.kwargs

    client.chat.completions.create.side_effect = [
        Stream([chunk(finish='length')]), Stream([chunk('Answer', finish='stop')])
    ]
    result = complete(client, 'model', [], 4096, 1, '[test]', stream=True)
    assert result['max_tokens_used'] == 8192

    error = openai.APITimeoutError(request=httpx.Request('POST', 'http://localhost/v1/chat/completions'))
    client.chat.completions.create.side_effect = [error, Stream([chunk('Recovered', finish='stop')])]
    with patch('run_answers.time.sleep'):
        assert complete(client, 'model', [], 4096, 2, '[test]', stream=True)['answer'] == 'Recovered'

    def interrupted():
        yield chunk('Discard this partial answer')
        raise httpx.ReadTimeout('stream stalled')

    broken_stream = Stream(interrupted())
    client.chat.completions.create.side_effect = [broken_stream, Stream([chunk('Recovered', finish='stop')])]
    with patch('run_answers.time.sleep'):
        assert complete(client, 'model', [], 4096, 2, '[test]', stream=True)['answer'] == 'Recovered'
    assert broken_stream.closed


def test_length_retries():
    client = Mock()
    client.chat.completions.create.side_effect = [
        Stream([chunk('Truncated', finish='length')]),
        Stream([chunk('Complete', finish='stop')]),
    ]
    result = complete(client, 'model', [], 8192, 1, '[test]', stream=True, max_token_limit=16384)
    assert result['answer'] == 'Complete' and result['max_tokens_used'] == 16384
    assert [call.kwargs['max_tokens'] for call in client.chat.completions.create.call_args_list] == [8192, 16384]
    client.chat.completions.create.side_effect = None
    client.chat.completions.create.return_value = NS(choices=[NS(message=NS(content='Still truncated'), finish_reason='length')], usage=None)
    try:
        complete(client, 'model', [], 16384, 1, '[test]', max_token_limit=16384)
    except RuntimeError as error:
        assert 'truncated answer' in str(error) and 'not cached' in str(error)
    else:
        raise AssertionError('An answer truncated at the ceiling must fail')


def test_suffix_resume():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cache_dir = root / 'group/name/cache/model'
        answers_path = root / 'group/name/evaluation/model/answers.json'
        cache_dir.mkdir(parents=True)
        answers_path.parent.mkdir(parents=True)
        pairs = [({'value': f'Question {i}'}, {'value': f'Gold {i}'}) for i in range(1, 4)]
        history = []
        for i, (question, _) in enumerate(pairs, 1):
            request = {'model': 'model', 'system': str(history), 'question': question['value'], 'images': [], 'image_max_pixels': 602112, 'max_tokens': 4096}
            cache = {'input': request, 'answer': f'Old {i}', 'finish_reason': 'length' if i == 2 else 'stop', 'usage': {}, 'max_tokens_used': 4096, 'seconds': 0}
            (cache_dir / f'answer_turn_{i:02d}.json').write_text(json.dumps(cache))
            history.append((question['value'], cache['answer']))
        answers_path.write_text(json.dumps([{'turn_index': i, 'finish_reason': 'length' if i == 2 else 'stop'} for i in range(1, 4)]))
        prefix = (cache_dir / 'answer_turn_01.json').read_bytes()
        assert clear_truncated_suffix(answers_path, cache_dir) == 2
        assert not answers_path.exists()
        assert (cache_dir / 'answer_turn_01.json').read_bytes() == prefix
        assert not (cache_dir / 'answer_turn_02.json').exists() and not (cache_dir / 'answer_turn_03.json').exists()
        assert clear_truncated_suffix(answers_path, cache_dir) is None

        args = NS(output_root=root, dataset=root / 'dataset.json', retry_length=True, force=False, answer_model='model', image_max_pixels=602112, max_tokens=4096, max_attempts=1, stream=False, max_token_limit=16384)
        client = Mock()
        client.chat.completions.create.side_effect = [
            NS(choices=[NS(message=NS(content='New 2'), finish_reason='stop')], usage=None),
            NS(choices=[NS(message=NS(content='New 3'), finish_reason='stop')], usage=None),
        ]
        with patch('run_answers.conversation_pairs', return_value=pairs), patch('run_answers.system_context', return_value='patient'), patch('run_answers.rollout_system_context', side_effect=lambda context, h: str(h)):
            assert answer_patient({'id': 'group__name', 'images': []}, args, client, 'model') == 'completed'
            assert answer_patient({'id': 'group__name', 'images': []}, args, client, 'model') == 'skipped'
        calls = client.chat.completions.create.call_args_list
        assert len(calls) == 2 and all(call.kwargs['max_tokens'] == 8192 for call in calls)
        assert 'New 2' in calls[1].kwargs['messages'][0]['content'] and 'Old 2' not in calls[1].kwargs['messages'][0]['content']
        assert (cache_dir / 'answer_turn_01.json').read_bytes() == prefix
        rows = json.loads(answers_path.read_text())
        assert [row['model_answer'] for row in rows] == ['Old 1', 'New 2', 'New 3']
        assert all(row['finish_reason'] == 'stop' for row in rows)


if __name__ == '__main__':
    test_complete()
    test_length_retries()
    test_suffix_resume()
    print('Streaming, retries, truncation ceiling, cache suffix cleanup and history-aware resumption checks passed.')
