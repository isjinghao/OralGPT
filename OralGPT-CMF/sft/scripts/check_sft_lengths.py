#!/usr/bin/env python3
"""Use the real SFT encoder with a high audit ceiling to detect training truncation."""
import argparse
import json
import math
from pathlib import Path
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--llama-root', type=Path, default=Path('/root/autodl-tmp/LLaMA-Factory-qwen35'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_root/'src'))
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    config = yaml.safe_load(args.config.read_text())
    model_args, data_args, training_args, _, _ = get_train_args(config)
    assert not data_args.packing and not data_args.streaming and not data_args.val_size
    assert not data_args.max_samples and not data_args.tokenized_path
    cutoff = data_args.cutoff_len
    data_args.cutoff_len = 1_000_000
    names = list(dict.fromkeys((data_args.dataset or []) + (data_args.eval_dataset or [])))
    assert names
    data_args.eval_on_each_dataset = True
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module['tokenizer'], data_args)
    data = Path(config['dataset_dir'])
    registry = json.loads((data/'dataset_info.json').read_text(encoding='utf-8'))
    report = dict(config=str(args.config), cutoff_len=cutoff, image_max_pixels=config['image_max_pixels'],
                  method='Actual LLaMA-Factory SFT encoder; audit ceiling 1000000; full source + target tokens.',
                  order_assumption='Single local non-streaming dataset, no sampling, packing or split; source order retained by map.',
                  splits={}, overflow=[])
    for split in names:
        data_args.dataset = [split]
        data_args.eval_dataset = None
        module = get_dataset(template, model_args, data_args, training_args, stage='sft', **tokenizer_module)
        dataset = module['train_dataset']
        source = json.loads((data/registry[split]['file_name']).read_text(encoding='utf-8'))
        assert len(dataset) == len(source), f'Dropped samples: {split}'
        lengths = []
        for index, item in enumerate(dataset):
            length = len(item['input_ids'])
            assert length < data_args.cutoff_len and len(item['labels']) == length
            assert any(label != -100 for label in item['labels'])
            lengths.append(length)
            if length > cutoff:
                report['overflow'].append(dict(split=split, index=index, id=source[index]['id'], tokens=length))
        ordered = sorted(lengths)
        report['splits'][split] = dict(samples=len(lengths), max_tokens=max(lengths),
                                      p95_tokens=ordered[math.ceil(len(ordered)*0.95)-1],
                                      max_id=source[lengths.index(max(lengths))]['id'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)
    if report['overflow']:
        raise SystemExit('Samples exceed the unchanged training cutoff; do not start training.')


if __name__ == '__main__':
    main()
