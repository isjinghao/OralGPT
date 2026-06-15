import os
import os.path as osp
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from .utils import build_judge, DEBUG_MESSAGE
from .image_base import ImageBaseDataset
from ..utils import track_progress_rich
from ..smp import *


# Base class with common functionality
class MMOralBase(ImageBaseDataset):
    
    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

# Class for open-ended questions
class MMOral_OPG_OPEN(MMOralBase):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMOral_OPG_OPEN': 'https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench/resolve/main/MMOral-OPG-Bench-Open-Ended.tsv'
    }
    
    DATASET_MD5 = {
        'MMOral_OPG_OPEN': 'd328b1b527ef7467b328d8b35d5f8155'
    }
    
    def build_prompt(self, line):
        tgt_path = self.dump_image(line)
        question = line['question']
        prompt = f'Question: {question}\nPlease provide a detailed and accurate answer to the question.'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils.mmoral import MMOral_opg_auxeval, MMOral_opg_acc
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=16384, **judge_kwargs)
            assert model.working(), ('MMOral-Open-ended evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMOral_opg_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMOral_opg_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score

# Class for closed-ended questions
class MMOral_OPG_CLOSED(MMOralBase):
    TYPE = 'MCQ'
    OPTION_COLUMNS = ['option1', 'option2', 'option3', 'option4']
    
    DATASET_URL = {
        'MMOral_OPG_CLOSED': 'https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench/resolve/main/MMOral-OPG-Bench-Closed-Ended.tsv'
    }
    
    DATASET_MD5 = {
        'MMOral_OPG_CLOSED': 'd70060e427f7d68a7cad1edb513b3a5f'
    }

    def post_build(self, dataset):
        for col in self.OPTION_COLUMNS:
            if col in self.data:
                self.data[col] = self.data[col].fillna('None')
    
    def build_prompt(self, line):
        tgt_path = self.dump_image(line)
        question = line['question']
        
        options_prompt = 'Options:\n'
        for i in [['A', '1'], ['B', '2'], ['C', '3'], ['D', '4']]:
            option_value = str(line[f'option{i[1]}'])
            options_prompt += f"{i[0]}. {option_value}\n"
            
        prompt = (f'Question: {question}\n' + options_prompt
                + 'Please answer the above multiple-choice question by selecting the single correct option (A, B, C, or D). '
                + 'If the provided information is insufficient to determine a clear answer, please choose the most likely '
                + 'correct option based on the available data and your judgment.')

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils.mmoral import get_single_choice_prediction
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        detail_result_file = eval_file.replace(f'.{suffix}', '_detailed_acc.csv')

        if not osp.exists(result_file) or not osp.exists(detail_result_file):
            data = load(eval_file)
            assert 'answer' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]

            tot = defaultdict(lambda: 0)
            score = defaultdict(lambda: 0)
            
            main_category_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec']
            categories = set()
            subcategories = set()
            
            for _, line in data.iterrows():
                category = line.get('category', 'unknown')
                categories.add(category)
                subcategory = category.replace(',', '_')
                subcategories.add(subcategory)
                
                for main_cat in main_category_list:
                    if main_cat in category:
                        tot[main_cat] += 1
                
                tot[category] += 1
                tot[subcategory] += 1
                tot['Overall'] += 1

            for i in tqdm(data.iterrows()):
                line = i[1]
                category = line.get('category', 'unknown')
                subcategory = category.replace(',', '_')
                
                index2ans = {
                    'A': line['option1'],
                    'B': line['option2'],
                    'C': line['option3'],
                    'D': line['option4']
                }
                
                fact_option = get_single_choice_prediction(line['prediction'], ['A', 'B', 'C', 'D'], index2ans)
                
                if fact_option == line['answer']:
                    for main_cat in main_category_list:
                        if main_cat in category:
                            score[main_cat] += 1
                    
                    score[category] += 1
                    score[subcategory] += 1
                    score['Overall'] += 1
            
            main_result = defaultdict(list)
            main_category_list.append('Overall')
            for cat in main_category_list:
                main_result['Category'].append(cat)
                main_result['tot'].append(tot[cat])
                main_result['acc'].append(score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0)
            
            detailed_categories = list(categories) + ['Overall']
            detailed_result = defaultdict(list)
            for cat in detailed_categories:
                detailed_result['Category'].append(cat)
                detailed_result['tot'].append(tot[cat])
                detailed_result['acc'].append(score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0)
            
            main_df = pd.DataFrame(main_result)
            detailed_df = pd.DataFrame(detailed_result)
            
            main_df = main_df.sort_values('Category')
            detailed_df = detailed_df.sort_values('Category')
            
            dump(main_df, result_file)
            dump(detailed_df, detail_result_file)

        result = pd.read_csv(result_file)
        return result

class MMOral_OMNI(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMOral_OMNI':
        'https://huggingface.co/datasets/OralGPT/MMOral-Omni-Bench/resolve/main/MMOral-Omni-Bench.tsv',
    }
    DATASET_MD5 = {
        'MMOral_OMNI': '139e90f132f02e2a87d60eff1c24254a',
    }

    def __init__(self, dataset='MMOral_OMNI', skip_noimg=False, categories=None):
        """
        Args:
            dataset: Dataset name.
            skip_noimg: Whether to skip samples without image.
            categories: Optional list of category keywords (e.g. ["PA", "CE", "PI"]).
                If provided, only samples whose `category` field contains any of these
                keywords are used. Main category keys include: II_Loc, II_Dx-I, II_Dx-R,
                PA, CE, PI, TP, IV, etc. None or empty list = use all categories.
        """
        if dataset != 'MMOral_OMNI' or dataset != 'MMOral_OMNI_Mini':
            import warnings
            warnings.warn(
                'To evaluate on MMOral-OMNI, we would suggest `MMOral_OMNI` for the default setting.'
            )
        self._categories_filter = categories if categories else None
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def post_build(self, dataset):
        """Filter by categories if self._categories_filter was set."""
        if getattr(self, '_categories_filter', None) and len(self._categories_filter) > 0:
            keywords = [str(kw) for kw in self._categories_filter]
            mask = self.data['category'].apply(
                lambda c: any(kw in str(c) for kw in keywords)
            )
            self.data = self.data[mask].reset_index(drop=True)


    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        ###### Handle No Image Case ######
        if line['image'] == 'nan':
            msgs = [dict(type='text', value=question)]
            return msgs
        ###### Add by bryce; END ######

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))

        return msgs
    
    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmoral import MMOral_Omni_auxeval, MMOral_Omni_acc
        
        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=16384, **judge_kwargs)
            # model = build_judge(max_tokens=3, **judge_kwargs) ; !!! invalid for [max_tokens=3] !!!
            assert model.working(), 'MMOral evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMOral_Omni_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v[
                        'score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMOral_Omni_acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        score_fine_pth = get_intermediate_file_path(storage, '_score_fine', 'csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score

class MMOral_OMNI_Mini(MMOral_OMNI):
    DATASET_URL = {
        'MMOral_OMNI_Mini':
        'https://huggingface.co/datasets/OralGPT/MMOral-Omni-Bench-Mini/resolve/main/MMOral-Omni-Bench-Mini.tsv',
    }
    DATASET_MD5 = {
        'MMOral_OMNI_Mini': '59dde61ef6186e6b6705860c913d7fe2',
    }

    def __init__(self, dataset='MMOral_OMNI_Mini', skip_noimg=False):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)