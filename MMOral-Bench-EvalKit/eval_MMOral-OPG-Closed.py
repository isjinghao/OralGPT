import os
import os.path as osp
import pandas as pd
import torch
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import base64
import io
import time
import requests
import json
import argparse

def prepare_image_dir(output_dir, dataset_name):
    img_dir = f"{output_dir}/{dataset_name}_images"
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def load_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    return df

def get_single_choice_prediction(response, all_choices, index2ans):
    """Extract single choice answer from model response"""
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    candidates = []

    # 1. Check for explicit option markers (A), (B), (C), (D)
    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)

    # 2. Check for option letters
    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)
            elif f' {choice}.' in response:
                candidates.append(choice)
            elif f' {choice},' in response:
                candidates.append(choice)
    
    # 3. Check for option content
    if len(candidates) == 0:
        for index, ans in index2ans.items():
            ans_str = str(ans)
            if ans_str in response:
                candidates.append(index)
    
    # 4. If multiple candidates, choose the first occurrence
    if len(candidates) > 0:
        positions = {}
        for c in candidates:
            pos = response.find(f' {c} ')
            if pos == -1:
                pos = response.find(f'({c})')
            if pos == -1:
                pos = response.find(str(index2ans[c]))
            if pos != -1:
                positions[c] = pos
        
        if positions:
            return min(positions.items(), key=lambda x: x[1])[0]
    
    # 5. If no option found, randomly select one
    return random.choice(all_choices)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_inference_gpt4o(dataset, img_dir, api_url, api_key):
    print("Running inference with GPT-4o...")
    results = defaultdict(list)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        results['index'].append(idx)
        results['question'].append(row['question'])
        results['option1'].append(row['option1'])
        results['option2'].append(row['option2'])
        results['option3'].append(row['option3'])
        results['option4'].append(row['option4'])
        results['answer'].append(row['answer'])
        results['category'].append(row['category'])
        
        image_path = osp.join(img_dir, f"{idx}.jpg")
        
        if not osp.exists(image_path):
            try:
                img_data = base64.b64decode(row['image'])
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                img.save(image_path)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                results['prediction'].append("Error")
                continue
        
        options_prompt = 'Options:\n'
        for i, letter in zip(range(1, 5), ['A', 'B', 'C', 'D']):
            option_value = str(row[f'option{i}'])
            options_prompt += f"{letter}. {option_value}\n"
        
        prompt = (f'Question: {row["question"]}\n{options_prompt}'
                + 'Please answer the above multiple-choice question by selecting the single correct option (A, B, C, or D). '
                + 'If the provided information is insufficient to determine a clear answer, please choose the most likely correct option based on the available data and your judgment.')
        
        try:
            base64_image = encode_image_to_base64(image_path)
            
            data = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.2
            }
            
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            response_json = response.json()
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                model_response = response_json["choices"][0]["message"]["content"]
                results['prediction'].append(model_response)
            else:
                print(f"Error for sample {idx}: {response_json}")
                results['prediction'].append("Error: API response format unexpected")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error during inference for sample {idx}: {e}")
            results['prediction'].append(f"Error: {str(e)}")
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                print("Rate limit hit, waiting for 60 seconds...")
                time.sleep(60)
    
    return pd.DataFrame(results)

def evaluate_results(results_df):
    print("Evaluating results...")
    
    processed_results = results_df.copy()
    processed_results['predicted_option'] = processed_results.apply(
        lambda row: get_single_choice_prediction(
            row['prediction'], 
            ['A', 'B', 'C', 'D'], 
            {
                'A': row['option1'],
                'B': row['option2'],
                'C': row['option3'],
                'D': row['option4']
            }
        ), 
        axis=1
    )
    
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    
    main_category_list = ['teeth', 'patho', 'his', 'jaw', 'summ', 'Overall']
    categories = set(processed_results['category'].unique())
    subcategories = set([cat.replace(',', '_') for cat in categories])
    
    for _, row in processed_results.iterrows():
        category = row['category']
        subcategory = category.replace(',', '_')
        
        for main_cat in main_category_list[:-1]:
            if main_cat in category:
                tot[main_cat] += 1
        
        tot[category] += 1
        tot[subcategory] += 1
        tot['Overall'] += 1
    
    for _, row in processed_results.iterrows():
        category = row['category']
        subcategory = category.replace(',', '_')
        
        if row['predicted_option'] == row['answer']:
            for main_cat in main_category_list[:-1]:
                if main_cat in category:
                    score[main_cat] += 1
            
            score[category] += 1
            score[subcategory] += 1
            score['Overall'] += 1
    
    main_result = defaultdict(list)
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
    
    return main_df, detailed_df, processed_results

def save_results(results_df, main_results, detailed_results, output_dir, dataset_name, model_name):
    results_file = f"{output_dir}/{model_name}_{dataset_name}_results.csv"
    main_result_file = f"{output_dir}/{model_name}_{dataset_name}_main_acc.csv"
    detail_result_file = f"{output_dir}/{model_name}_{dataset_name}_detailed_acc.csv"
    
    results_df.to_csv(results_file, index=False)
    main_results.to_csv(main_result_file, index=False)
    detailed_results.to_csv(detail_result_file, index=False)
    
    print(f"Results saved to {output_dir}")
    print("Main category results:")
    print(main_results)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and evaluation for oral radiology multiple-choice VQA")
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to the benchmark TSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--api_url', type=str, required=True,
                        help='URL for the GPT API')
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for GPT')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model being evaluated')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.benchmark_path)
    
    # Prepare image directory
    img_dir = prepare_image_dir(args.output_dir, args.dataset_name)
    
    # Run inference with GPT
    results = run_inference_gpt4o(dataset, img_dir, args.api_url, args.api_key)
    
    # Evaluate results
    main_results, detailed_results, processed_results = evaluate_results(results)
    
    # Save results
    save_results(processed_results, main_results, detailed_results, 
                args.output_dir, args.dataset_name, args.model_name)

if __name__ == "__main__":
    main()