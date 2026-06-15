import os
import os.path as osp
import pandas as pd
import torch
import json
import requests
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from base64 import b64decode, b64encode
import io
import argparse

def encode_image_to_base64(image, target_size=512):
    if target_size > 0:
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        image = image.resize((new_width, new_height))
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return b64encode(buffered.getvalue()).decode('utf-8')

def load_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    return df

def prepare_image_dir(output_dir, dataset_name):
    img_dir = f"{output_dir}/{dataset_name}_images"
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def build_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    prompt = """
Given the question, compare the ground truth and prediction from AI models, to generate a correctness score for the prediction.
The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical definition. | 30 | 1.0
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical definition. | 29 teeth are visualized with clear anatomical definition. | 0.0
What is the status of the wisdom teeth in the radiograph? | Three wisdom teeth are detected, all of which are impacted: #18, #28, and #48. | #18: impacted, #28: impacted, #48: erupted | 0.7
What is the condition of the teeth #26 and #14? | Teeth #26 and #14 show signs of periapical abscesses. | Teeth #26 and #23 show signs of periapical abscesses. | 0.5
What is the condition of the bone architecture and visible structures in the jaw? | No apparent bone loss is observed. Bilateral mandibular canals and maxillary sinuses are clearly visible. | Bilateral mandibular canals and maxillary sinuses are clearly visible. | 0.5
What is the clinical priority concern regarding the periapical lesions? | Periapical cysts at #11 and #12, and granuloma at #46 require endodontic evaluation. | Periapical lesions at #11, #12, and #46 require endodontic evaluation. | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Crown | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Crown at position: [1230, 627, 1276, 750] | 0.9
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Teeth at position: {\"point_2d\": [1242, 726]}},\n{Crown at position: {\"box_2d\": [1230, 627, 1276, 750]}} | 1.0
"""
    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt

def run_inference(dataset, img_dir, gpt_api_key, gpt_api_base, model_name="gpt-4o"):
    print("Running inference with GPT-4o...")
    results = defaultdict(list)
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {gpt_api_key}'
    }
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        results['index'].append(row['index'])
        results['image_name'].append(row.get('image_name', f"image_{idx}"))
        results['question'].append(row['question'])
        results['answer'].append(row['answer'])
        results['category'].append(row['category'])
        
        image_path = osp.join(img_dir, f"{row['index']}.jpg")
        
        if not osp.exists(image_path):
            try:
                img_data = b64decode(row['image'])
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                img.save(image_path)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                results['prediction'].append("Error")
                continue
        else:
            img = Image.open(image_path).convert('RGB')
        
        prompt = f'Question: {row["question"]}\nPlease provide a detailed and accurate answer to the question based on the dental radiographic image.'
        
        try:
            b64_img = encode_image_to_base64(img)
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a medical expert specialized in oral radiology. Answer questions about dental radiographic images accurately and comprehensively."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]}
                ],
                "temperature": 0.2
            }
            
            response = requests.post(
                gpt_api_base,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                gpt_response = result['choices'][0]['message']['content'].strip()
                results['prediction'].append(gpt_response)
            else:
                print(f"API Error: {response.status_code}, {response.text}")
                results['prediction'].append(f"API Error: {response.status_code}")
            
        except Exception as e:
            print(f"Error during inference for sample {idx}: {e}")
            results['prediction'].append(f"Error: {str(e)}")
    
    return pd.DataFrame(results)

def evaluate_with_gpt(results_df, img_dir, gpt_api_key, gpt_api_base):
    print("Starting evaluation...")
    print("Evaluating with GPT-4o...")
    
    results_df['score'] = None
    results_df['log'] = None
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {gpt_api_key}'
    }
    
    system_prompt = """
    You are a medical expert specialized in oral radiology. Your task is to evaluate the quality of answers to questions about oral radiographic images (like X-rays of teeth, jaws, etc.).
    
    Please provide a precise evaluation following the scoring system shown in the examples.
    """
    
    for idx, row in tqdm(results_df.iterrows(), total=len(results_df)):
        try:
            image_path = osp.join(img_dir, f"{row['index']}.jpg")
            if not osp.exists(image_path):
                print(f"Image not found: {image_path}")
                results_df.at[idx, 'score'] = 0
                results_df.at[idx, 'log'] = "Image not available for evaluation"
                continue
            
            img = Image.open(image_path).convert('RGB')
            b64_img = encode_image_to_base64(img)
            
            # Use the MMVET prompt format
            user_prompt = build_prompt(row)
            
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]}
                ],
                "temperature": 0.0
            }
            
            response = requests.post(
                gpt_api_base,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                gpt_response = result['choices'][0]['message']['content'].strip()
                
                # Try to parse the score from the MMVET format response
                # The response should be a simple number like "0.8"
                try:
                    # First try direct number extraction
                    for score in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]:
                        if score in gpt_response:
                            results_df.at[idx, 'score'] = float(score)
                            results_df.at[idx, 'log'] = f"MMVET score: {score}"
                            break
                    else:
                        # If no direct match, try to parse from more complex responses
                        try:
                            # Try to parse as JSON if it looks like JSON
                            if '{' in gpt_response and '}' in gpt_response:
                                evaluation = json.loads(gpt_response)
                                results_df.at[idx, 'score'] = float(evaluation.get('score', 0))
                                results_df.at[idx, 'log'] = evaluation.get('log', "No explanation provided")
                            # Otherwise, look for score mentions
                            elif "score" in gpt_response.lower():
                                score_text = gpt_response.lower().split("score")[1].split("\n")[0]
                                score = 0
                                if "1" in score_text or "one" in score_text:
                                    score = 1.0
                                elif "0.5" in score_text or "half" in score_text:
                                    score = 0.5
                                results_df.at[idx, 'score'] = score
                                results_df.at[idx, 'log'] = gpt_response
                            else:
                                # If all else fails, try to extract a float value
                                import re
                                numbers = re.findall(r"[0-9]+\.[0-9]+|[0-9]+", gpt_response)
                                if numbers:
                                    score_value = float(numbers[0])
                                    if 0 <= score_value <= 1:
                                        results_df.at[idx, 'score'] = score_value
                                        results_df.at[idx, 'log'] = f"Extracted score: {score_value}"
                                    else:
                                        results_df.at[idx, 'score'] = 0
                                        results_df.at[idx, 'log'] = f"Invalid score value: {score_value}, Response: {gpt_response}"
                                else:
                                    results_df.at[idx, 'score'] = 0
                                    results_df.at[idx, 'log'] = f"Failed to parse GPT response: {gpt_response}"
                        except json.JSONDecodeError:
                            results_df.at[idx, 'score'] = 0
                            results_df.at[idx, 'log'] = f"Failed to parse GPT response: {gpt_response}"
                except Exception as e:
                    print(f"Error parsing response for sample {idx}: {e}")
                    results_df.at[idx, 'score'] = 0
                    results_df.at[idx, 'log'] = f"Parsing error: {str(e)}, Response: {gpt_response}"
            else:
                print(f"API Error: {response.status_code}, {response.text}")
                results_df.at[idx, 'score'] = 0
                results_df.at[idx, 'log'] = f"API Error: {response.status_code}"
        
        except Exception as e:
            print(f"Error evaluating sample {idx}: {e}")
            results_df.at[idx, 'score'] = 0
            results_df.at[idx, 'log'] = f"Evaluation error: {str(e)}"
    
    return results_df

def calculate_scores(results_df):
    print("Calculating scores...")
    
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    
    main_category_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec', 'Report', 'Overall']
    categories = set(results_df['category'].unique())
    subcategories = set([cat.replace(',', '_') for cat in categories])
    
    for _, row in results_df.iterrows():
        category = row['category']
        subcategory = category.replace(',', '_')
        
        for main_cat in main_category_list[:-1]:
            if main_cat in category:
                tot[main_cat] += 1
                score[main_cat] += float(row['score'])
        
        tot[subcategory] += 1
        tot['Overall'] += 1
        
        score[subcategory] += float(row['score'])
        score['Overall'] += float(row['score'])
    
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
    
    return main_df, detailed_df

def save_results(results_df, main_results, detailed_results, output_dir, dataset_name):
    results_file = f"{output_dir}/{dataset_name}_results.csv"
    main_result_file = f"{output_dir}/{dataset_name}_main_acc.csv"
    detail_result_file = f"{output_dir}/{dataset_name}_detailed_acc.csv"
    excel_file = f"{output_dir}/{dataset_name}_evaluation.xlsx"
    
    results_df.to_csv(results_file, index=False)
    main_results.to_csv(main_result_file, index=False)
    detailed_results.to_csv(detail_result_file, index=False)
    
    try:
        with pd.ExcelWriter(excel_file) as writer:
            results_df.to_excel(writer, sheet_name='Results', index=False)
            main_results.to_excel(writer, sheet_name='Main Categories', index=False)
            detailed_results.to_excel(writer, sheet_name='Detailed Categories', index=False)
    except:
        results_df.to_excel(excel_file, index=False)
    
    print(f"Results saved to {output_dir}")
    print("Main category results:")
    print(main_results)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and evaluation for oral radiology VQA")
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to the benchmark TSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--gpt_api_key', type=str, required=True,
                        help='API key for GPT')
    parser.add_argument('--gpt_api_base', type=str, required=True,
                        help='Base URL for GPT API')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset'),
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to use for inference')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.benchmark_path)
    
    # Prepare image directory
    img_dir = prepare_image_dir(args.output_dir, args.dataset_name)
    
    # Run inference with GPT
    results = run_inference(dataset, img_dir, args.gpt_api_key, args.gpt_api_base, args.model_name)
    
    # Evaluate with GPT
    evaluated_results = evaluate_with_gpt(results, img_dir, args.gpt_api_key, args.gpt_api_base)
    
    # Calculate scores
    main_results, detailed_results = calculate_scores(evaluated_results)
    
    # Save results
    save_results(evaluated_results, main_results, detailed_results, args.output_dir, args.dataset_name)

if __name__ == "__main__":
    main()
