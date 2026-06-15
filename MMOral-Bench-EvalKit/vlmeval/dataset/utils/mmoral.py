from ...smp import *
import random


def build_mmoral_omni_gpt5_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    prompt = """
    You are an expert judge for a benchmark on oral and dental multimodal imaging diagnosis, examination, and treatment planning. Your task is to strictly evaluate how correct an AI model's prediction is compared to the ground truth. Just complete the last space of the correctness score without providing explanations or reasoning. The numeric correctness score must be one of 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).

    **Evaluation rules**
    - Only output the numeric score, nothing else. 
    - Treat clinically non-equivalent statements (e.g., wrong disease, wrong tooth or side, wrong stage or grade, wrong severity) as largely incorrect.
    - If the model’s prediction and the ground truth differ in their essential clinical content regarding the conditions or abnormalities, it should be treated as a major error.
    - If the ground truth includes fine-grained coordinates or tooth numbers that are omitted in the model’s prediction, but their clinical meaning remains consistent, treat it as a minor omission.
    - Judge semantic equivalence, not wording. Accept clinically equivalent synonyms (e.g., “caries” ≈ “tooth decay”; “periapical radiolucency” ≈ “apical rarefaction”).
    - Accept equivalent tooth numbering systems (FDI, Universal, Palmer).
    - Do not over-penalize minor phrasing differences; focus on clinical meaning.
    - If the model’s prediction includes additional statements that contradict the GT or introduce clinically false information, treat it as a major error, even if the rest matches.
    - The quality of the model’s prediction is determined by how closely it aligns with the ground truth, not by its length or verbosity. Both insufficient and excessive content should result in score penalties.

    **Scoring guide**  
    * 1.0: Fully correct; matches GT in all clinically material aspects; no extra wrong claims.
    * 0.7–0.9: Mostly correct; only minor omissions or clinically insignificant inaccuracies; no contradiction to GT.
    * 0.3–0.6: Partially correct; some correct elements but misses at least one core aspect or introduces a major error.
    * 0.1–0.2: Largely incorrect; minimal overlap with GT.
    * 0.0: Totally incorrect; no meaningful overlap; refusal to answer; generic non-answer; or answer contradicts the GT fundamentally.

    **Example format:**

    Question | Ground truth | Prediction | Correctness
    --- | --- | --- | ---
    This is an intraoral photograph of the oral cavity. What is the likely diagnosis? | Findings are most consistent with a minor aphthous ulcer (minor aphthous stomatitis) on the ventral anterior tongue. | The image shows a small, shallow sore on the underside of the front part of the tongue, suggestive of a minor aphthous ulcer. | 1.0
    This is an intraoral photograph. What disease(s) is demonstrated here? | The permanent tooth decay is detected in regions <box>[81,90,353,339]</box> and <box>[108,267,401,497]</box>. | This intraoral image shows the presence of dental caries (tooth decay) on multiple molars, characterized by brownish and darker areas on the occlusal surfaces of the teeth. The decay appears to be advancing, indicating active carious lesions. | 0.8
    Analysis of the lateral cephalometric radiograph requires:\n1. Landmark identification: Provide anatomical landmarks with coordinates in the format: <points>(x='…', y='…', alt='…')</points>\n2. CVM staging: What is the estimated Cervical Vertebral Maturation (CVM) stage based on this analysis? | <points x='1629' y='884' alt='A-point'></points>\n<points x='1594' y='1081' alt='Subnasale'></points>\n CVM Stage S6 | <points>(x='1627', y='874', alt='A-Point')</points>\n<points>(x='1592', y='1078', alt='Subnasale')</points>\n<points>(x='1599', y='1071', alt='Subnasale')</points>\n CVM Stage: S5 | 0.7
    The pathological image from the gingiva demonstrates a lesion measuring 4.5 cm. Evaluate this pathological patch's morphology and indicate if it corresponds to leukoplakia without dysplasia, leukoplakia with dysplasia, or oral squamous cell carcinoma. | Histopathological findings indicate leukoplakia without dysplasia. | The lesion most closely corresponds to leukoplakia with dysplasia or possibly early oral squamous cell carcinoma** depending on invasion, but based purely on this image, leukoplakia with dysplasia is a reasonable assessment. | 0.5
    This is an intraoral photograph of the oral cavity. What condition is visible? | Mixed dentition period, the upper central incisor space and mandibular space can be seen. | The image shows a case of severe dental crowding and malalignment of the teeth, likely due to malocclusion. The teeth are misaligned, overlapping, and displaced, which appears to be a case of advanced dental crowding. This condition often results from insufficient space in the dental arch for the erupted teeth. | 0.2
    This is a periapical X-ray image. Please identify the disease(s) or condition(s). | Periapical radiolucency at the apex of the posterior tooth consistent with chronic apical periodontitis (periapical granuloma versus small radicular cyst) associated with a non‑endodontically treated tooth; correlate clinically and consider endodontic therapy. | This periapical X-ray image shows multiple affected teeth with signs consistent with dental caries (cavities). There are no obvious signs of advanced periodontal disease or other abnormalities. | 0.0
    """

    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt

def MMOral_Omni_auxeval(model, line):
    def float_cvt(s):
        try:
            return float(s)
        except ValueError:
            return None

    prompt = build_mmoral_omni_gpt5_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 1:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, score=0.0)


def MMOral_Omni_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate2_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        grade = float(item['score'])
        cate_list = [
                    # 'TE', 'Oral Histopathology', 'Oral Mucosal Disease', 'Oral & Maxillofacial Radiology', 
                     'TP', 'Endodontics', 'Implant Dentistry', 'Periodontics',
                     'II_Loc',
                     'II_Dx-I', 'Orthodontics', 'Cancer', 'Gingivitis', 'Defective Dentition', 'Normality', 'Tooth Discoloration', 'Ulcer', 'Caries', 'Calculus',
                     'II_Dx-R', 'Fenestration and Dehiscence', 'Malocclusion Issues Assessment', # "Caries", 
                     'PA', 'Impacted Tooth', 'Pulpitis', 'Periodontitis', 'Apical Periodontitis', 'Mixed Dentition', 'Bone Loss', 'Root Canal Treatment', 'Crown', 'Restoration', 
                     'CE',
                     'PI', 'Leukoplakia with Dysplasia', 'Leukoplakia without Dysplasia', 'Oral Squamous Cell Carcinoma', 'Oral Submucous Fibrosis', # "Normal", 
                     'IV'
                     ]

        for capa in cate_list:
            if capa in cate:
                tot[capa] += 1
                score[capa] += grade
        tot['Overall'] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate2] += grade

    res = defaultdict(list)
    res2 = defaultdict(list)
    cate_list.append('Overall')
    cate2_list.append('Overall')
    for k in cate_list:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100 if tot[k] else 0)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 100 if tot[k] else 0)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    return res, res2

############## for MMOral-OPG-Bench ##############

def build_mmoral_opg_gpt4_prompt(line):
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

def MMOral_opg_auxeval(model, line):
    def float_cvt(s):
        try:
            return float(s)
        except ValueError:
            return None

    prompt = build_mmoral_opg_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 1:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, score=0.0)

def MMOral_opg_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate2_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        grade = float(item['score'])
        cate_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec', 'Report']
        for capa in cate_list:
            if capa in cate:
                tot[capa] += 1
                score[capa] += grade
        tot['Overall'] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate2] += grade

    res = defaultdict(list)
    res2 = defaultdict(list)
    cate_list.append('Overall')
    cate2_list.append('Overall')
    for k in cate_list:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 100)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    return res, res2

def get_single_choice_prediction(response, all_choices, index2ans):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    candidates = []

    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)
            elif f' {choice}.' in response:
                candidates.append(choice)
            elif f' {choice},' in response:
                candidates.append(choice)
    
    if len(candidates) == 0:
        for index, ans in index2ans.items():
            ans_str = str(ans)
            if ans_str in response:
                candidates.append(index)
    
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
    
    return random.choice(all_choices)