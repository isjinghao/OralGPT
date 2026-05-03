python /home/jinghao/projects/OralGPT-CMF/textrecord_treatment_qa_pipeline.py \
  --dataset-root /data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata \
  --output-dir /home/jinghao/projects/OralGPT-CMF/outputs/patient_json_treatment \
  --overwrite \
  --max-patients 10 \
  --workers 4 \
  --model gpt-5.4

