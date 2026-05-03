python cmf_XLa_CT_gpt_pipeline.py \
  --dataset-root /data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata \
  --output-dir ./outputs/patient_json_XLa_CT \
  --model claude-opus-4-7 \
  --max-patients 5 \
  --workers 1 \
  --overwrite

