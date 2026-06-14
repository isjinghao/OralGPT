python cmf_XLa_CT_gpt_pipeline.py \
  --dataset-root /data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata \
  --output-dir ./outputs/stage2_patient_json_XLa_CT \
  --model gpt-5 \
  --workers 16 \
  --overwrite \
  # --max-patients 5
