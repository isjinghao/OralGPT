python /home/jinghao/projects/OralGPT/OralGPT-CMF/cmf_diagnosis_qa_pipeline.py \
  --dataset-root /data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata \
  --output-dir /home/jinghao/projects/OralGPT/OralGPT-CMF/outputs/stage3_patient_json_diagnosis_qa \
  --examination-json-dir /home/jinghao/projects/OralGPT/OralGPT-CMF/outputs/stage1_patient_json_examination \
  --xla-ct-json-dir /home/jinghao/projects/OralGPT/OralGPT-CMF/outputs/stage2_patient_json_XLa_CT \
  --xray-report-json-dir /home/jinghao/projects/OralGPT/OralGPT-CMF/outputs/stage2_patient_json_xray_oralagent_report \
  --model gpt-5.4 \
  --workers 16 \
  --overwrite
  # --max-patients 5 \


