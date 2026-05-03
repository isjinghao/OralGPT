python /home/jinghao/projects/OralGPT-CMF/cmf_xray_oralagent_report_pipeline.py \
  --dataset-root /data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata \
  --output-dir /home/jinghao/projects/OralGPT-CMF/outputs/patient_json_xray_oralagent_report \
  --api-base http://0.0.0.0:8124/v1 \
  --model local-agent \
  --workers 1 \
  --overwrite
