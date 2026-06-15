# python run.py --config config_mmoral_opg.json \
# python run.py --config config_OralQA_ZH.json \
# python run.py --config config_mmoral_omni_OralAgent.json \
python run.py --config config_mmoral_opg.json \
              --api-nproc 64 \
              --work-dir '.' \
              --verbose \
              --mode all \
            #   --reuse