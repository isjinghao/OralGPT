# MRI BAGEL Dataset Scripts

This folder archives the code used to build the MRI T1/T2 BAGEL `unified_edit` datasets and the merged MRI training subset.

## Scripts

- `prepare_bagel_mri_t1_t2_data.py`  
  Builds cohort-level BAGEL parquet datasets for `Guizhou`, `KWC`, `Peking`, and `PWH`. The archived version uses environment variables or CLI overrides for raw data roots, rather than local absolute paths.

- `mri_visual_plane_align.py`  
  Provides in-plane transform scoring and application for mixed NIfTI/DICOM PWH data. The final PWH flow estimates one fixed transform per subject to avoid per-slice flip changes.

- `mri_pairs_export_slice_compare.py`  
  MRI DICOM/NIfTI loading, physical slice coordinate helpers, and QA preview utilities used by the dataset builder and PWH audit.

- `mri_pwh_pair_audit.py`  
  Audits PWH T1W NIfTI vs t2fs DICOM pairs and writes a `pair_audit_report.json`. This report is used to keep only subjects above the configured plane NCC threshold.

- `merge_bagel_mri_t1_t2_subsets.py`  
  Merges cohort-level BAGEL parquet datasets into one MRI subset while preserving provenance columns (`source_cohort`, `source_dataset_root`, `source_parquet_path`, `original_pair_id`, etc.).

## Reproduction Outline

Set raw data roots in the environment, or pass `--input-root` when supported:

```bash
export ORALGPT_MRI_GUIZHOU_ROOT=/path/to/Guizhou_images
export ORALGPT_MRI_KWC_ROOT=/path/to/KWC_images_by_sequences
export ORALGPT_MRI_PEKING_ROOT=/path/to/Peking
export ORALGPT_MRI_PWH_ROOT=/path/to/PWH
```

Generate cohort-level datasets:

```bash
python prepare_bagel_mri_t1_t2_data.py --cohort Guizhou --output-root /path/to/dataset_T1-T2_MRI/Guizhou
python prepare_bagel_mri_t1_t2_data.py --cohort KWC --output-root /path/to/dataset_T1-T2_MRI/KWC
python prepare_bagel_mri_t1_t2_data.py --cohort Peking --output-root /path/to/dataset_T1-T2_MRI/Peking
python prepare_bagel_mri_t1_t2_data.py --cohort PWH --output-root /path/to/dataset_T1-T2_MRI/PWH_fixed_flip
```

Merge the MRI subset:

```bash
python merge_bagel_mri_t1_t2_subsets.py \
  --output-root /path/to/dataset_T1-T2_MRI/MRI_T1_T2_all \
  --cohort Guizhou /path/to/dataset_T1-T2_MRI/Guizhou \
  --cohort KWC /path/to/dataset_T1-T2_MRI/KWC \
  --cohort Peking /path/to/dataset_T1-T2_MRI/Peking \
  --cohort PWH_fixed_flip /path/to/dataset_T1-T2_MRI/PWH_fixed_flip
```

The merged output includes:

- `train/mri_t1_to_t2/`
- `train/mri_t2_to_t1/`
- `test/mri_t1_to_t2/`
- `test/mri_t2_to_t1/`
- `parquet_info/`
- `bagel_dataset_info_snippet.py`
- `bagel_example_config_snippet.yaml`
- `merge_manifest.json`
- `summary.json`

## Privacy / GitHub Notes

The archived scripts have been sanitized to avoid local machine paths and raw dataset paths. Use environment variables or CLI arguments to provide site-local data paths when running them. Do not commit generated parquet files, audit reports containing private filesystem paths, or medical image data to GitHub.
