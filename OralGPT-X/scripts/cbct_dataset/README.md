## CBCT BAGEL Dataset Preparation

This folder records the scripts used to build BAGEL image-editing parquet
datasets for low-dose to standard-dose CBCT enhancement.

The scripts do not hard-code local machine paths. Pass the DICOM root with
`--input-root` and choose an output directory with `--output-root`.

### Expected DICOM Layout

The input root should contain one directory per volume and dose:

```text
<input-root>/
  100_337510_78/
  100_337510_333/
  10_348553_131/
  10_348553_333/
  41_338521_123/
  41_338521_333/
```

Directory names must end with one of `_78`, `_123`, `_131`, or `_333`.
DICOM slice filenames are expected to end in `_<slice-index>.dcm`.

### Preprocessing

`cbct_preprocess.py` provides the shared preprocessing used by all builders:

- apply DICOM `RescaleSlope` and `RescaleIntercept`
- clip intensities to `[-1000, 3000]`
- rescale to uint8 PNG bytes
- invert MONOCHROME1 images

The dataset builders additionally keep only original slices `50..240` and store
both preprocessed slice indices and original DICOM slice indices in each row.

### Dataset Builders

`prepare_bagel_cbct_edit_data.py` builds these 2-image tasks:

- `78 -> 333`: source slice `k` to target slice `k`
- `123 -> 333`: exact z-position match; common cases use source `4*n` to target `3*n`, special 125-slice cases use source `i` to target `2*i`
- `131 -> 333`: source slice `k` to target slice `2*k-1`

`prepare_bagel_cbct_zinterp_data.py` builds this 3-image task:

- `131_zinterp -> 333`: adjacent source slices `k,k+1` to target slice `2*k+1`

### Similarity Split

Both builders support volume-level split. With `--split-strategy similarity`,
the scripts rank volumes by mean normalized cross-correlation over all paired
slices, then place the highest-scoring 10% of volumes in test when
`--train-ratio 0.9`.

For the 2-image tasks, similarity is computed between source and target.
For `131_zinterp`, similarity is computed between the average of the two source
slices and the target.

### Generate Final Enhancement Datasets

```bash
python scripts/cbct_dataset/prepare_bagel_cbct_edit_data.py \
  --input-root /path/to/cbct_dicom_root \
  --output-root /path/to/dataset_Low-Dose_to_Standard_CBCT \
  --doses 78 123 131 \
  --split-strategy similarity
```

Outputs:

```text
<output-root>/
  train/cbct_78_to_333/
  train/cbct_123_to_333/
  train/cbct_131_to_333/
  test/cbct_78_to_333/
  test/cbct_123_to_333/
  test/cbct_131_to_333/
  parquet_info/
  split_manifest.json
  volume_similarity_report.json
  bagel_dataset_info_snippet.py
  bagel_example_config_snippet.yaml
```

### Generate Final 131 Z-Interpolation Dataset

```bash
python scripts/cbct_dataset/prepare_bagel_cbct_zinterp_data.py \
  --input-root /path/to/cbct_dicom_root \
  --output-root /path/to/dataset_Low-Dose_to_Standard_CBCT \
  --split-strategy similarity
```

Outputs:

```text
<output-root>/
  train/cbct_131_zinterp_to_333/
  test/cbct_131_zinterp_to_333/
  parquet_info/train_cbct_131_zinterp_to_333.json
  parquet_info/test_cbct_131_zinterp_to_333.json
  split_manifest_131_zinterp_to_333.json
  volume_similarity_report_131_zinterp.json
  bagel_131_zinterp_dataset_info_snippet.py
  bagel_131_zinterp_example_config_snippet.yaml
```

### Optional PNG Export For Debugging

```bash
python scripts/cbct_dataset/cbct_preprocess.py \
  --input-root /path/to/cbct_dicom_root \
  --output-root /path/to/preprocessed_png \
  --max-files 100
```

