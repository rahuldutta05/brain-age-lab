# Brain Age Lab — Streamlit demo

A single Streamlit application for exploring V3 cross-validation results and running the supplied SFCN checkpoint on compatible preprocessed MRI volumes.

## Important provenance note

This starter app was assembled from the uploaded `v3 approach (2).zip`. That archive contains V3 artifacts, not the later V4 checkpoint. It includes `m1_f0_s0.pt`, a single V3 fold/seed checkpoint. The app therefore labels its live uploaded-volume prediction as a single-checkpoint research demo and does **not** claim that this checkpoint is the final V4 model.

The archive's V3 evaluation file reports:
- Bias-corrected MAE: approximately 3.765 years
- Raw MAE: approximately 3.267 years
- Three-class age-group accuracy: approximately 78.16%
- Pearson r: approximately 0.885
- RMSE: approximately 4.812 years

These are the values in the supplied V3 result file. They may not match a newer experiment.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Run inference

**Layer 1 (MRI Viewer & Scan Input)** accepts user-uploaded `.npz` files containing a `data` array with shape `(160, 192, 160)`. The volume must be preprocessed according to the V3 pipeline: skull-stripped, affine-registered to MNI152 1 mm space, and brain-only intensity z-scored.

The app also provides an interactive option to generate and test with a synthetic 3D brain phantom directly in the UI if you do not have an OASIS-3 NPZ file immediately on hand.

Once a volume is loaded:
- **Layer 1**: Inspect 3D orthogonal slices (Axial, Coronal, Sagittal) and intensity profiles.
- **Layer 2**: Run live SFCN CPU inference to estimate brain age and inspect model confidence.
- **Layer 3**: Analyze Brain Age Gap (BAG = Predicted − Chronological Age) and compare with benchmark cohorts.
- **Layer 4**: Explore full test-set validation accuracy, scatter plots, error distributions, and subgroup MAE.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Deploy and test the Overview and Explore predictions tabs first.
5. Test live inference with a small, valid preprocessed NPZ only after confirming the deployment memory budget.

The checkpoint is approximately 12 MB. PyTorch is a comparatively large dependency and 3D inference can use significant RAM, so test the cloud deployment before relying on it during a panel presentation. Keep a local copy ready as a backup.

## Files

- `app.py`: Streamlit UI
- `model.py`: SFCN architecture matching the V3 checkpoint
- `inference.py`: checkpoint loading and single-volume inference
- `checkpoints/`: checkpoint, checkpoint metadata, age-bin configuration
- `data/raw_m1_predictions.csv`: recorded V3 predictions before brain-age-gap bias correction
- `data/bag_m1_predictions.csv`: recorded V3 predictions with the stored BAG/bias-correction outputs
- `data/demo_predictions_100.csv`: 100-row subset for interactive scan-record exploration
- `data/v3_accuracy_m1.json`: V3 evaluation summary

## Research and privacy notes

- This is a research demonstration, not a medical device.
- Brain age is not a clinical diagnosis.
- Use de-identified data and verify that you have permission to upload/share every MRI volume.
- Recorded cross-validation predictions and live checkpoint inference are distinct modes and are labeled separately in the UI.
