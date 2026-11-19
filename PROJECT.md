# PROJECT.md — Apple Leaf Disease Detection (Reproduction + Extensions)

> Instructions for Codex: read this whole file first. Work **one stage at a time**, in order.
> After each stage, run its smoke test, show the output, and stop for confirmation before the next stage.
> Do not invent results. Never hard-code metrics. All numbers in reports must come from saved logs.

## 1. Goal

Reproduce the paper **"Disease Detection in Apple Leaves Using Deep Convolutional Neural Network"**
(Bansal, Kumar, Kumar; *Agriculture* 2021, 11(7), 617): an ensemble of pre-trained **DenseNet121**,
**EfficientNet-B7** and **EfficientNet-B7 NoisyStudent** classifying apple leaves into 4 classes:
`healthy`, `multiple_diseases`, `rust`, `scab`.

Then extend it with four contributions (these are the basis of a possible follow-up paper):

1. **Leak-free evaluation**: split by image *before* any augmentation; check for near-duplicate images across splits.
2. **Cross-dataset generalization**: train on Plant Pathology 2020 (FGVC7), test on a different apple-leaf dataset.
3. **Imbalance handling** for the `multiple_diseases` class (class weights / focal loss / balanced sampling).
4. **Distillation** of the 3-model ensemble into one lightweight student (e.g., EfficientNet-B0 or MobileNetV3), with accuracy vs. latency vs. size comparison.

Plus: Grad-CAM explainability, calibration (temperature scaling), and a small demo app.

## 2. Tech stack

- Python 3.10+
- PyTorch + torchvision, **timm** for backbones
- Albumentations + OpenCV (augmentation), scikit-learn (splits, metrics), pandas, numpy
- matplotlib / seaborn (plots), `pytorch-grad-cam` (explainability), `imagehash` + Pillow (duplicate check)
- PyYAML (configs), tqdm, TensorBoard (or Weights & Biases) for logging
- Streamlit (demo app)
- `kaggle` CLI for dataset download

Pin everything in `requirements.txt`. Set seeds (python, numpy, torch) in one `seed_everything()` util.

## 3. Dataset

**Primary:** Kaggle competition *Plant Pathology 2020 – FGVC7*
(`kaggle competitions download -c plant-pathology-2020-fgvc7`). Requires the user to accept the
competition rules and provide their own Kaggle API token. Do not try to bypass this; if download
fails, print clear manual instructions.

- `train.csv`: columns `image_id, healthy, multiple_diseases, rust, scab` (one-hot).
- `images/`: jpg files named by `image_id`.
- Only the labeled training set is used for supervised training/evaluation (test labels are not public).
- Expect heavy class imbalance (`multiple_diseases` is smallest). Print the real counts in Stage 1.

**Cross-dataset test set (Stage 7):** a second apple-leaf dataset, e.g., Plant Pathology 2021 (FGVC8)
or the apple subset of PlantDoc / PlantVillage. Map overlapping classes only (scab, rust, healthy) and
document the mapping. If the user hasn't downloaded it, make this stage skippable via config.

## 4. Repository layout

```
apple-leaf/
├── PROJECT.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── densenet121.yaml
│   ├── effnet_b7.yaml
│   ├── effnet_b7_ns.yaml
│   └── student.yaml
├── data/                  # gitignored; raw + processed
├── src/
│   ├── data/              # dataset.py, splits.py, transforms.py, dedupe.py
│   ├── models/            # factory.py, ensemble.py
│   ├── train.py
│   ├── evaluate.py
│   ├── distill.py
│   ├── calibrate.py
│   ├── gradcam.py
│   └── utils/             # seed.py, metrics.py, logging.py
├── app/streamlit_app.py
├── outputs/               # checkpoints, logs, figures, results.csv (gitignored)
├── notebooks/             # optional Kaggle/Colab runner
└── tests/                 # smoke tests
```

All hyperparameters live in YAML configs, not in code.

## 5. Stages

### Stage 0 — Setup
- Create the layout, `requirements.txt`, `.gitignore`, `seed.py`, config loader.
- **Smoke test:** `python -c "import torch, timm; print(torch.__version__, timm.__version__)"`.

### Stage 1 — Data & splits
- Load `train.csv`, convert one-hot to a single integer label, print class counts.
- Implement **stratified 5-fold** split (seeded) saved to `data/folds.csv`. Splitting happens on raw
  images, **before** any augmentation.
- Implement `dedupe.py`: compute perceptual hashes (`imagehash.phash`) and report near-duplicate pairs
  **across** train/val folds. Save a report to `outputs/duplicates.csv`. Do not silently delete anything.
- `dataset.py`: PyTorch Dataset reading from `folds.csv`; augmentation is applied **on the fly, train fold only**.
- **Smoke test:** print class distribution per fold and load one batch of 8 images; save a grid image.

### Stage 2 — Augmentation
- Train transforms (Albumentations): resize (config: 384 default, 512 optional), horizontal/vertical flip,
  random rotate, brightness/contrast, Gaussian/motion blur, normalization with ImageNet stats.
- Optional experiment flag `canny_aug: true` that adds a Canny-edge variant, as in the paper.
- Validation/test transforms: resize + normalize only.
- **Smoke test:** save a figure showing one image under 8 random augmentations.

### Stage 3 — Models & training
- `models/factory.py` builds backbones via `timm.create_model(name, pretrained=True, num_classes=4)`:
  - `densenet121`
  - `tf_efficientnet_b7` (use the standard ImageNet-pretrained tag)
  - `tf_efficientnet_b7.ns_jpg_in1k` (NoisyStudent)
  - If a tag is unavailable in the installed timm version, list available tags with
    `timm.list_pretrained('*efficientnet_b7*')` and ask the user which to use. Don't guess silently.
- `train.py`: AdamW, cosine LR with warmup, mixed precision (`torch.cuda.amp`), gradient accumulation
  option, early stopping on validation macro-F1, label smoothing option, checkpoint best model per fold.
- Loss options (config): cross-entropy, class-weighted CE, focal loss. Optional `WeightedRandomSampler`.
- Log per epoch: loss, accuracy, macro-F1, per-class recall → TensorBoard + `outputs/<run>/log.csv`.
- **Smoke test:** train 2 epochs on 64 images with a small input size; confirm checkpoint is written.

### Stage 4 — Ensemble & evaluation
- `ensemble.py`: average softmax probabilities of the three models (also support weighted average).
- `evaluate.py` computes on held-out folds: accuracy, per-class precision/recall/F1, macro-F1, ROC-AUC
  (one-vs-rest), confusion matrix (saved as PNG), and appends one row per experiment to
  `outputs/results.csv` (columns: run_id, models, loss, img_size, fold, metrics, timestamp).
- Report mean ± std across folds.
- **Smoke test:** evaluate on a tiny subset and verify `results.csv` is created.

### Stage 5 — Explainability & calibration
- `gradcam.py`: Grad-CAM overlays for correct and incorrect predictions per class; save grids.
- `calibrate.py`: temperature scaling on a validation split; report ECE before/after and a reliability diagram.

### Stage 6 — Distillation
- `distill.py`: train a student (config: `efficientnet_b0` or `mobilenetv3_large_100`) with a loss of
  `alpha * KL(student_T, teacher_T) + (1 - alpha) * CE`, where the teacher is the ensemble's averaged logits/probs.
- Report a comparison table: accuracy, macro-F1, `multiple_diseases` recall, parameters (M), model size (MB),
  CPU latency (ms/image, batch 1), for each single model, the ensemble, and the student.

### Stage 7 — Cross-dataset test (skippable)
- Evaluate the trained models/ensemble/student on the second dataset (overlapping classes only).
- Report the accuracy/F1 drop vs. in-dataset results. This is the key generalization result.

### Stage 8 — Demo app
- `app/streamlit_app.py`: upload a leaf image → predicted class, class probabilities (calibrated),
  Grad-CAM overlay, and a clear "this is not a substitute for expert diagnosis" note.
- Use the student model by default, ensemble as an option.

### Stage 9 — Documentation
- `README.md`: setup, how to download data, how to reproduce each stage with exact commands.
- Auto-generate `outputs/report.md` from `results.csv` (tables and figure links only — no invented numbers).

## 6. Rules for Codex

- Never fabricate results, citations, or dataset statistics. Print real values from the data.
- Never put augmentation before the train/val split.
- Keep each stage runnable from the command line, e.g. `python -m src.train --config configs/effnet_b7.yaml --fold 0`.
- GPU is not available in this sandbox: only run smoke tests here, and provide the exact full-training
  commands for the user to run on Kaggle/Colab (T4/P100). Use mixed precision and a smaller image size
  (e.g., 384) if memory is tight.
- Ask the user before installing anything not in `requirements.txt` or downloading large files.
- Commit after each stage with a clear message.

## 7. Definition of done

- [ ] All stages 0–6 and 8–9 implemented with passing smoke tests (stage 7 if the second dataset is available)
- [ ] `outputs/results.csv` has rows for the 3 single models, the ensemble, and the student
- [ ] Duplicate report, confusion matrices, Grad-CAM grids, reliability diagram saved
- [ ] README reproduces everything from a clean clone
- [ ] Demo app runs locally with `streamlit run app/streamlit_app.py`
