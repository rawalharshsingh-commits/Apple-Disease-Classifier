# Apple leaf disease classifier

A beginner-friendly PyTorch reproduction of Bansal, Kumar, and Kumar's 2021 paper, [“Disease Detection in Apple Leaves Using Deep Convolutional Neural Network”](https://www.mdpi.com/2077-0472/11/7/617). It trains three ImageNet-initialized image classifiers and combines their class probabilities by averaging.

The four output classes are `healthy`, `multiple_diseases`, `rust` (cedar apple rust), and `scab`.

## 1. Set up the project

Use Python 3.10 or newer. From this folder, create and activate a virtual environment, then install the listed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate it with `.venv\Scripts\activate`.

## 2. Prepare the Kaggle dataset

Download `plant-pathology-2020-fgvc7.zip` from Kaggle after accepting the competition rules. Put the ZIP in this project folder, then extract the labeled images and create a stratified 85/15 split:

```bash
python -m apple_disease.prepare_data --zip plant-pathology-2020-fgvc7.zip
```

The command creates:

```text
data/plant-pathology-2020-fgvc7/
├── train.csv
├── test.csv
├── sample_submission.csv
└── images/
outputs/split.csv
```

Only the rows in `train.csv` have labels and are used for supervised training. `test.csv` identifies competition test images but has no class labels, so those images are not included in training or validation.

The ZIP has 3,642 image files but 1,821 labeled rows. Those labeled rows contain 516 healthy, 91 multiple-disease, 622 rust, and 592 scab images. The paper reports a 3,642-image labeled set and an 85/15 split of 3,095/547. This Kaggle release does not provide labels for all 3,642 images, so this reproduction uses the 1,821 labeled rows (about 85/15 train/validation) and cannot exactly recreate that reported split. It also uses a fixed seed of 42 and stratifies by class because the paper does not give a split seed.

## 3. Train the three models

The paper describes 512 × 512 inputs, 20 epochs, batch size 128, transfer learning, and Adam. These are the training defaults. Train each model separately:

```bash
python -m apple_disease.train --model densenet121
python -m apple_disease.train --model efficientnet_b7
python -m apple_disease.train --model efficientnet_b7_noisy_student
```

The first model uses ImageNet-pretrained DenseNet121. The second uses ImageNet-pretrained EfficientNet-B7. The third uses EfficientNet-B7 Noisy Student weights. The Noisy Student weights are a separate pretrained checkpoint and may take time to download the first time.

Training augmentation is created online and only applied to training images. It includes horizontal and vertical flips, rotation, blur, and Canny edge augmentation. Use `--no-canny` to disable Canny. The paper describes these methods but does not report every transform probability or numeric setting; this implementation makes those choices explicit in `apple_disease/data.py`.

For a quick pipeline check using a small subset of **real labeled images**, without downloading pretrained weights:

```bash
python -m apple_disease.train \
  --model densenet121 --epochs 1 --batch-size 8 --image-size 128 \
  --limit 16 --workers 0 --device cpu --no-pretrained
```

That smoke run checks data loading and training code only. It is not a faithful experiment and its metrics should not be interpreted as model performance.

### Hardware options

The paper trained on a TPU. A GPU is recommended for the full 512-pixel runs. If your GPU runs out of memory, reduce the batch size while keeping other settings fixed:

```bash
python -m apple_disease.train --model densenet121 --batch-size 16
```

On Apple Silicon, automatic device selection uses MPS when PyTorch reports it is available. You can select CPU, CUDA, or MPS explicitly with `--device`.

## 4. Evaluate the models and ensemble

After training all three checkpoints:

```bash
python -m apple_disease.evaluate
```

To evaluate one model before all three are trained:

```bash
python -m apple_disease.evaluate --models densenet121
```

Evaluation reports accuracy, macro precision, macro recall, and macro F1, writes a confusion-matrix PNG per model, and saves metrics in `outputs/evaluation.csv`. The ensemble is the arithmetic mean of the three models’ softmax probabilities, matching the paper's averaging approach.

## 5. Try one image

After training all three models, classify one image from the command line:

```bash
python -m apple_disease.predict path/to/apple_leaf.jpg
```

Choose one model with `--model densenet121`, `--model efficientnet_b7`, or `--model efficientnet_b7_noisy_student`. The default is the three-model ensemble.

Or launch the local upload demo:

```bash
streamlit run app/streamlit_app.py
```

The demo is for research and education. It is not a substitute for expert diagnosis.

## Where files go

- `apple_disease/` contains the data, model, training, evaluation, and prediction code.
- `app/streamlit_app.py` is the image-upload demo.
- `data/` contains extracted dataset files and is ignored by Git.
- `outputs/` contains splits, training logs, model checkpoints, evaluation tables, and confusion matrices; it is ignored by Git.

## Reproduction notes

The article reports 96.25% validation accuracy for its averaged model. That is the paper's reported result, not a result produced by this repository. Reproduction numbers depend on the accessible labeled data, pretrained weight versions, library versions, hardware, split seed, and underspecified augmentation settings. This project prints and saves only metrics from runs you execute.
