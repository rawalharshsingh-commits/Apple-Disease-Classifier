# Running apple_leaf.py on Kaggle / Colab

## 0. Setup (Kaggle notebook: turn on GPU, add the "plant-pathology-2020-fgvc7" competition data)
```
!pip install -q timm albumentations
!python apple_leaf.py folds --data_dir /kaggle/input/plant-pathology-2020-fgvc7 --out_dir outputs
```
(On Colab: download the data with the Kaggle CLI first and pass its folder as --data_dir.)

## 1. Smoke test (1-2 minutes, confirms everything works)
```
!python apple_leaf.py train --model densenet121 --fold 0 --epochs 2 --limit 64 --img_size 224 --tag _smoke
!python apple_leaf.py eval --runs densenet121_smoke --folds 0
```

## 2. Baseline: train the three models on one fold first (check results before running all 5)
```
!python apple_leaf.py train --model densenet121 --fold 0
!python apple_leaf.py train --model effb7        --fold 0 --bs 8 --accum 2
!python apple_leaf.py train --model effb7_ns     --fold 0 --bs 8 --accum 2
!python apple_leaf.py eval --runs densenet121 effb7 effb7_ns --ensemble densenet121 effb7 effb7_ns --folds 0 --note baseline
```
Then repeat for --fold 1..4 and run eval with --folds 0 1 2 3 4.
Kaggle sessions are time-limited: train one model/fold per session or save outputs/ as a dataset between runs.

## 3. Imbalance experiments (compare against the baseline's multiple_diseases recall)
```
!python apple_leaf.py train --model effb7_ns --fold 0 --bs 8 --accum 2 --loss focal_weighted --tag _focal
!python apple_leaf.py train --model effb7_ns --fold 0 --bs 8 --accum 2 --balanced_sampler --tag _bal
```

## 4. Distillation into a small student
```
!python apple_leaf.py train --model student_b0 --fold 0 --tag _kd --teachers densenet121 effb7 effb7_ns
!python apple_leaf.py eval --runs student_b0_kd --folds 0
!python apple_leaf.py bench --runs densenet121 effb7 effb7_ns student_b0_kd
```

Outputs (in outputs/): folds.csv, *.pt checkpoints, *_val.npz predictions, *_log.csv, results.csv, cm_*.png, bench.csv.
