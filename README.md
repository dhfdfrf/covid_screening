# COVID Screening

COVID-19 chest image lesion segmentation project based on the QaTa-COV19 dataset.

The project includes preprocessing, model training, inference, evaluation, and thesis documentation. It implements an improved TransUNet2D v11 model with a shifted-window Transformer bottleneck, attention-gated skip connections, DropPath regularization, and Sobel-guided boundary refinement. It also compares and ensembles models such as U-Net, Attention U-Net, Swin-Unet, and UCTRansNet.

## v15 Inference Ensemble

v15 is the current default inference path. It is a validation-selected probability ensemble, not a newly trained single backbone:

- `0.5 * transunet2d_v14_aug_ema_ft20_best`
- `0.5 * uctransnet2d_qata_best`
- threshold `0.49`, `min_area=0`, `TTA=none`

The config is saved at:

```bash
configs/v15_ensemble_config.json
```

Fixed test result with the validation-selected setting:

```text
Dice=0.787900, IoU=0.685124, Precision=0.783126, Recall=0.848235
```

## v16 Lesion-Prior TransUNet

v16 is the model-side innovation version for the thesis. It keeps the stable v12/v14 backbone, then adds a lightweight lesion-prior branch:

- Image-derived prior from grayscale intensity, Sobel edge magnitude, local mean, and local contrast.
- Prior-guided skip attention that modulates each decoder skip connection.
- Auxiliary prior supervision through `prior_boundary_combo` loss, together with deep supervision and boundary supervision.

Recommended warm-start command:

```bash
python src/train_unet2d_qata.py --data_dir data/processed/qata --model transunet2d_v16 --epochs 20 --batch 16 --lr 2e-5 --loss prior_boundary_combo --boundary_weight 0.05 --warmup_epochs 2 --grad_clip 1.0 --augment --ema_decay 0.999 --select_metric best_threshold --resume outputs/transunet2d_v14_aug_ema_ft20_best.pt --resume_partial --run_tag transunet2d_v16_prior_ft20
```

## v17 Prior-Calibrated TransUNet

v17 is the safer model-innovation version after v16. It keeps the stable v12/v14 segmentation path unchanged, then adds an uncertainty-aware residual calibration branch:

- The prior branch still uses intensity, Sobel edges, local mean, and local contrast.
- The main decoder skip connections remain the stable v12/v14 attention gates.
- A zero-initialized residual calibrator adjusts logits using prediction uncertainty, prediction edges, image edges, and the prior map.
- `prior_calibration_combo` adds weak prior supervision and a small calibration regularizer, reducing the risk of the prior branch overpowering the backbone.

Recommended short-run command:

```bash
python src/train_unet2d_qata.py --data_dir data/processed/qata --model transunet2d_v17 --epochs 8 --batch 16 --lr 1e-5 --loss prior_calibration_combo --boundary_weight 0.05 --warmup_epochs 1 --grad_clip 1.0 --augment --ema_decay 0.999 --select_metric best_threshold --resume outputs/transunet2d_v14_aug_ema_ft20_best.pt --resume_partial --run_tag transunet2d_v17_calib_ft8
```

## v18 Frequency-Prior TransUNet

v18 keeps the stable v12/v14 segmentation path and injects a multi-frequency lesion prior with zero-initialized residual adapters:

- The frequency prior combines grayscale intensity, low-frequency context, high-frequency residuals, Sobel edge magnitude, and Laplacian response.
- Prior features are injected at four encoder scales through residual adapters.
- Each adapter starts with `gamma=0`, so warm-starting from v14 begins close to the original backbone.
- `frequency_prior_combo` uses weak prior supervision to avoid the prior branch overpowering segmentation learning.

Recommended short-run command:

```bash
python src/train_unet2d_qata.py --data_dir data/processed/qata --model transunet2d_v18 --epochs 8 --batch 16 --lr 1e-5 --loss frequency_prior_combo --boundary_weight 0.05 --warmup_epochs 1 --grad_clip 1.0 --augment --ema_decay 0.999 --select_metric best_threshold --resume outputs/transunet2d_v14_aug_ema_ft20_best.pt --resume_partial --run_tag transunet2d_v18_freq_ft8
```

## v14 Robust Short-Run Training

v14 keeps the stable v12 architecture and focuses on generalization: safe CXR augmentation, EMA weights, and checkpoint selection by validation threshold sweep.

```bash
python src/train_unet2d_qata.py --data_dir data/processed/qata --model transunet2d_v14 --epochs 20 --batch 16 --lr 2e-5 --loss tversky_bce --warmup_epochs 2 --grad_clip 1.0 --augment --ema_decay 0.999 --select_metric best_threshold --resume outputs/transunet2d_v12_tversky_ft_best.pt --run_tag transunet2d_v14_aug_ema_ft20
```

After training, use the checkpoint's logged `val_best_threshold`, or run:

```bash
python scripts/eval_qata_thresholds.py --data_dir data/processed/qata --model transunet2d_v14 --ckpt outputs/transunet2d_v14_aug_ema_ft20_best.pt --split val
```

## v13 Short-Run Training

For a 20-epoch fine-tune, use the v13 model with the new boundary-combo loss and warm-start from the existing v12 checkpoint:

```bash
python src/train_unet2d_qata.py --data_dir data/processed/qata --model transunet2d_v13 --epochs 20 --batch 16 --lr 1e-4 --loss boundary_combo --boundary_weight 0.05 --warmup_epochs 2 --grad_clip 1.0 --resume outputs/transunet2d_v12_deep_boundary_ft_best.pt --resume_partial --run_tag transunet2d_v13_ft20
```

Then search a better threshold on the validation split:

```bash
python scripts/eval_qata_thresholds.py --data_dir data/processed/qata --model transunet2d_v13 --ckpt outputs/transunet2d_v13_ft20_best.pt --split val
```

## Local GUI

You can launch a local desktop window for single-image segmentation:

```bash
python app_v12_gui.py
```

The GUI loads a chest image and previews the predicted infection mask and overlay. The default path is `configs/v15_ensemble_config.json`; you can still manually choose a v12/v13/v14 checkpoint.

## Local Web App

You can also launch a browser-based local web app without installing Flask or FastAPI:

```bash
python app_v12_web.py
```

Then open `http://127.0.0.1:8000` in your browser.
If a ground-truth mask is uploaded or can be matched automatically from the local QaTa dataset, the page will also show Dice/IoU/Precision/Recall and an error overlay. The web app defaults to the v15 ensemble config and automatically fills its threshold/min-area/TTA defaults.

## Presentation Materials

The project includes two updated presentation versions for the v18-aware / LowDice expert strategy:

- `docs/covid_screening_project_v18aware_defense_code_redone.pptx`: defense version with a concise code walkthrough for presentation.
- `docs/covid_screening_project_v18aware_study_code_walkthrough.pptx`: study version with terminology, file responsibilities, call-chain explanation, and line-by-line code notes.
- `docs/covid_screening_project_v18aware_detailed_v4.pptx`: detailed base version used before rebuilding the final code section.

The PowerPoint generation helper is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_ppt_code_redone_versions.ps1
```

## Project Structure

```text
covid_screening
├── docs/                 # Thesis drafts and generated Word documents
├── scripts/              # Preprocessing and evaluation scripts
├── src/                  # Datasets, models, training, and inference code
├── environment.yml       # Conda environment definition
└── .gitignore            # Excludes datasets, checkpoints, outputs, and local files
```

## Notes

Large datasets, model checkpoints, generated outputs, and local Kaggle configuration files are intentionally excluded from Git. Prepare the QaTa-COV19 dataset locally before running training or evaluation scripts.
