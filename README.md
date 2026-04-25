# COVID Screening

COVID-19 chest image lesion segmentation project based on the QaTa-COV19 dataset.

The project includes preprocessing, model training, inference, evaluation, and thesis documentation. It implements an improved TransUNet2D v11 model with a shifted-window Transformer bottleneck, attention-gated skip connections, DropPath regularization, and Sobel-guided boundary refinement. It also compares and ensembles models such as U-Net, Attention U-Net, Swin-Unet, and UCTRansNet.

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
