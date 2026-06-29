# Exploring Transfer Learning on Oxford-IIIT Pets

Course project for **DD2424 Deep Learning in Data Science** at KTH Royal Institute of Technology.

This repository contains the code notebooks for an image-classification study on the Oxford-IIIT Pet dataset. The project compares several transfer-learning approaches for binary cat/dog classification and 37-class breed classification:

- ResNet18 fine-tuning from ImageNet weights.
- Simultaneous and gradual unfreezing strategies.
- Semi-supervised pseudo-labeling with limited labeled data.
- Vision Transformer fine-tuning.
- LoRA-style parameter-efficient adaptation for ResNet layers.

## Repository Map

| Path | Purpose |
| --- | --- |
| `grade_E.ipynb` | Baseline ResNet18 experiments, binary classification, multi-class fine-tuning, unfreezing strategies, and class-imbalance tests. |
| `grade_A.ipynb` | Extended experiments with pseudo-labeling, Vision Transformers, and LoRA adaptation. |

## Selected Results

| Experiment | Result |
| --- | --- |
| Binary cat/dog classification with ResNet18 | 99.02% test accuracy |
| Multi-class ResNet18 fine-tuning | 90.54% test accuracy for the selected Strategy 1 baseline |
| Class-imbalance mitigation | 86.86% to 87.68% test accuracy with weighted loss and over-sampling |
| Pseudo-labeling with 1% labeled data | 42.90% supervised to 51.51% semi-supervised |
| Pseudo-labeling with 10% labeled data | 77.81% supervised to 81.44% semi-supervised |
| Vision Transformer fine-tuning | 92.34% test accuracy |
| ResNet18 with LoRA adapters | 89.26% test accuracy with 71,680 additional trainable parameters |
| ResNet50 with LoRA adapters | 91.77% test accuracy with 143,360 additional trainable parameters |

## Report

For the full methodology, experiments, and discussion, see the [group project report](docs/project-report.pdf).

## Setup

Create an environment and install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The notebooks download the Oxford-IIIT Pet dataset through `torchvision` into `./dataset`. The report experiments were run on a Google Cloud instance with an NVIDIA T4 GPU.

## Running

Open the notebooks with Jupyter:

```bash
jupyter notebook grade_E.ipynb
jupyter notebook grade_A.ipynb
```

Use a CUDA-capable environment for the training notebooks. CPU execution works for inspection but is slow for the full experiments.

## Notes

The notebooks keep selected outputs so the reported results can be inspected without rerunning every experiment. Downloaded datasets and model checkpoints are excluded from version control.

## Contributors

| Contributor | Contact |
| --- | --- |
| Diogo Paulo | `diogop@kth.se` |
| Hugo Dezerto | `hugoad@kth.se` |
| Maria Sebastião | `mcms2@kth.se` |
