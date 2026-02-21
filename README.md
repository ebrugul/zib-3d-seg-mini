# zib-3d-seg-mini

End-to-end 3D medical image segmentation pipeline using MONAI and PyTorch.

Implements a 3D U-Net trained on the Medical Segmentation Decathlon Task09 (Spleen) dataset, including volumetric preprocessing, patch-based training, sliding-window inference, and Dice-based evaluation.

## Tech Stack
- PyTorch
- MONAI
- 3D U-Net
- Dice loss
- Sliding window inference

## Run

```bash
conda create -n zib3d python=3.11 -y
conda activate zib3d
python -m pip install torch monai nibabel matplotlib
python -u src/train.py
```

Dataset downloads automatically on first run.
