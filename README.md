# CytologIA — HybridNet (CNN + ViT)

PyTorch implementation of **HybridNet**, a hybrid architecture for cytological cell classification (23 classes). It combines:

- a **CNN branch**: multi-scale Inception block, dilated convolutions, CBAM attention, and residual blocks with Squeeze-and-Excitation;
- a **ViT branch**: patch embedding followed by 6 Transformer encoder blocks;
- a **cross-attention fusion** module, where the CNN features query the ViT patch tokens, followed by an MLP classification head.

## Repository structure

```
├── model.py        # HybridNet and all its building blocks
├── dataloader.py   # Dataset (bbox cropping), stratified split, DataLoaders
├── engine.py       # One train / validation epoch
├── utils.py        # Seeding, warm-up + cosine scheduler, class weights
├── train.py        # Training script (command-line arguments)
├── predict.py      # Single-image inference
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended. Mixed precision (AMP) is enabled by default on GPU.

## Data format

Annotations are provided as a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `NAME` | Image file name (extension optional) |
| `x1, y1, x2, y2` | Bounding box of the cell (pixels) |
| `class` | Class label |

Each cell is cropped from its image using the bounding box (plus a configurable margin) and resized to `img_size × img_size`. Rows with invalid boxes (`x2 <= x1` or `y2 <= y1`) are ignored.

## Training

```bash
python train.py \
    --csv_path data/labels.csv \
    --images_dir data/images \
    --output_dir outputs \
    --epochs 60 --batch_size 32 --lr 3e-4
```

Run `python train.py --help` to list all arguments. The main ones are:

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_size` | 224 | Input resolution (must be divisible by 16) |
| `--bbox_margin` | 0.10 | Relative margin added around each bounding box |
| `--val_ratio` | 0.15 | Fraction of data used for validation (stratified) |
| `--epochs` | 60 | Maximum number of epochs |
| `--batch_size` | 32 | Batch size |
| `--grad_accum_steps` | 1 | Gradient accumulation steps |
| `--lr` / `--weight_decay` | 3e-4 / 0.05 | AdamW hyperparameters |
| `--warmup_epochs` | 5 | Linear warm-up, then cosine decay |
| `--label_smoothing` | 0.1 | Label smoothing in the cross-entropy loss |
| `--patience` | 12 | Early stopping patience (validation macro-F1) |
| `--no_class_weights` | off | Disable class-balanced loss weights |
| `--no_amp` | off | Disable mixed precision |
| `--num_workers` | 0 | DataLoader workers |
| `--seed` | 42 | Random seed |

**Windows note:** keep `--num_workers 0` (default) if the DataLoader hangs. With `--num_workers > 0`, the `spawn` multiprocessing context is used automatically, and the script entry point is protected by `if __name__ == "__main__"`.

### Outputs

Everything is written to `--output_dir`:

```
outputs/
├── args.json                     # arguments used for the run
├── label_mapping.json            # class name -> index
├── checkpoints/
│   ├── best.pt                   # best validation macro-F1
│   └── last.pt                   # last epoch (with optimizer state)
└── logs/
    ├── history.json              # per-epoch losses and metrics
    └── classification_report.txt # per-class report computed with best.pt
```

## Inference

```bash
python predict.py --checkpoint outputs/checkpoints/best.pt --image example.png --top_k 5
```

The checkpoint stores the label mapping and model hyperparameters, so no extra configuration is needed.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{your_key,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## License

Add a license of your choice (e.g. MIT) in a `LICENSE` file.
