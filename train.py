"""Entraînement de HybridNet sur CytologIA.

Exemple :
    python train.py --csv_path data/labels.csv --images_dir data/images \
                    --output_dir outputs --epochs 60 --batch_size 32
"""

import os
import json
import time
import argparse

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from model import build_model
from dataloader import make_train_val_datasets, make_dataloaders
from engine import run_epoch
from utils import set_seed, get_lr_scheduler, compute_class_weights


def get_args():
    p = argparse.ArgumentParser(
        description="CytologIA - HybridNet (CNN + ViT, fusion cross-attention)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Chemins
    p.add_argument("--csv_path", type=str, required=True, help="CSV (NAME,x1,y1,x2,y2,class)")
    p.add_argument("--images_dir", type=str, required=True, help="Dossier des images")
    p.add_argument("--output_dir", type=str, default="outputs", help="Dossier de sortie")

    # Données
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--bbox_margin", type=float, default=0.10)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 recommandé sur Windows (évite les deadlocks)")

    # Modèle
    p.add_argument("--cnn_deep_ch", type=int, default=512)
    p.add_argument("--vit_embed_dim", type=int, default=384)
    p.add_argument("--vit_depth", type=int, default=6)
    p.add_argument("--vit_heads", type=int, default=8)
    p.add_argument("--vit_dropout", type=float, default=0.1)
    p.add_argument("--fusion_dropout", type=float, default=0.1)
    p.add_argument("--head_drop1", type=float, default=0.4)
    p.add_argument("--head_drop2", type=float, default=0.3)

    # Entraînement
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--no_class_weights", action="store_true", help="Désactive la pondération des classes")
    p.add_argument("--patience", type=int, default=12, help="Early stopping (macro-F1 val)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true", help="Désactive la précision mixte")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_ds, val_ds, label_mapping = make_train_val_datasets(
        args.csv_path, args.images_dir, img_size=args.img_size, val_ratio=args.val_ratio,
        bbox_margin=args.bbox_margin, seed=args.seed,
        label_mapping_path=os.path.join(args.output_dir, "label_mapping.json"),
    )
    print(f"Train : {len(train_ds)} | Val : {len(val_ds)} | Classes : {len(label_mapping)}")
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, args.batch_size, args.num_workers)

    model_kwargs = dict(
        cnn_deep_ch=args.cnn_deep_ch, vit_embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth, vit_heads=args.vit_heads,
        vit_dropout=args.vit_dropout, fusion_dropout=args.fusion_dropout,
        head_drop1=args.head_drop1, head_drop2=args.head_drop2,
    )
    model = build_model(num_classes=len(label_mapping), img_size=args.img_size,
                        **model_kwargs).to(device)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    weights = None if args.no_class_weights else compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def checkpoint(epoch, val_f1, with_optim=False):
        ck = {"epoch": epoch, "model_state": model.state_dict(), "label_mapping": label_mapping,
              "val_f1": val_f1, "model_kwargs": model_kwargs, "img_size": args.img_size}
        if with_optim:
            ck["optimizer_state"] = optimizer.state_dict()
        return ck

    best_f1, patience_counter, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            train=True, grad_accum_steps=args.grad_accum_steps,
            grad_clip_norm=args.grad_clip, mixed_precision=use_amp)
        va_loss, va_acc, va_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, None, scaler, device,
            train=False, mixed_precision=use_amp)
        lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
              f"lr={lr:.2e} | {time.time() - t0:.1f}s")

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1, lr=lr))
        with open(os.path.join(log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        torch.save(checkpoint(epoch, va_f1, with_optim=True), os.path.join(ckpt_dir, "last.pt"))
        if va_f1 > best_f1:
            best_f1, patience_counter = va_f1, 0
            torch.save(checkpoint(epoch, va_f1), os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> Nouveau meilleur modèle (val_f1={va_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping après {epoch} epochs.")
                break

    # Rapport final sur le MEILLEUR checkpoint (et non la dernière epoch)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)["model_state"])
    _, _, f1, targets, preds = run_epoch(model, val_loader, criterion, None, None, scaler, device,
                                         train=False, mixed_precision=use_amp)
    inv = {v: k for k, v in label_mapping.items()}
    labels = list(range(len(inv)))
    report = classification_report(targets, preds, labels=labels,
                                   target_names=[inv[i] for i in labels], zero_division=0)
    print("\nRapport de classification (best.pt) :\n", report)
    with open(os.path.join(log_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Meilleur F1 macro validation : {best_f1:.4f}\nCheckpoints : {ckpt_dir}")


if __name__ == "__main__":  # obligatoire sous Windows
    main()
