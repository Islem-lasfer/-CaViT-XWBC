"""Boucle d'une epoch (train ou validation)."""

import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score


def run_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
              train=True, grad_accum_steps=1, grad_clip_norm=1.0, mixed_precision=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []
    scale = grad_accum_steps if train else 1

    if train:
        optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="train" if train else "val ", leave=False)

    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda",
                                enabled=mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y) / scale

            if train:
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        total_loss += loss.item() * scale
        n_batches += 1
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    avg_loss = total_loss / max(1, n_batches)
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro, all_targets, all_preds
