"""Inférence sur une image avec un checkpoint entraîné.

Exemple :
    python predict.py --checkpoint outputs/checkpoints/best.pt --image cellule.png --top_k 5
"""

import argparse

import torch
from PIL import Image

from model import build_model
from dataloader import get_transforms


def get_args():
    p = argparse.ArgumentParser(description="CytologIA - inférence")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--top_k", type=int, default=5)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)

    label_mapping = ck["label_mapping"]
    inv = {v: k for k, v in label_mapping.items()}
    img_size = ck.get("img_size", 224)

    model = build_model(num_classes=len(label_mapping), img_size=img_size,
                        **ck.get("model_kwargs", {})).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    x = get_transforms(img_size, train=False)(Image.open(args.image).convert("RGB"))
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0]
        top = torch.topk(probs, k=min(args.top_k, probs.numel()))
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {inv[i]:30s} {p * 100:.2f}%")


if __name__ == "__main__":
    main()
