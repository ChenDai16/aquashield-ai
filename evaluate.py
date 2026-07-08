# -*- coding: utf-8 -*-
"""
evaluate.py — Đánh giá lại AquaShield AI trên tập Test chính thức.
Tạo: classification report, confusion matrix (PNG + CSV), accuracy.

CÁCH CHẠY:
    python evaluate.py --data "ShrimpDiseaseDataset/Tiger Prawn Diseases/Test" \
                       --weights model/resnet18_transfer.pt --out results

Kết quả tham chiếu (đã kiểm định): 98.05% trên 256 ảnh (64 ảnh × 4 lớp).
"""
import os, glob, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision
from torchvision import transforms

CLASS_NAMES = ["BG (Đen mang)", "BG+WSSV", "HS (Khỏe)", "WSSV (Đốm trắng)"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_model(weights):
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    ck = torch.load(weights, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    m.load_state_dict(state, strict=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ShrimpDiseaseDataset/Tiger Prawn Diseases/Test")
    ap.add_argument("--weights", default="model/resnet18_transfer.pt")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights).to(dev)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    classes = sorted(os.listdir(args.data))
    print("Lớp (thứ tự alphabet):", classes)

    paths, ys = [], []
    for i, c in enumerate(classes):
        for p in sorted(glob.glob(os.path.join(args.data, c, "*"))):
            paths.append(p); ys.append(i)
    ys = np.array(ys)
    preds = np.zeros(len(paths), int)

    with torch.no_grad():
        B = 32
        for s in range(0, len(paths), B):
            xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s+B]]).to(dev)
            preds[s:s+B] = model(xb).argmax(1).cpu().numpy()

    acc = (preds == ys).mean()
    K = 4
    cm = np.zeros((K, K), int)
    for t, p in zip(ys, preds):
        cm[t, p] += 1

    # report
    lines = [f"Test accuracy: {acc*100:.2f}%  (n={len(ys)})\n",
             f"{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}"]
    Ps, Rs, Fs, Ss = [], [], [], []
    for k in range(K):
        tp = cm[k, k]; fp = cm[:, k].sum()-tp; fn = cm[k, :].sum()-tp
        p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
        f = 2*p*r/(p+r) if p+r else 0; sup = cm[k, :].sum()
        Ps.append(p); Rs.append(r); Fs.append(f); Ss.append(sup)
        lines.append(f"{CLASS_NAMES[k]:<20}{p:>10.3f}{r:>10.3f}{f:>10.3f}{sup:>9d}")
    w = np.array(Ss)/sum(Ss)
    lines += ["",
              f"{'accuracy':<20}{'':>10}{'':>10}{acc:>10.3f}{len(ys):>9d}",
              f"{'macro avg':<20}{np.mean(Ps):>10.3f}{np.mean(Rs):>10.3f}{np.mean(Fs):>10.3f}{len(ys):>9d}",
              f"{'weighted avg':<20}{np.average(Ps,weights=w):>10.3f}"
              f"{np.average(Rs,weights=w):>10.3f}{np.average(Fs,weights=w):>10.3f}{len(ys):>9d}"]
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(args.out, "classification_report_full.txt"), "w", encoding="utf-8").write(report + "\n")
    np.savetxt(os.path.join(args.out, "confusion_matrix_full.csv"), cm, fmt="%d",
               delimiter=",", header=",".join(CLASS_NAMES), comments="")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=140)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        ax.set_title(f"Confusion Matrix — acc={acc*100:.2f}% (n={len(ys)})", fontweight="bold")
        th = cm.max()/2
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > th else "#1a3a5c", fontweight="bold")
        plt.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "confusion_matrix_full.png"), bbox_inches="tight")
        print("Đã lưu confusion_matrix_full.png")
    except Exception as e:
        print("Bỏ qua vẽ hình:", e)


if __name__ == "__main__":
    main()
