import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
CKPT_A   = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_effb3_daft_cutmix/best_model.pth"
CKPT_B   = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_effb3_daft_mixup_v2/best_model.pth"

TEST_CSV = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/test_split.csv"
IMAGE_ROOT = r"C:/Users/Ata/midasmultimodalimagedatasetforaibasedskincancer"
OUTPUT_DIR = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run14_ensemble_run2_run4v2"

IMG_SIZE      = 300
N_AUGMENTS    = 8
MIN_PRECISION = 0.40
SEED          = 42

WEIGHT_A = 0.5
WEIGHT_B = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

_IMAGE_INDEX: dict = {}

def _build_image_index():
    global _IMAGE_INDEX
    if _IMAGE_INDEX:
        return
    print(f"  Indexing images under: {IMAGE_ROOT}")
    for root, _, files in os.walk(IMAGE_ROOT):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                _IMAGE_INDEX[fname.lower()] = os.path.join(root, fname)
    print(f"  Index built: {len(_IMAGE_INDEX)} images")

def resolve_image_path(raw_path: str) -> str:
    _build_image_index()
    if os.path.isfile(raw_path):
        return raw_path
    fname = os.path.basename(str(raw_path))
    candidate = os.path.join(IMAGE_ROOT, fname)
    if os.path.isfile(candidate):
        return candidate
    return _IMAGE_INDEX.get(fname.lower(), "")


def load_test_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "label_encoded" in df.columns:
        df["label"] = df["label_encoded"].astype(int)
    elif "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
    elif "midas_melanoma" in df.columns:
        df["label"] = (df["midas_melanoma"].astype(str).str.strip().str.lower() == "yes").astype(int)
    else:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")
    path_col = next(
        (c for c in ["image_path", "abs_path", "path"] if c in df.columns), None
    )
    if path_col is None:
        raise ValueError(f"No image path column. Columns: {df.columns.tolist()}")
    df["abs_path"] = df[path_col].apply(lambda p: resolve_image_path(str(p)))
    df = df[df["abs_path"] != ""].reset_index(drop=True)
    print(f"  Test set: {len(df)} samples | Melanoma: {df['label'].sum()} | Non-mel: {(df['label']==0).sum()}")
    return df

def build_tabular_features(df: pd.DataFrame):
    cat_cols = ["midas_gender", "midas_fitzpatrick", "midas_location",
                "midas_ethnicity", "midas_race",
                "clinical_impression_1", "clinical_impression_2", "clinical_impression_3"]
    num_cols = ["midas_age", "midas_distance", "length_(mm)", "width_(mm)"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    feat_df  = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        le = LabelEncoder()
        feat_df[c] = le.fit_transform(feat_df[c].astype(str).fillna("unknown"))
    feat_df[num_cols] = feat_df[num_cols].fillna(feat_df[num_cols].median())
    X = feat_df.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float32), X.shape[1]

class DAFTBlock(nn.Module):
    def __init__(self, tab_dim: int, num_channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature_map, tab):
        params = self.mlp(tab)
        gamma, beta = params.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * 0.5
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * feature_map + beta


class EfficientNetDAFT(nn.Module):
    def __init__(self, tab_dim: int, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        feat_dim       = backbone.classifier[1].in_features
        self.daft      = DAFTBlock(tab_dim=tab_dim, num_channels=feat_dim, hidden=128)
        self.tab_mlp   = nn.Sequential(
            nn.Linear(tab_dim, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Dropout(0.3), nn.Linear(64, 32), nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + 32, 256),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, tab):
        tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        x   = self.features(img)
        x   = self.daft(x, tab)
        x   = self.avgpool(x).flatten(1)
        t   = self.tab_mlp(tab)
        return self.classifier(torch.cat([x, t], dim=1))


def load_model(ckpt_path: str, tab_dim: int) -> EfficientNetDAFT:
    model = EfficientNetDAFT(tab_dim=tab_dim).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_auc = ckpt.get("val_auc", "?")
    ep      = ckpt.get("epoch", "?")
    auc_str = f"{val_auc:.4f}" if isinstance(val_auc, float) else str(val_auc)
    print(f"  Loaded checkpoint: epoch {ep}, val AUC {auc_str}")
    return model


def get_tta_transforms(n: int = 8):
    clean = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    aug = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return [clean] + [aug] * (n - 1)


@torch.no_grad()
def run_tta(model: EfficientNetDAFT, test_df: pd.DataFrame,
            tabular: np.ndarray, n: int = N_AUGMENTS) -> np.ndarray:
    model.eval()
    tfms   = get_tta_transforms(n)
    probs  = []
    for idx in range(len(test_df)):
        path = test_df["abs_path"].iloc[idx]
        tab  = torch.tensor(tabular[idx]).unsqueeze(0).to(DEVICE)
        try:
            img_pil = Image.open(path).convert("RGB")
        except Exception:
            probs.append(0.5)
            continue
        view_probs = []
        for tfm in tfms:
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
            logit = model(img_t, tab)
            p     = F.softmax(logit, dim=1)[0, 1].item()
            view_probs.append(p)
        probs.append(float(np.mean(view_probs)))
    return np.array(probs, dtype=np.float32)


def find_best_threshold_fine(probs: np.ndarray, labels: np.ndarray,
                             min_precision: float = MIN_PRECISION):
    best_t  = 0.5
    best_r  = 0.0
    results = []
    for t in np.arange(0.05, 0.96, 0.005):
        t  = round(float(t), 4)
        pr = (probs >= t).astype(int)
        if pr.sum() == 0:
            continue
        p = precision_score(labels, pr, zero_division=0)
        r = recall_score(labels, pr, zero_division=0)
        results.append((t, p, r))
        if p >= min_precision and r > best_r:
            best_r = r
            best_t = t
    if best_r == 0.0:
        for floor in [min_precision - 0.01, min_precision - 0.02]:
            for t, p, r in results:
                if p >= floor and r > best_r:
                    best_r = r; best_t = t
            if best_r > 0:
                break
    return best_t, results


def plot_thresh_curve(results, best_t: float, save_path: str, title: str):
    ts  = [r[0] for r in results]
    ps  = [r[1] for r in results]
    rs  = [r[2] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, ps, label="Precision", color="#2196F3")
    ax.plot(ts, rs, label="Recall",    color="#F44336")
    ax.axvline(best_t, color="#4CAF50", linestyle="--",
               label=f"Threshold = {best_t:.3f}")
    ax.axhline(MIN_PRECISION, color="grey", linestyle=":", alpha=0.6,
               label=f"Min precision = {MIN_PRECISION}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def result_block(tag: str, probs: np.ndarray, labels: np.ndarray) -> tuple:
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    t, results = find_best_threshold_fine(probs, labels)
    preds = (probs >= t).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)
    s = (
        f"--- {tag} ---\n"
        f"AUC:       {auc:.4f}\n"
        f"F1:        {f1:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"Threshold: {t:.3f}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"
        f"{classification_report(labels, preds, target_names=['no melanoma','melanoma'])}\n"
    )
    return s, t, results, auc, rec, cm[1,1]


def main():
    print("\n" + "="*60)
    print("ENSEMBLE: Run 2 + Run 4 v2")
    print("="*60)

    print("\nLoading test set...")
    test_df = load_test_csv(TEST_CSV)
    X_test, tab_dim = build_tabular_features(test_df)
    labels  = test_df["label"].values.astype(np.int64)
    print(f"Tabular feature dim: {tab_dim}")

    print(f"\nLoading Model A (run 2): {CKPT_A}")
    model_a = load_model(CKPT_A, tab_dim)
    print(f"\nLoading Model B (run 4 v2): {CKPT_B}")
    model_b = load_model(CKPT_B, tab_dim)

    print(f"\nRunning TTA ({N_AUGMENTS} views) — Model A...")
    probs_a = run_tta(model_a, test_df, X_test)
    print(f"Running TTA ({N_AUGMENTS} views) — Model B...")
    probs_b = run_tta(model_b, test_df, X_test)

    print(f"\nEnsembling: {WEIGHT_A:.0%} × A + {WEIGHT_B:.0%} × B")
    probs_ens = WEIGHT_A * probs_a + WEIGHT_B * probs_b

    print("\nRunning fine threshold search on all three...")
    block_a,   t_a,   res_a,   auc_a,   rec_a,   tp_a   = result_block("Model A — Run 2 (TTA)",    probs_a,   labels)
    block_b,   t_b,   res_b,   auc_b,   rec_b,   tp_b   = result_block("Model B — Run 4 v2 (TTA)", probs_b,   labels)
    block_ens, t_ens, res_ens, auc_ens, rec_ens, tp_ens = result_block("Ensemble A+B (TTA)",       probs_ens, labels)

    best_single_auc = max(auc_a, auc_b)
    best_single_tp  = max(tp_a, tp_b)
    delta_auc = auc_ens - best_single_auc
    delta_tp  = tp_ens  - best_single_tp

    report = (
        "===== ENSEMBLE TEST RESULTS =====\n\n"
        f"Models:\n"
        f"  A = Run 2  checkpoint : {CKPT_A}\n"
        f"  B = Run 4v2 checkpoint: {CKPT_B}\n"
        f"  Weights: A={WEIGHT_A}, B={WEIGHT_B}\n"
        f"  TTA views: {N_AUGMENTS}\n\n"
        + block_a
        + block_b
        + block_ens
        + f"--- Delta (ensemble vs best single) ---\n"
        f"AUC:    {delta_auc:+.4f}\n"
        f"TP caught: {tp_ens} vs {best_single_tp} ({delta_tp:+d})\n"
    )

    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
        f.write(report)

    plot_thresh_curve(res_a,   t_a,   os.path.join(OUTPUT_DIR, "threshold_curve_modelA.png"),
                      "Model A — Run 2")
    plot_thresh_curve(res_b,   t_b,   os.path.join(OUTPUT_DIR, "threshold_curve_modelB.png"),
                      "Model B — Run 4 v2")
    plot_thresh_curve(res_ens, t_ens, os.path.join(OUTPUT_DIR, "threshold_curve_ensemble.png"),
                      "Ensemble (A + B averaged)")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
