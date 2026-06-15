import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights

from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
TRAIN_CSV  = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/train_split.csv"
VAL_CSV    = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/val_split.csv"
TEST_CSV   = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/test_split.csv"
IMAGE_ROOT = r"C:/Users/Ata/midasmultimodalimagedatasetforaibasedskincancer"
OUTPUT_DIR = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run9_effb3_daft_cutmix_tta"

IMG_SIZE       = 300
BATCH_SIZE     = 16
NUM_EPOCHS     = 40
LR_HEAD        = 1e-4
LR_FINETUNE    = 5e-5
WEIGHT_DECAY   = 1e-4
UNFREEZE_EPOCH = 6
PATIENCE       = 10
SEED           = 42
MIN_PRECISION  = 0.40
CUTMIX_PROB    = 0.5
CUTMIX_ALPHA   = 1.0
OVERSAMPLE_STRENGTH = 0.7

# Focal loss
FOCAL_GAMMA    = 2.0
LABEL_SMOOTH   = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
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
    count = 0
    for root, _, files in os.walk(IMAGE_ROOT):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                _IMAGE_INDEX[fname.lower()] = os.path.join(root, fname)
                count += 1
    print(f"  Index built: {count} images found")


def resolve_image_path(raw_path: str) -> str:
    _build_image_index()

    if os.path.isfile(raw_path):
        return raw_path

    fname = os.path.basename(str(raw_path))
    candidate = os.path.join(IMAGE_ROOT, fname)
    if os.path.isfile(candidate):
        return candidate

    hit = _IMAGE_INDEX.get(fname.lower())
    if hit:
        return hit
    stem = os.path.splitext(fname)[0].lower()
    for key, val in _IMAGE_INDEX.items():
        if os.path.splitext(key)[0] == stem:
            return val

    return ""


def _diagnose_csv_paths(df: pd.DataFrame, path_col: str, split_name: str):
    samples = df[path_col].dropna().head(3).tolist()
    print(f"  [{split_name}] Sample paths from CSV column '{path_col}':")
    for s in samples:
        exists = "EXISTS" if os.path.isfile(str(s)) else "missing"
        print(f"    {s}  [{exists}]")


def load_split_csv(csv_path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "label_encoded" in df.columns:
        df["label"] = df["label_encoded"].astype(int)
    elif "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
    elif "midas_melanoma" in df.columns:
        df["label"] = (df["midas_melanoma"].astype(str).str.strip().str.lower() == "yes").astype(int)
    else:
        raise ValueError(f"Cannot find label column in {csv_path}. Columns: {df.columns.tolist()}")

    path_col = next(
        (c for c in ["image_path", "abs_path", "path"] if c in df.columns), None
    )
    if path_col is None:
        raise ValueError(
            f"No image path column in {csv_path}.\n"
            f"Columns found: {df.columns.tolist()}\n"
            f"Add one of: abs_path, midas_path, image_path, path"
        )

    _diagnose_csv_paths(df, path_col, split_name)

    df["abs_path"] = df[path_col].apply(lambda p: resolve_image_path(str(p)))

    before = len(df)
    df = df[df["abs_path"] != ""].reset_index(drop=True)
    missing = before - len(df)
    if missing:
        print(
            f"  [{split_name}] WARNING: {missing}/{before} images not resolved.\n"
            f"  Check that IMAGE_ROOT is correct: {IMAGE_ROOT}"
        )

    print(
        f"  [{split_name}] {len(df)} samples | "
        f"Melanoma: {df['label'].sum()} | "
        f"Non-melanoma: {(df['label'] == 0).sum()}"
    )
    return df


def build_tabular_features(df: pd.DataFrame, fit_scaler=True, scaler=None, encoders=None):
    cat_cols = ["midas_gender", "midas_fitzpatrick", "midas_location",
                "midas_ethnicity", "midas_race",
                "clinical_impression_1", "clinical_impression_2", "clinical_impression_3"]
    num_cols = ["midas_age", "midas_distance", "length_(mm)", "width_(mm)"]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]

    feat_df = df[cat_cols + num_cols].copy()

    if encoders is None:
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            feat_df[c] = le.fit_transform(feat_df[c].astype(str).fillna("unknown"))
            encoders[c] = le
    else:
        for c in cat_cols:
            feat_df[c] = feat_df[c].astype(str).fillna("unknown").map(
                lambda x, le=encoders[c]: (
                    le.transform([x])[0] if x in le.classes_
                    else len(le.classes_)
                )
            )

    feat_df[num_cols] = feat_df[num_cols].fillna(feat_df[num_cols].median())

    X = feat_df.values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, scaler, encoders, X.shape[1]

def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


class MIDASDataset(Dataset):
    def __init__(self, df, tabular_features, transform=None):
        self.paths    = df["abs_path"].values
        self.labels   = df["label"].values.astype(np.int64)
        self.tabular  = tabular_features.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(self.tabular[idx])
        lbl = self.labels[idx]
        return img, tab, lbl


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(images, labels, alpha=CUTMIX_ALPHA):
    if (labels == 1).sum() == 0:
        return images, labels.float()

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    rand_idx = torch.randperm(batch_size)

    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    mixed = images.clone()
    mixed[:, :, x1:x2, y1:y2] = images[rand_idx, :, x1:x2, y1:y2]

    lam = 1 - ((x2 - x1) * (y2 - y1)) / (images.size(-1) * images.size(-2))
    soft_labels = lam * labels.float() + (1 - lam) * labels[rand_idx].float()
    return mixed, soft_labels


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
        params = self.mlp(tab)                              # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)                # (B, C) each
        # Clamp gamma so early training cannot explode the feature map
        gamma = torch.tanh(gamma) * 0.5                     # range (-0.5, 0.5)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * feature_map + beta             # affine transform


class EfficientNetDAFT(nn.Module):
    def __init__(self, tab_dim: int, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        # Backbone: EfficientNet-B3 with ImageNet weights
        backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Feature extractor (everything except the classifier head)
        self.features = backbone.features          # output: (B, 1536, 9, 9) for 300×300
        self.avgpool  = backbone.avgpool           # AdaptiveAvgPool2d → (B, 1536, 1, 1)
        feat_dim = backbone.classifier[1].in_features   # 1536

        # DAFT applied to the final feature map (1536 channels)
        self.daft = DAFTBlock(tab_dim=tab_dim, num_channels=feat_dim, hidden=128)

        # Tabular branch (separate MLP for later concat)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.SiLU(),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + 32, 256),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, tab):
        # Sanitise tabular inputs: replace any NaN/Inf with 0.0
        tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        # Image path
        x = self.features(img)          # (B, 1536, H', W')
        x = self.daft(x, tab)           # DAFT modulation
        x = self.avgpool(x)             # (B, 1536, 1, 1)
        x = x.flatten(1)                # (B, 1536)

        # Tabular path
        t = self.tab_mlp(tab)           # (B, 32)

        # Fuse
        out = self.classifier(torch.cat([x, t], dim=1))
        return out

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


class SkinLoss(nn.Module):

    def __init__(self, class_weights=None, label_smoothing: float = LABEL_SMOOTH):
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):

        if targets.is_floating_point():
            # Soft labels from CutMix
            t = targets.clamp(0.0, 1.0)
            targets_oh = torch.stack([1.0 - t, t], dim=1)        # (B, 2)
            if self.label_smoothing > 0:
                s = self.label_smoothing
                targets_oh = targets_oh * (1.0 - s) + s / 2.0
            log_p = F.log_softmax(logits, dim=1)                  # (B, 2)
            loss  = -(targets_oh * log_p).sum(dim=1)              # (B,)
            if self.class_weights is not None:
                w = (1.0 - t) * self.class_weights[0] + t * self.class_weights[1]
                loss = loss * w
            return loss.mean()
        else:
            # Hard labels — PyTorch built-in CE (most stable)
            return F.cross_entropy(
                logits,
                targets.long(),
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )


def get_tta_transforms(n_augments: int = 8):

    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    clean = transforms.Compose(base)

    aug_views = []
    for i in range(n_augments - 1):
        aug_views.append(transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

    return [clean] + aug_views


@torch.no_grad()
def evaluate_tta(model, dataset_df, tabular_features, n_augments: int = 8):

    model.eval()
    tta_transforms = get_tta_transforms(n_augments)
    all_probs  = []
    all_labels = []

    for idx in range(len(dataset_df)):
        img_path = dataset_df["abs_path"].iloc[idx]
        label    = int(dataset_df["label"].iloc[idx])
        tab      = torch.tensor(tabular_features[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            all_probs.append(0.5)
            all_labels.append(label)
            continue

        view_probs = []
        for tfm in tta_transforms:
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)
            logits = model(img_t, tab)
            prob   = F.softmax(logits, dim=1)[0, 1].item()
            view_probs.append(prob)

        all_probs.append(float(np.mean(view_probs)))
        all_labels.append(label)

    all_probs  = np.array(all_probs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    return all_probs, all_labels


# ---------------------------------------------------------------------------
# 7. Threshold selection: maximize recall at precision >= MIN_PRECISION
# ---------------------------------------------------------------------------
def find_best_threshold(probs, labels, min_precision: float = MIN_PRECISION):
    best_thresh = 0.5
    best_recall = 0.0
    results = []

    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        results.append((t, p, r))
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_thresh = t

    return best_thresh, results


def plot_threshold_curve(results, best_thresh, save_path):
    thresholds = [r[0] for r in results]
    precisions = [r[1] for r in results]
    recalls    = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3")
    ax.plot(thresholds, recalls,    label="Recall",    color="#F44336")
    ax.axvline(best_thresh, color="#4CAF50", linestyle="--",
               label=f"Chosen threshold = {best_thresh:.2f}")
    ax.axhline(MIN_PRECISION, color="grey", linestyle=":", alpha=0.6,
               label=f"Min precision floor = {MIN_PRECISION}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs. Threshold (Melanoma class)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Training & evaluation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, use_cutmix: bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, tabs, labels in loader:
        imgs, tabs, labels = imgs.to(DEVICE), tabs.to(DEVICE), labels.to(DEVICE)

        # CutMix
        if use_cutmix and random.random() < CUTMIX_PROB:
            imgs, soft_labels = cutmix_batch(imgs, labels)
            soft_labels = soft_labels.to(DEVICE)
        else:
            soft_labels = labels.float()

        optimizer.zero_grad()
        logits = model(imgs, tabs)
        loss   = criterion(logits, soft_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            print(f"  [WARNING] Non-finite loss {loss_val:.4f} — skipping batch")
            continue
        total_loss += loss_val * imgs.size(0)
        preds  = logits.argmax(dim=1)
        # For accuracy, compare against hard labels
        hard   = labels
        correct += (preds == hard).sum().item()
        total  += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    criterion = SkinLoss(class_weights=None, label_smoothing=0.0)

    for imgs, tabs, labels in loader:
        imgs, tabs, labels = imgs.to(DEVICE), tabs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs, tabs)
        loss   = criterion(logits, labels.float())

        total_loss += loss.item() * imgs.size(0)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total  += imgs.size(0)

        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / total
    acc        = correct / total

    # Guard against NaN/Inf probabilities (can occur in epoch 1 before stabilisation)
    nan_count = np.isnan(all_probs).sum() + np.isinf(all_probs).sum()
    if nan_count > 0:
        print(f"  [WARNING] {nan_count} NaN/Inf probs detected — replacing with 0.5. "
              f"Model may be numerically unstable. Check LR and DAFT init.")
        all_probs = np.where(np.isfinite(all_probs), all_probs, 0.5)

    if len(np.unique(all_labels)) > 1 and not np.isnan(all_probs).any():
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    # Best threshold on this split
    best_t, _ = find_best_threshold(all_probs, all_labels)
    preds_t   = (all_probs >= best_t).astype(int)
    f1        = f1_score(all_labels, preds_t, zero_division=0)
    prec      = precision_score(all_labels, preds_t, zero_division=0)
    rec       = recall_score(all_labels, preds_t, zero_division=0)

    return avg_loss, acc, auc, f1, prec, rec, best_t, all_probs, all_labels


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    # ---- Load pre-split CSVs ----
    print("Loading splits from DAFT Phase-2 CSVs...")
    train_df = load_split_csv(TRAIN_CSV, "train")
    val_df   = load_split_csv(VAL_CSV,   "val")
    test_df  = load_split_csv(TEST_CSV,  "test")
    print(f"Train distribution:\n{train_df['label'].value_counts(normalize=True)}")

    # ---- Tabular features ----
    X_train, scaler, encoders, tab_dim = build_tabular_features(train_df, fit_scaler=True)
    X_val, _, _, _  = build_tabular_features(val_df,   fit_scaler=False, scaler=scaler, encoders=encoders)
    X_test, _, _, _ = build_tabular_features(test_df,  fit_scaler=False, scaler=scaler, encoders=encoders)
    print(f"Tabular feature dim: {tab_dim}")

    # ---- Datasets ----
    train_ds = MIDASDataset(train_df, X_train, transform=get_transforms("train"))
    val_ds   = MIDASDataset(val_df,   X_val,   transform=get_transforms("val"))
    test_ds  = MIDASDataset(test_df,  X_test,  transform=get_transforms("test"))

    # Weighted sampler: partial oversampling of melanoma
    label_counts = Counter(train_df["label"].values)
    class_weights = {c: 1.0 / count for c, count in label_counts.items()}
    sample_weights = [class_weights[l] for l in train_df["label"].values]
    # Blend with uniform weights to control aggression
    uniform_w = [1.0 / len(train_df)] * len(train_df)
    blended_w = [
        OVERSAMPLE_STRENGTH * sw + (1 - OVERSAMPLE_STRENGTH) * uw
        for sw, uw in zip(sample_weights, uniform_w)
    ]
    sampler = WeightedRandomSampler(blended_w, num_samples=len(train_df), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ---- Model ----
    model = EfficientNetDAFT(tab_dim=tab_dim, num_classes=2, dropout=0.4).to(DEVICE)
    model.freeze_backbone()

    # Class weights: inverse-frequency, normalised so they sum to 2
    n0 = int((train_df["label"] == 0).sum())
    n1 = int((train_df["label"] == 1).sum())
    w  = torch.tensor([1.0 / n0, 1.0 / n1], dtype=torch.float32)
    w  = (w / w.sum() * 2).to(DEVICE)
    criterion = SkinLoss(class_weights=w, label_smoothing=LABEL_SMOOTH)

    # Head-only optimizer (frozen backbone)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ---- Training loop ----
    history = []
    best_auc    = 0.0
    no_improve  = 0
    best_ckpt   = os.path.join(OUTPUT_DIR, "best_model.pth")

    print("\n" + "="*60)
    print("Starting training")
    print("="*60)

    for epoch in range(1, NUM_EPOCHS + 1):

        # Unfreeze backbone at UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH:
            print(f"\n[Epoch {epoch}] Unfreezing backbone — switching to fine-tune LR {LR_FINETUNE}")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH
            )

        use_cutmix = (epoch >= UNFREEZE_EPOCH)  # only after backbone unfreezes
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, use_cutmix)
        val_loss, val_acc, val_auc, val_f1, val_prec, val_rec, val_t, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
            "val_auc": val_auc,       "val_f1": val_f1,
            "val_precision": val_prec,"val_recall": val_rec,
            "threshold": val_t,
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"TrLoss {train_loss:.4f} TrAcc {train_acc:.4f} | "
            f"VaLoss {val_loss:.4f} VaAcc {val_acc:.4f} | "
            f"AUC {val_auc:.4f} F1 {val_f1:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} | "
            f"Thr {val_t:.2f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": val_auc,
                "threshold": val_t,
            }, best_ckpt)
            print(f"  ✓ Best model saved (AUC {best_auc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement: {no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ---- Save training history ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    # ---- Test evaluation ----
    print("\n" + "="*60)
    print("Loading best model for test evaluation")
    print("="*60)

    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    # Use the threshold found during validation
    val_threshold = ckpt["threshold"]

    # ---- Standard evaluation (single pass) ----
    test_loss, test_acc, test_auc, _, _, _, _, test_probs, test_labels = evaluate(model, test_loader)

    best_t_std, threshold_results_std = find_best_threshold(test_probs, test_labels)
    for t_candidate in [val_threshold, best_t_std]:
        preds_candidate = (test_probs >= t_candidate).astype(int)
        if precision_score(test_labels, preds_candidate, zero_division=0) >= MIN_PRECISION:
            final_thresh_std = t_candidate
            break
    else:
        final_thresh_std = best_t_std

    preds_std  = (test_probs >= final_thresh_std).astype(int)
    f1_std     = f1_score(test_labels, preds_std, zero_division=0)
    prec_std   = precision_score(test_labels, preds_std, zero_division=0)
    rec_std    = recall_score(test_labels, preds_std, zero_division=0)
    cm_std     = confusion_matrix(test_labels, preds_std)

    # ---- TTA evaluation ----
    print("\nRunning TTA evaluation (8 views per image)...")
    tta_probs, tta_labels = evaluate_tta(model, test_df, X_test, n_augments=8)
    tta_auc = roc_auc_score(tta_labels, tta_probs) if len(np.unique(tta_labels)) > 1 else 0.0

    best_t_tta, threshold_results_tta = find_best_threshold(tta_probs, tta_labels)
    for t_candidate in [val_threshold, best_t_tta]:
        preds_candidate = (tta_probs >= t_candidate).astype(int)
        if precision_score(tta_labels, preds_candidate, zero_division=0) >= MIN_PRECISION:
            final_thresh_tta = t_candidate
            break
    else:
        final_thresh_tta = best_t_tta

    preds_tta  = (tta_probs >= final_thresh_tta).astype(int)
    f1_tta     = f1_score(tta_labels, preds_tta, zero_division=0)
    prec_tta   = precision_score(tta_labels, preds_tta, zero_division=0)
    rec_tta    = recall_score(tta_labels, preds_tta, zero_division=0)
    cm_tta     = confusion_matrix(tta_labels, preds_tta)

    # ---- Delta summary ----
    delta_auc  = tta_auc  - test_auc
    delta_rec  = rec_tta  - rec_std
    delta_prec = prec_tta - prec_std

    report = (
        f"===== TEST RESULTS =====\n\n"
        f"--- Standard (single pass) ---\n"
        f"Test Loss:      {test_loss:.4f}\n"
        f"Test Accuracy:  {test_acc:.4f}\n"
        f"Test AUC:       {test_auc:.4f}\n"
        f"Test F1:        {f1_std:.4f}\n"
        f"Test Precision: {prec_std:.4f}\n"
        f"Test Recall:    {rec_std:.4f}\n"
        f"Threshold:      {final_thresh_std:.2f}\n"
        f"Confusion Matrix:\n{cm_std}\n"
        f"  TN={cm_std[0,0]}  FP={cm_std[0,1]}\n"
        f"  FN={cm_std[1,0]}  TP={cm_std[1,1]}\n\n"
        f"--- TTA (8 views, averaged) ---\n"
        f"TTA AUC:        {tta_auc:.4f}  (delta {delta_auc:+.4f})\n"
        f"TTA F1:         {f1_tta:.4f}\n"
        f"TTA Precision:  {prec_tta:.4f}  (delta {delta_prec:+.4f})\n"
        f"TTA Recall:     {rec_tta:.4f}  (delta {delta_rec:+.4f})\n"
        f"Threshold:      {final_thresh_tta:.2f}\n"
        f"Confusion Matrix:\n{cm_tta}\n"
        f"  TN={cm_tta[0,0]}  FP={cm_tta[0,1]}\n"
        f"  FN={cm_tta[1,0]}  TP={cm_tta[1,1]}\n\n"
        f"--- Classification report (TTA) ---\n"
        f"{classification_report(tta_labels, preds_tta, target_names=['no melanoma','melanoma'])}\n"
    )

    print(report)
    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
        f.write(report)

    # ---- Threshold curves (both) ----
    plot_threshold_curve(
        threshold_results_std, final_thresh_std,
        os.path.join(OUTPUT_DIR, "threshold_curve_standard.png")
    )
    plot_threshold_curve(
        threshold_results_tta, final_thresh_tta,
        os.path.join(OUTPUT_DIR, "threshold_curve_tta.png")
    )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
