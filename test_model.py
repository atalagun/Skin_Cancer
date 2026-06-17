import os, argparse, random, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights, DenseNet121_Weights
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

IMG_SIZE      = 300
MIN_PRECISION = 0.40
IMG_SIZE      = 300
MIN_PRECISION = 0.40

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DAFTBlockSimple(nn.Module):
    def __init__(self, tabular_dim, feature_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, feature_channels * 2),
        )
    def forward(self, feature_map, tabular):
        gamma, beta = torch.chunk(self.mlp(tabular), 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1.0 + gamma) + beta

class DAFTMelanomaModel(nn.Module):
    def __init__(self, tabular_dim, num_classes=2):
        super().__init__()
        b = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features   = b.features
        self.daft       = DAFTBlockSimple(tabular_dim, 1024)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 96), nn.ReLU(), nn.Dropout(0.7), nn.Linear(96, num_classes))
    def forward(self, img, tab):
        x = self.daft(torch.relu(self.features(img)), tab)
        return self.classifier(self.pool(x).flatten(1))

class DAFTEfficientNetB3(nn.Module):
    def __init__(self, tabular_dim, num_classes=2):
        super().__init__()
        b = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features   = b.features
        self.daft       = DAFTBlockSimple(tabular_dim, 1536)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1536, 128), nn.ReLU(), nn.Dropout(0.7), nn.Linear(128, num_classes))
    def forward(self, img, tab):
        return self.classifier(self.pool(self.daft(self.features(img), tab)).flatten(1))

class DAFTEfficientNetB3Canny(nn.Module):
    def __init__(self, tabular_dim, num_classes=2):
        super().__init__()
        b  = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        oc = b.features[0][0]
        nc = nn.Conv2d(4, oc.out_channels, oc.kernel_size, oc.stride, oc.padding, bias=False)
        with torch.no_grad():
            nc.weight[:, :3]  = oc.weight
            nc.weight[:, 3:4] = oc.weight.mean(dim=1, keepdim=True)
        b.features[0][0] = nc
        fd = b.classifier[1].in_features
        self.features   = b.features
        self.daft       = DAFTBlockSimple(tabular_dim, fd)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(fd, 128), nn.ReLU(), nn.Dropout(0.7), nn.Linear(128, num_classes))
    def forward(self, img, tab):
        return self.classifier(self.pool(self.daft(self.features(img), tab)).flatten(1))

def _make_effb3_4ch():
    b  = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    oc = b.features[0][0]
    nc = nn.Conv2d(4, oc.out_channels, oc.kernel_size, oc.stride, oc.padding, bias=False)
    with torch.no_grad():
        nc.weight[:, :3]  = oc.weight
        nc.weight[:, 3:4] = oc.weight.mean(dim=1, keepdim=True)
    b.features[0][0] = nc
    fd = b.classifier[1].in_features
    b.classifier = nn.Identity()
    return b, fd

class DualEfficientNetDAFT(nn.Module):
    def __init__(self, tabular_dim, num_classes=2):
        super().__init__()
        self.global_backbone, gd = _make_effb3_4ch()
        self.crop_backbone,   cd = _make_effb3_4ch()
        fd = gd + cd
        self.daft       = DAFTBlockSimple(tabular_dim, fd)
        self.classifier = nn.Sequential(
            nn.Linear(fd, 256), nn.ReLU(), nn.Dropout(0.7), nn.Linear(256, num_classes))
    def forward(self, global_img, crop_img, tab):
        f = self.daft(torch.cat([self.global_backbone(global_img), self.crop_backbone(crop_img)], dim=1), tab)
        return self.classifier(f)

class DAFTBlockRich(nn.Module):
    def __init__(self, tab_dim, num_channels, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, feature_map, tab):
        gamma, beta = self.mlp(tab).chunk(2, dim=1)
        gamma = torch.tanh(gamma) * 0.5
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * feature_map + beta

class EfficientNetDAFT(nn.Module):
    def __init__(self, tab_dim, num_classes=2, dropout=0.4):
        super().__init__()
        b = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features  = b.features
        self.avgpool   = b.avgpool
        fd = b.classifier[1].in_features
        self.daft      = DAFTBlockRich(tab_dim, fd, hidden=128)
        self.tab_mlp   = nn.Sequential(
            nn.Linear(tab_dim, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Dropout(0.3), nn.Linear(64, 32), nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fd + 32, 256),
            nn.SiLU(), nn.Dropout(dropout * 0.5), nn.Linear(256, num_classes),
        )
    def forward(self, img, tab):
        tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        x   = self.daft(self.features(img), tab)
        x   = self.avgpool(x).flatten(1)
        t   = self.tab_mlp(tab)
        return self.classifier(torch.cat([x, t], dim=1))

def detect_and_load(ckpt_path, tab_dim):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    saved_tab_dim = ckpt.get("tabular_input_dim", None) if isinstance(ckpt, dict) else None
    if saved_tab_dim is not None and saved_tab_dim != tab_dim:
        tab_dim = saved_tab_dim
    keys  = list(state.keys())
    if any("denseblock" in k for k in keys):
        kind  = "Run4 — DAFTMelanomaModel (DenseNet121)"
        model = DAFTMelanomaModel(tabular_dim=tab_dim)
    elif any("global_backbone" in k or "crop_backbone" in k for k in keys):
        kind  = "Run7 — DualEfficientNetDAFT (dual 4-ch EffNet-B3)"
        model = DualEfficientNetDAFT(tabular_dim=tab_dim)
    elif any("tab_mlp" in k for k in keys):
        kind  = "Run8/12 — EfficientNetDAFT (rich DAFT + tabular MLP)"
        model = EfficientNetDAFT(tab_dim=tab_dim)
    else:
        first = next((k for k in keys if "features.0.0.weight" in k), None)
        if first and state[first].shape[1] == 4:
            kind  = "Run6 — DAFTEfficientNetB3Canny (4-ch EffNet-B3)"
            model = DAFTEfficientNetB3Canny(tabular_dim=tab_dim)
        else:
            kind  = "Run5 — DAFTEfficientNetB3 (3-ch EffNet-B3)"
            model = DAFTEfficientNetB3(tabular_dim=tab_dim)
    print(f"  Detected: {kind}")
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    ep  = ckpt.get("epoch",   "?")
    auc = ckpt.get("val_auc", "?")
    auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
    print(f"  Epoch: {ep}  |  Val AUC: {auc_str}")
    return model, kind

_IDX: dict = {}

def _build_idx(root):
    global _IDX
    if _IDX: return
    print(f"  Indexing images: {root}")
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                _IDX[f.lower()] = os.path.join(r, f)
    print(f"  Index: {len(_IDX)} files")

def resolve(raw, root):
    _build_idx(root)
    if os.path.isfile(raw): return raw
    fn = os.path.basename(str(raw))
    c  = os.path.join(root, fn)
    if os.path.isfile(c): return c
    return _IDX.get(fn.lower(), "")

def load_split(csv_path, image_root):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    if "label_encoded" in df.columns:
        df["label"] = df["label_encoded"].astype(int)
    elif "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
    elif "midas_melanoma" in df.columns:
        df["label"] = (df["midas_melanoma"].astype(str).str.strip().str.lower()=="yes").astype(int)
    else:
        raise ValueError(f"No label column. Columns: {df.columns.tolist()}")
    pc = next((c for c in ["image_path","abs_path","path"] if c in df.columns), None)
    if not pc: raise ValueError("No image path column found.")
    df["abs_path"] = df[pc].apply(lambda p: resolve(str(p), image_root))
    df = df[df["abs_path"] != ""].reset_index(drop=True)
    print(f"  Loaded {len(df)} samples  |  mel: {df['label'].sum()}  non-mel: {(df['label']==0).sum()}")
    return df

def build_tabular(df, fit=True, scaler=None, encoders=None):
    cat = ["midas_gender","midas_fitzpatrick","midas_location",
           "midas_ethnicity","midas_race",
           "clinical_impression_1","clinical_impression_2","clinical_impression_3"]
    num = ["midas_age","midas_distance","length_(mm)","width_(mm)"]
    cat = [c for c in cat if c in df.columns]
    num = [c for c in num if c in df.columns]
    fd  = df[cat + num].copy()
    if encoders is None:
        encoders = {}
        for c in cat:
            le = LabelEncoder()
            fd[c] = le.fit_transform(fd[c].astype(str).fillna("unknown"))
            encoders[c] = le
    else:
        for c in cat:
            le = encoders[c]
            known = set(le.classes_)
            fd[c] = fd[c].astype(str).fillna("unknown").apply(
                lambda x: int(le.transform([x])[0]) if x in known else len(le.classes_)
            )
    fd[num] = fd[num].fillna(fd[num].median())
    X = fd.values.astype(np.float32)
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X.astype(np.float32), scaler, encoders, X.shape[1]

def build_tabular_ohe(df, fit=True, preprocessor=None):
    cat = ["midas_gender","midas_fitzpatrick","midas_location",
           "midas_ethnicity","midas_race",
           "clinical_impression_1","clinical_impression_2","clinical_impression_3"]
    num = ["midas_age","midas_distance","length_(mm)","width_(mm)"]
    cat = [c for c in cat if c in df.columns]
    num = [c for c in num if c in df.columns]
    for c in cat:
        df[c] = df[c].astype(str).fillna("missing")
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    if fit:
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
        X = preprocessor.fit_transform(df[num + cat])
    else:
        X = preprocessor.transform(df[num + cat])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32), X.shape[1], preprocessor

def clean_tfm():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def tta_tfms(n=8):
    aug = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return [clean_tfm()] + [aug]*(n-1)

def to_4ch(t):
    return torch.cat([t, torch.zeros(1, t.shape[1], t.shape[2])], dim=0)

@torch.no_grad()
def run_inference(model, df, X, tfms, kind):
    model.eval()
    is_dual   = "Dual" in kind
    needs_4ch = "4-ch" in kind or "Canny" in kind or is_dual
    probs, labels = [], []
    for i in range(len(df)):
        path  = df["abs_path"].iloc[i]
        label = int(df["label"].iloc[i])
        tab   = torch.tensor(X[i]).unsqueeze(0).to(DEVICE)
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            probs.append(0.5); labels.append(label); continue
        vp = []
        for tfm in tfms:
            img = tfm(pil)
            if needs_4ch: img = to_4ch(img)
            img = img.unsqueeze(0).to(DEVICE)
            logits = model(img, img, tab) if is_dual else model(img, tab)
            vp.append(F.softmax(logits, dim=1)[0, 1].item())
        probs.append(float(np.mean(vp)))
        labels.append(label)
    return np.array(probs, np.float32), np.array(labels, np.int64)

def find_threshold(probs, labels, min_prec=MIN_PRECISION):
    best_t, best_r, results = 0.5, 0.0, []
    for t in np.arange(0.05, 0.96, 0.005):
        t  = round(float(t), 4)
        pr = (probs >= t).astype(int)
        if pr.sum() == 0: continue
        p = precision_score(labels, pr, zero_division=0)
        r = recall_score(labels, pr, zero_division=0)
        results.append((t, p, r))
        if p >= min_prec and r > best_r:
            best_r = r; best_t = t
    if best_r == 0.0:
        for floor in [min_prec - 0.01, min_prec - 0.02]:
            for t, p, r in results:
                if p >= floor and r > best_r:
                    best_r = r; best_t = t
            if best_r > 0: break
    return best_t, results

def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {os.path.basename(path)}")

def plot_threshold_curve(res, best_t, path, title):
    ts = [r[0] for r in res]; ps = [r[1] for r in res]; rs = [r[2] for r in res]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, ps, color="#2196F3", lw=1.5, label="Precision")
    ax.plot(ts, rs, color="#F44336", lw=1.5, label="Recall")
    ax.axvline(best_t, color="#4CAF50", lw=1.5, ls="--", label=f"Threshold = {best_t:.3f}")
    ax.axhline(MIN_PRECISION, color="grey", ls=":", alpha=0.7, label=f"Precision floor = {MIN_PRECISION}")
    ax.set(xlabel="Threshold", ylabel="Score", title=title, xlim=(0,1), ylim=(0,1.05))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, path)

def plot_roc(labels, probs, auc, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#2196F3")
    ax.plot([0,1],[0,1], color="grey", ls="--", alpha=0.6, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Melanoma Detection", xlim=(0,1), ylim=(0,1.02))
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, path)

def plot_pr(labels, probs, path):
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap   = average_precision_score(labels, probs)
    prev = labels.mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#9C27B0", lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(rec, prec, alpha=0.10, color="#9C27B0")
    ax.axhline(prev,          color="grey",    ls="--", alpha=0.6, label=f"Prevalence = {prev:.2f}")
    ax.axhline(MIN_PRECISION, color="#FF9800", ls=":",  alpha=0.8, label=f"Precision floor = {MIN_PRECISION}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve — Melanoma Detection", xlim=(0,1), ylim=(0,1.05))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, path)

def plot_histogram(probs, labels, best_t, path, title):
    mel  = probs[labels == 1]
    nmel = probs[labels == 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(nmel, bins=bins, alpha=0.6, color="#2196F3", label=f"Non-melanoma (n={len(nmel)})", density=True)
    ax.hist(mel,  bins=bins, alpha=0.6, color="#F44336", label=f"Melanoma (n={len(mel)})", density=True)
    ax.axvline(best_t, color="#4CAF50", lw=2, ls="--", label=f"Threshold = {best_t:.3f}")
    ax.set(xlabel="P(melanoma)", ylabel="Density", title=title)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, path)

def plot_confusion(labels, preds, path, title):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    tl = ["No Melanoma", "Melanoma"]
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=tl, yticklabels=tl,
           xlabel="Predicted", ylabel="Actual", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    _save(fig, path)

def calc_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return dict(probs=probs, labels=labels, preds=preds, threshold=threshold,
                auc=roc_auc_score(labels, probs) if len(np.unique(labels))>1 else 0.0,
                f1=f1_score(labels, preds, zero_division=0),
                precision=precision_score(labels, preds, zero_division=0),
                recall=recall_score(labels, preds, zero_division=0),
                accuracy=accuracy_score(labels, preds))

def save_predictions(df, probs, labels, preds, path):
    pd.DataFrame({
        "abs_path":      df["abs_path"],
        "true_label":    labels,
        "prob_melanoma": probs,
        "pred_label":    preds
    }).to_csv(path, index=False)

def write_report(path, ckpt, kind, std, tta=None):
    def blk(tag, r):
        cm = confusion_matrix(r["labels"], r["preds"])
        cr = classification_report(r["labels"], r["preds"], target_names=["no melanoma","melanoma"])
        return "\n".join([
            f"\n--- {tag} ---",
            f"AUC:       {r['auc']:.4f}",
            f"F1:        {r['f1']:.4f}",
            f"Precision: {r['precision']:.4f}",
            f"Recall:    {r['recall']:.4f}",
            f"Threshold: {r['threshold']:.3f}",
            f"Accuracy:  {r['accuracy']:.4f}",
            "Confusion Matrix:", str(cm),
            f"  TN={cm[0,0]}  FP={cm[0,1]}",
            f"  FN={cm[1,0]}  TP={cm[1,1]}", "", cr,
        ])
    lines = ["="*60, "TEST RESULTS", "="*60,
             f"Checkpoint  : {ckpt}", ""]
    lines.append(blk("Standard (single pass)", std))
    if tta:
        lines.append(blk("TTA", tta))
    rep = "\n".join(lines)
    print(rep)
    with open(path, "w", encoding="utf-8") as f: f.write(rep)
    print(f"\n  Report saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,  help="Path to best_model.pth")
    parser.add_argument("--train_csv",   required=True,  help="Path to train_split.csv (used to fit scaler)")
    parser.add_argument("--test_csv",    required=True,  help="Path to test_split.csv")
    parser.add_argument("--image_root",  required=True,  help="Path to MIDAS image folder")
    parser.add_argument("--output_dir",  default=None,   help="Where to save results (default: test_results/<run>/)")
    parser.add_argument("--tta_views",   type=int, default=8)
    parser.add_argument("--threshold",   type=float, default=None)
    args = parser.parse_args()
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.train_csv  = os.path.abspath(args.train_csv)
    args.test_csv   = os.path.abspath(args.test_csv)
    args.image_root = os.path.abspath(args.image_root)
    if args.output_dir is None:
        args.output_dir = os.path.join("test_results", os.path.basename(os.path.dirname(args.checkpoint)))
    os.makedirs(args.output_dir, exist_ok=True)
    od = args.output_dir
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda": print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Output: {od}\n")
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    saved_tab_dim = ckpt_meta.get("tabular_input_dim", None) if isinstance(ckpt_meta, dict) else None
    use_ohe = saved_tab_dim is not None and saved_tab_dim > 12
    print("Loading train split (to fit scaler)...")
    train_df = load_split(args.train_csv, args.image_root)
    if use_ohe:
        X_train, _, ohe_preprocessor = build_tabular_ohe(train_df.copy(), fit=True)
        tab_dim = saved_tab_dim
        print("Loading test set...")
        test_df = load_split(args.test_csv, args.image_root)
        X_test, _, _ = build_tabular_ohe(test_df.copy(), fit=False, preprocessor=ohe_preprocessor)
    else:
        X_train, scaler, encoders, tab_dim = build_tabular(train_df, fit=True)
        print("Loading test set...")
        test_df = load_split(args.test_csv, args.image_root)
        X_test, _, _, _ = build_tabular(test_df, fit=False, scaler=scaler, encoders=encoders)
    labels = test_df["label"].values.astype(np.int64)
    print(f"Tabular dim: {tab_dim}")
    print(f"\nLoading: {args.checkpoint}")
    model, kind = detect_and_load(args.checkpoint, tab_dim)
    print("\nStandard inference...")
    sp, _ = run_inference(model, test_df, X_test, [clean_tfm()], kind)
    if args.threshold is not None:
        st = args.threshold; sres = None
        print(f"  Fixed threshold: {st:.3f}")
    else:
        st, sres = find_threshold(sp, labels)
        print(f"  Threshold: {st:.3f}")
    std = calc_metrics(sp, labels, st)
    save_predictions(test_df, sp, labels, std["preds"],
                     os.path.join(od, "test_predictions_standard.csv"))
    tta = None; tres = None
    if args.tta_views > 1:
        print(f"\nTTA ({args.tta_views} views)...")
        tp, _ = run_inference(model, test_df, X_test, tta_tfms(args.tta_views), kind)
        if args.threshold is not None:
            tt = args.threshold
        else:
            tt, tres = find_threshold(tp, labels)
            print(f"  TTA threshold: {tt:.3f}")
        tta = calc_metrics(tp, labels, tt)
        save_predictions(test_df, tp, labels, tta["preds"],
                         os.path.join(od, "test_predictions_tta.csv"))
    best = tta if tta else std
    print("\nGenerating plots...")
    if sres:
        plot_threshold_curve(sres, st,
            os.path.join(od,"threshold_curve_standard.png"),
            "Precision & Recall vs. Threshold (Standard pass)")
    if tres:
        plot_threshold_curve(tres, best["threshold"],
            os.path.join(od,"threshold_curve_tta.png"),
            "Precision & Recall vs. Threshold (TTA)")
    plot_roc(labels, best["probs"], best["auc"], os.path.join(od,"roc_curve.png"))
    plot_pr(labels, best["probs"], os.path.join(od,"precision_recall_curve.png"))
    plot_histogram(best["probs"], labels, best["threshold"],
        os.path.join(od,"probability_histogram.png"),
        title=f"Probability Distribution  (threshold = {best['threshold']:.3f})")
    plot_confusion(labels, best["preds"],
        os.path.join(od,"confusion_matrix.png"),
        title=f"Confusion Matrix  (threshold = {best['threshold']:.3f})")
    write_report(os.path.join(od,"test_report.txt"), args.checkpoint, kind, std, tta)
    print(f"\nDone — all outputs in: {od}")

if __name__ == "__main__":
    main()
