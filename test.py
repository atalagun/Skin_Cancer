import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score

TRAIN_CSV = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run4_daft_phase2/train_split.csv"
TEST_CSV = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run4_daft_phase2/test_split.csv"
IMAGE_ROOT = r"C:/Users/Ata/midasmultimodalimagedatasetforaibasedskincancer"
CHECKPOINT_PATH = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run12_effb3_daft_mixup_v2/best_model.pth"
OUTPUT_DIR = r"C:/Users/Ata/Desktop/eval_short_best_model"
IMG_SIZE = 300
BATCH_SIZE = 16
FIXED_THRESHOLD_STANDARD = 0.71
FIXED_THRESHOLD_TTA = 0.66
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
image_index = {}

def build_image_index():
    global image_index
    if image_index:
        return
    for root, _, files in os.walk(IMAGE_ROOT):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                image_index[f.lower()] = os.path.join(root, f)

def resolve_image_path(p):
    build_image_index()
    p = str(p)
    if os.path.isfile(p):
        return p
    fname = os.path.basename(p)
    candidate = os.path.join(IMAGE_ROOT, fname)
    if os.path.isfile(candidate):
        return candidate
    if fname.lower() in image_index:
        return image_index[fname.lower()]
    stem = os.path.splitext(fname)[0].lower()
    for k, v in image_index.items():
        if os.path.splitext(k)[0] == stem:
            return v
    return ""

def load_split(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "label_encoded" in df.columns:
        df["label"] = df["label_encoded"].astype(int)
    elif "label" in df.columns:
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = (df["midas_melanoma"].astype(str).str.lower().str.strip() == "yes").astype(int)
    path_col = None
    for c in ["image_path", "abs_path", "path", "midas_path"]:
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        raise ValueError("No image path column found.")
    df["abs_path"] = df[path_col].apply(resolve_image_path)
    df = df[df["abs_path"] != ""].reset_index(drop=True)
    return df

def build_tabular(df, fit=True, scaler=None, encoders=None):
    cat_cols = ["midas_gender", "midas_fitzpatrick", "midas_location", "midas_ethnicity", "midas_race", "clinical_impression_1", "clinical_impression_2", "clinical_impression_3"]
    num_cols = ["midas_age", "midas_distance", "length_(mm)", "width_(mm)"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    xdf = df[cat_cols + num_cols].copy()
    if encoders is None:
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            xdf[c] = le.fit_transform(xdf[c].astype(str).fillna("unknown"))
            encoders[c] = le
    else:
        for c in cat_cols:
            le = encoders[c]
            known = set(le.classes_)
            xdf[c] = xdf[c].astype(str).fillna("unknown").map(lambda x: int(le.transform([x])[0]) if x in known else len(le.classes_))
    for c in num_cols:
        xdf[c] = pd.to_numeric(xdf[c], errors="coerce")
        med = xdf[c].median()
        if pd.isna(med):
            med = 0.0
        xdf[c] = xdf[c].fillna(med)
    X = xdf.values.astype(np.float32)
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X.astype(np.float32), scaler, encoders, X.shape[1]

class MIDASDataset(Dataset):
    def __init__(self, df, X):
        self.paths = df["abs_path"].values
        self.labels = df["label"].values.astype(np.int64)
        self.X = X.astype(np.float32)
        self.tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.tfm(img)
        tab = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return img, tab, y

class DAFTBlock(nn.Module):
    def __init__(self, tab_dim, channels, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, channels * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x, tab):
        gamma, beta = self.mlp(tab).chunk(2, dim=1)
        gamma = torch.tanh(gamma) * 0.5
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1 + gamma) * x + beta

class EfficientNetDAFT(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = backbone.classifier[1].in_features
        self.daft = DAFTBlock(tab_dim, feat_dim)
        self.tab_mlp = nn.Sequential(nn.Linear(tab_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.SiLU())
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(feat_dim + 32, 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 2))
    def forward(self, img, tab):
        tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.features(img)
        x = self.daft(x, tab)
        x = self.avgpool(x).flatten(1)
        t = self.tab_mlp(tab)
        return self.classifier(torch.cat([x, t], dim=1))

def load_model(tab_dim):
    model = EfficientNetDAFT(tab_dim).to(DEVICE)
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model

@torch.no_grad()
def predict_standard(model, loader):
    model.eval()
    probs, labels = [], []
    total_loss, total = 0.0, 0
    for img, tab, y in loader:
        img, tab, y = img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
        logits = model(img, tab)
        loss = F.cross_entropy(logits, y)
        p = F.softmax(logits, dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        labels.extend(y.cpu().numpy())
        total_loss += loss.item() * img.size(0)
        total += img.size(0)
    return total_loss / total, np.array(probs), np.array(labels)

@torch.no_grad()
def predict_tta(model, df, X):
    model.eval()
    base = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    probs, labels = [], []
    for i in range(len(df)):
        img = Image.open(df["abs_path"].iloc[i]).convert("RGB")
        tab = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        views = [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM), img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)]
        ps = []
        for v in views:
            x = base(v).unsqueeze(0).to(DEVICE)
            logits = model(x, tab)
            ps.append(F.softmax(logits, dim=1)[0, 1].item())
        probs.append(np.mean(ps))
        labels.append(int(df["label"].iloc[i]))
    return np.array(probs), np.array(labels)

def metrics(probs, labels, threshold):
    pred = (probs >= threshold).astype(int)
    return {"threshold": threshold, "accuracy": accuracy_score(labels, pred), "auc": roc_auc_score(labels, probs), "f1": f1_score(labels, pred, zero_division=0), "precision": precision_score(labels, pred, zero_division=0), "recall": recall_score(labels, pred, zero_division=0), "cm": confusion_matrix(labels, pred), "report": classification_report(labels, pred, target_names=["no melanoma", "melanoma"], zero_division=0), "pred": pred}

def plot_roc(labels, probs, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_pr(labels, probs, path):
    p, r, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    base = labels.mean()
    plt.figure(figsize=(6, 6))
    plt.plot(r, p, label=f"AP={ap:.4f}")
    plt.axhline(base, linestyle="--", label=f"Baseline={base:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_cm(cm, path):
    txt = np.array([[f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"], [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred no", "Pred melanoma"])
    plt.yticks([0, 1], ["True no", "True melanoma"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, txt[i, j], ha="center", va="center", fontsize=14, fontweight="bold")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_predictions(df, probs, labels, pred, path):
    pd.DataFrame({"abs_path": df["abs_path"], "true_label": labels, "prob_melanoma": probs, "pred_label": pred}).to_csv(path, index=False)

def write_report(path, loss, m_std, m_tta):
    cm1 = m_std["cm"]
    cm2 = m_tta["cm"]
    text = f"""===== TEST RESULTS =====
Device: {DEVICE}
Checkpoint: {CHECKPOINT_PATH}
--- Standard single pass ---
Test Loss:      {loss:.4f}
Accuracy:       {m_std['accuracy']:.4f}
AUC:            {m_std['auc']:.4f}
F1:             {m_std['f1']:.4f}
Precision:      {m_std['precision']:.4f}
Recall:         {m_std['recall']:.4f}
Threshold:      {m_std['threshold']:.2f}
Confusion Matrix:
{cm1}
TN={cm1[0,0]} FP={cm1[0,1]}
FN={cm1[1,0]} TP={cm1[1,1]}
Classification report:
{m_std['report']}
--- TTA ---
Accuracy:       {m_tta['accuracy']:.4f}
AUC:            {m_tta['auc']:.4f}
F1:             {m_tta['f1']:.4f}
Precision:      {m_tta['precision']:.4f}
Recall:         {m_tta['recall']:.4f}
Threshold:      {m_tta['threshold']:.2f}
Confusion Matrix:
{cm2}
TN={cm2[0,0]} FP={cm2[0,1]}
FN={cm2[1,0]} TP={cm2[1,1]}
Classification report:
{m_tta['report']}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

def main():
    print("Device:", DEVICE)
    train_df = load_split(TRAIN_CSV)
    test_df = load_split(TEST_CSV)
    X_train, scaler, encoders, tab_dim = build_tabular(train_df, fit=True)
    X_test, _, _, _ = build_tabular(test_df, fit=False, scaler=scaler, encoders=encoders)
    test_ds = MIDASDataset(test_df, X_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = load_model(tab_dim)
    loss, probs_std, labels_std = predict_standard(model, test_loader)
    probs_tta, labels_tta = predict_tta(model, test_df, X_test)
    print("Fixed standard threshold:", FIXED_THRESHOLD_STANDARD)
    print("Fixed TTA threshold:", FIXED_THRESHOLD_TTA)
    m_std = metrics(probs_std, labels_std, FIXED_THRESHOLD_STANDARD)
    m_tta = metrics(probs_tta, labels_tta, FIXED_THRESHOLD_TTA)
    save_predictions(test_df, probs_std, labels_std, m_std["pred"], os.path.join(OUTPUT_DIR, "test_predictions_standard.csv"))
    save_predictions(test_df, probs_tta, labels_tta, m_tta["pred"], os.path.join(OUTPUT_DIR, "test_predictions_tta.csv"))
    plot_roc(labels_tta, probs_tta, os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plot_pr(labels_tta, probs_tta, os.path.join(OUTPUT_DIR, "precision_recall_curve.png"))
    plot_cm(m_tta["cm"], os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    write_report(os.path.join(OUTPUT_DIR, "test_report.txt"), loss, m_std, m_tta)
    print("Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()