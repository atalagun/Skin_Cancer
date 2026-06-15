import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torchvision.transforms.functional as F

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

EXCEL_PATH = r"C:/Users/Ata/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx"
IMAGE_ROOT = r"C:/Users/Ata/midasmultimodalimagedatasetforaibasedskincancer"

OUTPUT_DIR = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run4_daft_phase2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_daft_phase2.pth")

BATCH_SIZE = 16
NUM_EPOCHS = 25
FREEZE_EPOCHS = 5
PATIENCE = 6

LR_HEAD = 1e-4
LR_FINE_TUNE = 7e-6
WEIGHT_DECAY = 2e-4

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

df = pd.read_excel(EXCEL_PATH)
print("Initial shape:", df.shape)

df = df.dropna(subset=["midas_record_id", "midas_melanoma"]).copy()
df["midas_melanoma"] = df["midas_melanoma"].astype(str).str.strip().str.lower()

label_map = {
    "yes": "yes",
    "no": "no",
    "1": "yes",
    "0": "no",
    "true": "yes",
    "false": "no"
}

df["midas_melanoma"] = df["midas_melanoma"].map(label_map)
df = df.dropna(subset=["midas_melanoma"]).copy()

print("\nAfter label cleaning:", df.shape)
print(df["midas_melanoma"].value_counts())

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_image_path(row):
    folder = safe_str(row.get("midas_path", ""))
    fname = safe_str(row.get("midas_file_name", ""))

    if not fname:
        return None

    candidates = []

    if folder:
        candidates.append(os.path.join(IMAGE_ROOT, folder, fname))

    candidates.append(os.path.join(IMAGE_ROOT, fname))

    for sub in ["images", "image", "img", "imgs", "release", "data"]:
        if folder:
            candidates.append(os.path.join(IMAGE_ROOT, sub, folder, fname))
        candidates.append(os.path.join(IMAGE_ROOT, sub, fname))

    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p

    return None


df["image_path"] = df.apply(build_image_path, axis=1)
df = df[df["image_path"].apply(lambda x: isinstance(x, str) and os.path.exists(x))].copy()

print("\nAfter path filtering:", df.shape)
print(df["midas_melanoma"].value_counts())

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
train_idx, temp_idx = next(gss1.split(df, groups=df["midas_record_id"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df = df.iloc[temp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["midas_record_id"]))

val_df = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print("\nSplit sizes")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("\nTrain distribution:")
print(train_df["midas_melanoma"].value_counts(normalize=True))
print("\nVal distribution:")
print(val_df["midas_melanoma"].value_counts(normalize=True))
print("\nTest distribution:")
print(test_df["midas_melanoma"].value_counts(normalize=True))

label_col = "midas_melanoma"

numeric_cols = [
    "midas_age",
    "midas_distance",
    "length_(mm)",
    "width_(mm)"
]

categorical_cols = [
    "midas_iscontrol",
    "midas_location",
    "midas_gender",
    "midas_fitzpatrick",
    "midas_ethnicity",
    "midas_race",
    "clinical_impression_1",
    "clinical_impression_2",
    "clinical_impression_3"
]

numeric_cols = [c for c in numeric_cols if c in df.columns]
categorical_cols = [c for c in categorical_cols if c in df.columns]

valid_numeric_cols = []

for col in numeric_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    if train_df[col].notna().sum() == 0:
        print(f"Dropping numeric column because all values are missing: {col}")
        continue

    median_value = train_df[col].median()

    train_df[col] = train_df[col].fillna(median_value)
    val_df[col] = val_df[col].fillna(median_value)
    test_df[col] = test_df[col].fillna(median_value)

    valid_numeric_cols.append(col)

numeric_cols = valid_numeric_cols

for col in categorical_cols:
    train_df[col] = train_df[col].fillna("missing").astype(str)
    val_df[col] = val_df[col].fillna("missing").astype(str)
    test_df[col] = test_df[col].fillna("missing").astype(str)

feature_cols = numeric_cols + categorical_cols

print("\nFinal numeric columns:", numeric_cols)
print("Final categorical columns:", categorical_cols)


label_encoder = LabelEncoder()

train_df["label_encoded"] = label_encoder.fit_transform(train_df[label_col])
val_df["label_encoded"] = label_encoder.transform(val_df[label_col])
test_df["label_encoded"] = label_encoder.transform(test_df[label_col])

print("\nLabel mapping:")
for cls, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(cls, "->", idx)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_train_tab = preprocessor.fit_transform(train_df[feature_cols])
X_val_tab = preprocessor.transform(val_df[feature_cols])
X_test_tab = preprocessor.transform(test_df[feature_cols])

if hasattr(X_train_tab, "toarray"):
    X_train_tab = X_train_tab.toarray()
    X_val_tab = X_val_tab.toarray()
    X_test_tab = X_test_tab.toarray()

X_train_tab = X_train_tab.astype(np.float32)
X_val_tab = X_val_tab.astype(np.float32)
X_test_tab = X_test_tab.astype(np.float32)

tabular_input_dim = X_train_tab.shape[1]
num_classes = len(label_encoder.classes_)

print("\nTabular input dimension:", tabular_input_dim)
print("Number of classes:", num_classes)

class RandomGamma:
    def __init__(self, gamma_range=(0.98, 1.02), p=0.15):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return F.adjust_gamma(img, gamma)
        return img


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    RandomGamma(gamma_range=(0.98, 1.02), p=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class MIDASDataset(Dataset):
    def __init__(self, dataframe, tabular_array, image_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.tabular = tabular_array
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

        tabular = torch.tensor(self.tabular[idx], dtype=torch.float32)
        label = torch.tensor(int(row["label_encoded"]), dtype=torch.long)

        return image, tabular, label


train_dataset = MIDASDataset(train_df, X_train_tab, train_transform)
val_dataset = MIDASDataset(val_df, X_val_tab, eval_transform)
test_dataset = MIDASDataset(test_df, X_test_tab, eval_transform)

train_labels = train_df["label_encoded"].values
class_counts = np.bincount(train_labels)

class_sample_weights = 1.0 / class_counts
sample_weights = class_sample_weights[train_labels]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=int(len(sample_weights) * 0.85),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

class DAFTBlock(nn.Module):
    def __init__(self, tabular_dim, feature_channels):
        super().__init__()

        # Smaller DAFT MLP to reduce overfitting
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, feature_channels * 2)
        )

    def forward(self, feature_map, tabular):
        gamma_beta = self.mlp(tabular)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return feature_map * (1.0 + gamma) + beta


class DAFTMelanomaModel(nn.Module):
    def __init__(self, tabular_dim, num_classes):
        super().__init__()

        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        self.features = backbone.features
        self.daft = DAFTBlock(tabular_dim=tabular_dim, feature_channels=1024)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(1024, 96),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(96, num_classes)
        )

    def forward(self, image, tabular):
        x = self.features(image)
        x = torch.relu(x)

        x = self.daft(x, tabular)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)


model = DAFTMelanomaModel(tabular_input_dim, num_classes).to(DEVICE)


def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = True


freeze_backbone(model)

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, smoothing=0.03):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        ce_loss = -(true_dist * log_probs).sum(dim=1)

        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs) ** self.gamma

        loss = focal_factor * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


alpha = [0.38, 0.62]
criterion = FocalLossWithSmoothing(alpha=alpha, gamma=1.5, smoothing=0.03)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD,
    weight_decay=WEIGHT_DECAY
)


def train_one_epoch():
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, tabular, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        tabular = tabular.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images, tabular)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader.sampler)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate_probs(loader):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, tabular, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            tabular = tabular.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images, tabular)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, np.array(all_labels), np.array(all_probs)


def metrics_from_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")

    return acc, precision, recall, f1, auc, y_pred


def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return float(best_threshold), float(best_f1)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_f1": [],
    "val_auc": [],
    "val_threshold": []
}

best_val_f1 = -1.0
best_threshold = 0.5
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    if epoch == FREEZE_EPOCHS:
        print("\nUnfreezing DenseNet backbone for fine-tuning...")
        unfreeze_backbone(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR_FINE_TUNE,
            weight_decay=WEIGHT_DECAY
        )

    train_loss, train_acc = train_one_epoch()

    val_loss, y_val, val_probs = evaluate_probs(val_loader)
    threshold, _ = find_best_threshold(y_val, val_probs)

    val_acc, val_precision, val_recall, val_f1, val_auc, _ = metrics_from_threshold(
        y_val,
        val_probs,
        threshold
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)
    history["val_threshold"].append(threshold)

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"Val Precision: {val_precision:.4f}")
    print(f"Val Recall:    {val_recall:.4f}")
    print(f"Val F1:        {val_f1:.4f}")
    print(f"Val AUC:       {val_auc:.4f}")
    print(f"Best threshold this epoch: {threshold:.2f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_threshold = threshold
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_threshold": float(best_threshold),
                "label_classes": list(label_encoder.classes_),
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "feature_cols": feature_cols,
                "tabular_input_dim": int(tabular_input_dim)
            },
            MODEL_PATH
        )

        print("Best DAFT Phase-2 model saved:", MODEL_PATH)
    else:
        epochs_without_improvement += 1
        print(f"No improvement: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


epochs_range = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DAFT Phase-2 Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "daft_phase2_loss_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DAFT Phase-2 Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "daft_phase2_accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["val_f1"], label="Validation F1")
plt.plot(epochs_range, history["val_auc"], label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("DAFT Phase-2 Validation F1 and AUC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "daft_phase2_f1_auc_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["val_threshold"], label="Best Threshold")
plt.xlabel("Epoch")
plt.ylabel("Threshold")
plt.title("DAFT Phase-2 Best Validation Threshold per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "daft_phase2_threshold_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nLoading best DAFT Phase-2 model for final test...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
best_threshold = checkpoint["best_threshold"]

print("Best validation threshold:", best_threshold)

test_loss, y_test, test_probs = evaluate_probs(test_loader)

test_acc, test_precision, test_recall, test_f1, test_auc, y_pred = metrics_from_threshold(
    y_test,
    test_probs,
    best_threshold
)

print("\n===== DAFT PHASE-2 TEST RESULTS =====")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1:        {test_f1:.4f}")
print(f"Test AUC:       {test_auc:.4f}")
print(f"Threshold:      {best_threshold:.2f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False)

history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "daft_phase2_training_history.csv"), index=False)

pred_df = pd.DataFrame({
    "true_label": y_test,
    "prob_melanoma": test_probs,
    "pred_label": y_pred
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "daft_phase2_test_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "daft_phase2_test_report.txt"), "w") as f:
    f.write("===== DAFT PHASE-2 TEST RESULTS =====\n")
    f.write(f"Device: {DEVICE}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write("\n")
    f.write(f"Test Loss:      {test_loss:.4f}\n")
    f.write(f"Test Accuracy:  {test_acc:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall:    {test_recall:.4f}\n")
    f.write(f"Test F1:        {test_f1:.4f}\n")
    f.write(f"Test AUC:       {test_auc:.4f}\n")
    f.write(f"Threshold:      {best_threshold:.2f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")
    f.write("Method:\n")
    f.write("DAFT Phase-2 DenseNet121\n")
    f.write("Changes:\n")
    f.write("- Smaller DAFT MLP\n")
    f.write("- Stronger dropout\n")
    f.write("- Focal loss with label smoothing\n")
    f.write("- Partial weighted oversampling\n")
    f.write("- Threshold tuning\n")

print("\nSaved everything to:")
print(OUTPUT_DIR)