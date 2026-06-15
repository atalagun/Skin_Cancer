import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
OUTPUT_DIR = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run1_less_overfit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_less_overfit.pth")

BATCH_SIZE = 16
NUM_EPOCHS = 25
FREEZE_EPOCHS = 5
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINE_TUNE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 6

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.benchmark = True

df = pd.read_excel(EXCEL_PATH)

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

print("After label cleaning:", df.shape)
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

label_encoder = LabelEncoder()
train_df["label_encoded"] = label_encoder.fit_transform(train_df[label_col])
val_df["label_encoded"] = label_encoder.transform(val_df[label_col])
test_df["label_encoded"] = label_encoder.transform(test_df[label_col])

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
print("Classes:", label_encoder.classes_)


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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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

class MultimodalMelanomaModel(nn.Module):
    def __init__(self, tabular_dim, num_classes):
        super().__init__()

        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        image_feature_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()

        self.image_backbone = backbone

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(image_feature_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):
        img_feat = self.image_backbone(image)
        tab_feat = self.tabular_net(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.fusion_net(fused)

model = MultimodalMelanomaModel(tabular_input_dim, num_classes).to(DEVICE)
def freeze_backbone(model):
    for param in model.image_backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.image_backbone.parameters():
        param.requires_grad = True

freeze_backbone(model)
train_labels = train_df["label_encoded"].values
class_counts = np.bincount(train_labels)
class_weights = len(train_labels) / (num_classes * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE_HEAD,
    weight_decay=WEIGHT_DECAY
)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, tabular, labels in loader:
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

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
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
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, precision, recall, f1, auc, all_labels, all_preds


history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_f1": [],
    "val_auc": []
}

best_val_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    if epoch == FREEZE_EPOCHS:
        print("\nUnfreezing DenseNet backbone for fine-tuning...")
        unfreeze_backbone(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE_FINE_TUNE,
            weight_decay=WEIGHT_DECAY
        )

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, _, _ = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"Val Precision: {val_precision:.4f}")
    print(f"Val Recall:    {val_recall:.4f}")
    print(f"Val F1:        {val_f1:.4f}")
    print(f"Val AUC:       {val_auc:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved:", MODEL_PATH)
    else:
        epochs_without_improvement += 1
        print(f"No improvement count: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

epochs_range = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["val_f1"], label="Validation F1")
plt.plot(epochs_range, history["val_auc"], label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation F1 and AUC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "f1_auc_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nLoading best model for final test...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, y_true, y_pred = evaluate(model, test_loader)

print("\n===== TEST RESULTS =====")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1:        {test_f1:.4f}")
print(f"Test AUC:       {test_auc:.4f}")

report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
cm = confusion_matrix(y_true, y_pred)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)


train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_split.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
    f.write("===== TEST RESULTS =====\n")
    f.write(f"Test Loss:      {test_loss:.4f}\n")
    f.write(f"Test Accuracy:  {test_acc:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall:    {test_recall:.4f}\n")
    f.write(f"Test F1:        {test_f1:.4f}\n")
    f.write(f"Test AUC:       {test_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nSaved everything to:")
print(OUTPUT_DIR)