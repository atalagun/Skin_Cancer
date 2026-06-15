import os
import cv2
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

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

TRAIN_CSV = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/train_split.csv"
VAL_CSV   = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/val_split.csv"
TEST_CSV  = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_daft_phase2/test_split.csv"

OUTPUT_DIR = r"C:/Users/Ata/PycharmProjects/SkinCancerData/outputs_Run6_rgb_clahe_canny_crop_daft"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_rgb_clahe_canny_crop_daft.pth")

IMG_SIZE = 300
BATCH_SIZE = 8
NUM_EPOCHS = 25
FREEZE_EPOCHS = 5
PATIENCE = 6

LR_HEAD = 1e-4
LR_FINE_TUNE = 5e-6
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

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

IMAGE_COL = "image_path"

if "label_encoded" not in train_df.columns:
    mapping = {"no": 0, "yes": 1}
    train_df["label_encoded"] = train_df["midas_melanoma"].astype(str).str.lower().map(mapping)
    val_df["label_encoded"] = val_df["midas_melanoma"].astype(str).str.lower().map(mapping)
    test_df["label_encoded"] = test_df["midas_melanoma"].astype(str).str.lower().map(mapping)

LABEL_COL = "label_encoded"

train_df = train_df[train_df[IMAGE_COL].apply(lambda x: isinstance(x, str) and os.path.exists(x))].reset_index(drop=True)
val_df = val_df[val_df[IMAGE_COL].apply(lambda x: isinstance(x, str) and os.path.exists(x))].reset_index(drop=True)
test_df = test_df[test_df[IMAGE_COL].apply(lambda x: isinstance(x, str) and os.path.exists(x))].reset_index(drop=True)

print("\nAfter image filtering:")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("\nTrain label distribution:")
print(train_df[LABEL_COL].value_counts(normalize=True))


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

numeric_cols = [c for c in numeric_cols if c in train_df.columns]
categorical_cols = [c for c in categorical_cols if c in train_df.columns]

valid_numeric_cols = []

for col in numeric_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    if train_df[col].notna().sum() == 0:
        print("Dropping numeric column:", col)
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
num_classes = 2

print("Tabular input dim:", tabular_input_dim)


def auto_crop_lesion_rgb(img_rgb, padding=30):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img_rgb

    h, w = gray.shape
    image_area = h * w

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if 0.002 * image_area < area < 0.80 * image_area:
            valid.append(c)

    if len(valid) == 0:
        return img_rgb

    c = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    crop = img_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return img_rgb

    return crop


def apply_clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return out


def auto_canny_channel(img_rgb, sigma=0.33):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray, lower, upper)
    return edges


class RandomGamma:
    def __init__(self, gamma_range=(0.98, 1.02), p=0.15):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return F.adjust_gamma(img, gamma)
        return img


train_rgb_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    RandomGamma(gamma_range=(0.98, 1.02), p=0.15),
])

eval_rgb_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])


def pil_to_4ch_tensor(pil_img):
    img_rgb = np.array(pil_img).astype(np.uint8)

    canny = auto_canny_channel(img_rgb).astype(np.float32) / 255.0

    img = np.array(pil_img).astype(np.float32) / 255.0

    img_4ch = np.dstack([
        img[:, :, 0],
        img[:, :, 1],
        img[:, :, 2],
        canny
    ])

    img_4ch = torch.tensor(img_4ch).permute(2, 0, 1).float()

    mean = torch.tensor([0.485, 0.456, 0.406, 0.0]).view(4, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 1.0]).view(4, 1, 1)

    img_4ch = (img_4ch - mean) / std

    return img_4ch


class MIDASCannyDataset(Dataset):
    def __init__(self, dataframe, tabular_array, train=True):
        self.df = dataframe.reset_index(drop=True)
        self.tabular = tabular_array
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row[IMAGE_COL]
        label = int(row[LABEL_COL])

        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_rgb = auto_crop_lesion_rgb(img_rgb, padding=30)
        img_rgb = apply_clahe_rgb(img_rgb)

        pil_img = Image.fromarray(img_rgb)

        if self.train:
            pil_img = train_rgb_transform(pil_img)
        else:
            pil_img = eval_rgb_transform(pil_img)

        image_tensor = pil_to_4ch_tensor(pil_img)

        tabular_tensor = torch.tensor(self.tabular[idx], dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, tabular_tensor, label_tensor


train_dataset = MIDASCannyDataset(train_df, X_train_tab, train=True)
val_dataset = MIDASCannyDataset(val_df, X_val_tab, train=False)
test_dataset = MIDASCannyDataset(test_df, X_test_tab, train=False)


train_labels = train_df[LABEL_COL].values.astype(int)
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


class DAFTEfficientNetB3Canny(nn.Module):
    def __init__(self, tabular_dim, num_classes):
        super().__init__()

        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        old_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        backbone.features[0][0] = new_conv

        self.features = backbone.features

        feature_channels = backbone.classifier[1].in_features

        self.daft = DAFTBlock(
            tabular_dim=tabular_dim,
            feature_channels=feature_channels
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(feature_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):
        x = self.features(image)
        x = self.daft(x, tabular)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = DAFTEfficientNetB3Canny(tabular_input_dim, num_classes).to(DEVICE)


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


criterion = FocalLossWithSmoothing(alpha=[0.38, 0.62], gamma=1.5, smoothing=0.03)

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

    return total_loss / len(train_loader.sampler), accuracy_score(all_labels, all_preds)


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

    return total_loss / len(loader.dataset), np.array(all_labels), np.array(all_probs)


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
        print("\nUnfreezing EfficientNet-B3 backbone...")
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
        y_val, val_probs, threshold
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
                "tabular_input_dim": int(tabular_input_dim),
                "img_size": IMG_SIZE,
                "model": "EfficientNet-B3 + DAFT + CLAHE + Canny + AutoCrop"
            },
            MODEL_PATH
        )

        print("Best model saved:", MODEL_PATH)
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
plt.title("RGB+CLAHE+Canny+Crop DAFT Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("RGB+CLAHE+Canny+Crop DAFT Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["val_f1"], label="Validation F1")
plt.plot(epochs_range, history["val_auc"], label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("RGB+CLAHE+Canny+Crop DAFT F1 and AUC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "f1_auc_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["val_threshold"], label="Best Threshold")
plt.xlabel("Epoch")
plt.ylabel("Threshold")
plt.title("Best Validation Threshold per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "threshold_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nLoading best model for test...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
best_threshold = checkpoint["best_threshold"]

test_loss, y_test, test_probs = evaluate_probs(test_loader)

test_acc, test_precision, test_recall, test_f1, test_auc, y_pred = metrics_from_threshold(
    y_test, test_probs, best_threshold
)

print("\n===== TEST RESULTS =====")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1:        {test_f1:.4f}")
print(f"Test AUC:       {test_auc:.4f}")
print(f"Threshold:      {best_threshold:.2f}")

report = classification_report(y_test, y_pred, target_names=["no", "yes"])
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

pd.DataFrame({
    "true_label": y_test,
    "prob_melanoma": test_probs,
    "pred_label": y_pred
}).to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
    f.write("===== TEST RESULTS =====\n")
    f.write(f"Device: {DEVICE}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
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
    f.write("\n\nMethod:\n")
    f.write("EfficientNet-B3 + DAFT + 4-channel RGB+Canny input + CLAHE + auto lesion crop\n")

print("\nSaved everything to:")
print(OUTPUT_DIR)