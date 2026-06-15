import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

INPUT_DIR  = r"C:/Users/Ata/midasmultimodalimagedatasetforaibasedskincancer"
OUTPUT_DIR = r"C:/Users/Ata/Desktop/image_processing_results"
MAX_IMAGES = 5
IMG_SIZE   = 300
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_images(root, max_n):
    found = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                found.append(os.path.join(r, f))
        if len(found) >= max_n * 20:
            break
    if len(found) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    random.shuffle(found)
    return found[:max_n]

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def save_rgb(arr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(out_path)

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def apply_resize(img_rgb):
    return cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

def apply_hflip(img_rgb):
    return cv2.flip(img_rgb, 1)

def apply_vflip(img_rgb):
    return cv2.flip(img_rgb, 0)

def apply_rotation(img_rgb, angle=10):
    h, w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_rgb, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def apply_colour_jitter(img_rgb, brightness=0.08, contrast=0.08, saturation=0.05):
    tfm = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    return np.array(tfm(Image.fromarray(img_rgb)))

def apply_gaussian_5x5(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(cv2.GaussianBlur(gray, (5, 5), 0), cv2.COLOR_GRAY2RGB)

def apply_gaussian_3x3(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2RGB)

def apply_otsu(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

def apply_morph_open(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.cvtColor(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.COLOR_GRAY2RGB)

def apply_morph_close(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel), cv2.COLOR_GRAY2RGB)

def apply_auto_crop(img_rgb, padding=30):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    h, w = gray.shape
    image_area = h * w
    valid = [c for c in contours if 0.002 * image_area < cv2.contourArea(c) < 0.80 * image_area]
    if not valid:
        return img_rgb
    x, y, bw, bh = cv2.boundingRect(max(valid, key=cv2.contourArea))
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(w, x + bw + padding), min(h, y + bh + padding)
    crop = img_rgb[y1:y2, x1:x2]
    return img_rgb if crop.size == 0 else cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def apply_canny(img_rgb, sigma=0.33):
    gray = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    v = np.median(gray)
    edges = cv2.Canny(gray, int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v)))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_blackhat(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    return cv2.cvtColor(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel), cv2.COLOR_GRAY2RGB)

def apply_hair_removal(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img_rgb, mask, 3, cv2.INPAINT_TELEA)

def apply_cutmix(img_a, img_b, lam=0.7):
    h, w = img_a.shape[:2]
    cut_ratio = np.sqrt(1 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
    x2, y2 = min(w, cx + cut_w // 2), min(h, cy + cut_h // 2)
    mixed = img_a.copy()
    mixed[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]
    cv2.rectangle(mixed, (x1, y1), (x2, y2), (255, 0, 0), 2)
    actual_lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed, actual_lam

def apply_mixup(img_a, img_b, lam=0.75):
    mixed = lam * img_a.astype(np.float32) + (1 - lam) * img_b.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)

def process_single(img_path, out_stem, img_b_path=None):
    resized = apply_resize(load_rgb(img_path))
    results = {}

    def add(label, arr, folder):
        results[label] = arr
        save_rgb(arr, os.path.join(OUTPUT_DIR, folder, f"{out_stem}.png"))

    results["01 Original"] = resized.copy()
    add("02 Resize 300x300",        apply_resize(load_rgb(img_path)),  "01_resize")
    add("03 H-Flip",                apply_hflip(resized),              "02_hflip")
    add("04 V-Flip",                apply_vflip(resized),              "03_vflip")
    add("05 Rotation +10°",         apply_rotation(resized),           "04_rotation")
    add("06 Colour Jitter",         apply_colour_jitter(resized),      "05_colour_jitter")
    add("07 Gaussian 5x5\n(pre-Otsu)",  apply_gaussian_5x5(resized),  "06_gaussian_blur_5x5")
    add("08 Gaussian 3x3\n(pre-Canny)", apply_gaussian_3x3(resized),  "07_gaussian_blur_3x3")
    add("09 Otsu Mask",             apply_otsu(resized),               "08_otsu_mask")
    add("10 Morph Open",            apply_morph_open(resized),         "09_morph_open")
    add("11 Morph Close",           apply_morph_close(resized),        "10_morph_close")
    add("12 Auto-Crop",             apply_auto_crop(resized),          "11_auto_crop")
    add("13 CLAHE\n(L-channel)",    apply_clahe(resized),              "12_clahe")
    add("14 Canny Edges\n(4th ch)", apply_canny(resized),              "13_canny_edge")
    add("15 Blackhat\n(hair map)",  apply_blackhat(resized),           "14_blackhat")
    add("16 Hair Removal",          apply_hair_removal(resized),       "15_hair_removal")

    if img_b_path is not None:
        img_b = apply_resize(load_rgb(img_b_path))
        r, lam = apply_cutmix(resized, img_b)
        add(f"17 CutMix\n(λ={lam:.2f})", r,                          "16_cutmix")
        add("18 MixUp\n(λ=0.75)",        apply_mixup(resized, img_b), "17_mixup")
    else:
        results["17 CutMix"] = resized.copy()
        results["18 MixUp"]  = resized.copy()

    return results

def save_comparison_grid(results_dict, save_path, title="Image Processing Methods"):
    labels = list(results_dict.keys())
    images = list(results_dict.values())
    n    = len(images)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    axes = axes.flatten()
    for i, (label, img) in enumerate(zip(labels, images)):
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=8, pad=4)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(save_path)}")

def main():
    print(f"Scanning for images in: {INPUT_DIR}")
    image_paths = find_images(INPUT_DIR, MAX_IMAGES)
    print(f"Found {len(image_paths)} images to process\n")
    img_b = image_paths[1] if len(image_paths) >= 2 else image_paths[0]
    all_results = {}
    for i, path in enumerate(image_paths):
        name = stem(path)
        print(f"[{i+1}/{len(image_paths)}] Processing: {name}")
        results = process_single(
            img_path=path,
            out_stem=name,
            img_b_path=img_b if path != img_b else (
                image_paths[0] if path != image_paths[0] else None)
        )
        all_results[name] = results
        save_comparison_grid(results,
                             save_path=os.path.join(OUTPUT_DIR, f"grid_{name}.png"),
                             title=f"All Methods — {name}")
    print("Generating summary comparison grid...")
    save_comparison_grid(all_results[stem(image_paths[0])],
                         save_path=os.path.join(OUTPUT_DIR, "comparison_grid.png"),
                         title="Complete Image Processing Method Comparison")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    for d in sorted(os.listdir(OUTPUT_DIR)):
        if os.path.isdir(os.path.join(OUTPUT_DIR, d)):
            n = len(os.listdir(os.path.join(OUTPUT_DIR, d)))
            print(f"  {d}/ ({n} file{'s' if n != 1 else ''})")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")

if __name__ == "__main__":
    main()
