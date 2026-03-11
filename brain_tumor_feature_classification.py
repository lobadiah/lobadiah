# %% [markdown]
# # Project 1: Dataset Classification using Feature-Based Learning
# ## Brain Tumor Classification (Tumor vs Non-Tumor) using Handcrafted Features + Machine Learning
#
# **Goal:** Build a classification system using traditional feature extraction + machine learning.
#
# **Rubric Steps**
# 1. **Dataset Preparation:** load dataset, split into train/validation/test, apply preprocessing (resizing/normalization).
# 2. **Feature Extraction:** handcrafted features (HOG, LBP, intensity/color histograms, edge descriptors).
# 3. **Classification:** train classifiers (SVM, Random Forest, k-NN, Logistic Regression).
# 4. **Performance Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.

# %% [markdown]
# ## 1) Import Required Libraries + Global Settings

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    display = print

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from skimage.feature import hog, local_binary_pattern
from skimage import filters, color
from skimage.io import imread
from skimage.transform import resize

from pathlib import Path
from collections import Counter
import warnings
import time
import os
import json
import joblib

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✅ Libraries imported and settings configured.")

# %% [markdown]
# ## 2) Dataset Preparation
#
# **Rubric Step 1:** load dataset, preprocessing (resize/normalize), then split into Train/Val/Test.
#
# Dataset path: C:\Users\louis\Desktop\ML Project\brain_tumor_dataset
# Expected folder structure:
# ```
# brain_tumor_dataset/
#   yes/   (tumor images)
#   no/    (non-tumor images)
# ```

# %% [markdown]
# ### 2.1 Dataset Loader

# %%
class BrainTumorDataset:
    """
    Load images from:
    data_path/
        yes/  (tumor)
        no/   (non-tumor)
    """
    def __init__(self, data_path, img_size=(128, 128)):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.class_names = []

    def load_dataset(self):
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)

        classes = [d for d in self.data_path.iterdir() if d.is_dir()]
        classes = sorted(classes, key=lambda p: p.name)  # stable order
        self.class_names = [c.name for c in classes]

        for class_idx, class_dir in enumerate(classes):
            print(f"\nLoading '{class_dir.name}' images...")

            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            for img_path in image_files:
                try:
                    img = imread(img_path)

                    # Convert to grayscale if RGB
                    if len(img.shape) == 3:
                        img = color.rgb2gray(img)

                    # Resize
                    img = resize(img, self.img_size, anti_aliasing=True)

                    # Normalize to [0, 1]
                    img = img.astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                    self.images.append(img)
                    self.labels.append(class_idx)

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        print("\n✅ Dataset loaded successfully!")
        print(f"   Total images: {len(self.images)}")
        print(f"   Image shape: {self.images[0].shape}")
        print(f"   Classes: {self.class_names}")
        print(f"   Class distribution: {Counter(self.labels)}")

        return self.images, self.labels, self.class_names

# %% [markdown]
# ### 2.2 Synthetic Dataset (optional demo — skip if using real dataset)

# %%
def create_synthetic_data(n_samples=300, img_size=(128, 128)):
    np.random.seed(RANDOM_STATE)
    images, labels = [], []

    for i in range(n_samples):
        img = np.random.randn(*img_size)

        if i < n_samples // 2:  # Tumor
            center = np.random.randint(30, 100, 2)
            img[center[0]-15:center[0]+15, center[1]-15:center[1]+15] += 2
            labels.append(1)
        else:  # Non-tumor
            labels.append(0)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images.append(img.astype(np.float32))

    return np.array(images), np.array(labels)

# %% [markdown]
# ### 2.3 Load Data + Quick EDA (class distribution + sample images)
#
# ✅ Real dataset path set to:
# C:\Users\louis\Desktop\ML Project\brain_tumor_dataset
#
# Folder structure expected:
# brain_tumor_dataset/
#   yes/   (tumor)
#   no/    (non-tumor)
#
# Set USE_SYNTHETIC = True only if you do not have the dataset locally.

# %%
USE_SYNTHETIC = False
IMG_SIZE = (128, 128)

if USE_SYNTHETIC:
    print("Using synthetic dataset (demo mode).")
    images, labels = create_synthetic_data(n_samples=300, img_size=IMG_SIZE)
    class_names = ["No Tumor", "Tumor"]
else:
    # Raw string r"..." handles Windows backslashes correctly
    data_path = r"C:\Users\louis\Desktop\ML Project\brain_tumor_dataset"
    dataset = BrainTumorDataset(data_path, img_size=IMG_SIZE)
    images, labels, folder_class_names = dataset.load_dataset()

    # Map folder names to display labels (alphabetical order: no=0, yes=1)
    if set(folder_class_names) == {"no", "yes"}:
        class_names = ["No Tumor", "Tumor"]
    else:
        class_names = folder_class_names

print("\nDataset statistics:")
print(f"  Total samples: {len(images)}")
for i, name in enumerate(class_names):
    count = int(np.sum(labels == i))
    print(f"  {name}: {count} ({count/len(images)*100:.1f}%)")

# Visualize sample images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

unique_labels = np.unique(labels)
if len(unique_labels) < 2:
    raise ValueError("Dataset appears to have <2 classes. Check folder structure.")

idx_class0 = np.where(labels == 0)[0]
idx_class1 = np.where(labels == 1)[0]

n_show = 4
if len(idx_class0) < n_show or len(idx_class1) < n_show:
    raise ValueError("Not enough images in one class to display 4 samples.")

for i in range(n_show):
    axes[0, i].imshow(images[idx_class0[i]], cmap="gray")
    axes[0, i].set_title(f"{class_names[0]} {i+1}")
    axes[0, i].axis("off")

    axes[1, i].imshow(images[idx_class1[i]], cmap="gray")
    axes[1, i].set_title(f"{class_names[1]} {i+1}")
    axes[1, i].axis("off")

plt.suptitle("Sample Images", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.4 Split into Train / Validation / Test (Rubric Step 1)
# We split **images** first, then extract features for each split separately.
# This avoids accidental mixing and keeps the workflow clear.

# %%
idx = np.arange(len(images))

idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx, labels, test_size=0.30, random_state=RANDOM_STATE, stratify=labels
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

images_train = images[idx_train]
images_val   = images[idx_val]
images_test  = images[idx_test]

print("Split sizes:")
print(f"  Train: {len(images_train)}")
print(f"  Val:   {len(images_val)}")
print(f"  Test:  {len(images_test)}")

# %% [markdown]
# ## 3) Feature Extraction (Rubric Step 2)
#
# Extract handcrafted features:
# - HOG
# - LBP histogram
# - Intensity histogram
# - Edge statistics (Sobel)

# %%
class FeatureExtractor:
    def __init__(self, hog_params=None, lbp_params=None, hist_bins=32):
        self.hog_params = hog_params or {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
        }
        self.lbp_params = lbp_params or {
            "radius": 3,
            "n_points": 24,
            "method": "uniform",
        }
        self.hist_bins = hist_bins
        self.feature_names_ = None

    def _hog(self, image):
        return hog(
            image,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            block_norm=self.hog_params["block_norm"],
            feature_vector=True,
        )

    def _lbp_hist(self, image):
        lbp = local_binary_pattern(
            image,
            self.lbp_params["n_points"],
            self.lbp_params["radius"],
            self.lbp_params["method"],
        )
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist

    def _intensity_hist(self, image):
        hist, _ = np.histogram(image.ravel(), bins=self.hist_bins, range=(0, 1), density=True)
        return hist

    def _edge_stats(self, image):
        edges_x = filters.sobel_h(image)
        edges_y = filters.sobel_v(image)
        mag = np.sqrt(edges_x**2 + edges_y**2)

        return np.array([
            np.mean(mag),
            np.std(mag),
            np.percentile(mag, 25),
            np.percentile(mag, 75),
            np.sum(mag > 0.3) / mag.size,
        ], dtype=np.float32)

    def extract_one(self, image):
        hog_feat  = self._hog(image)
        lbp_feat  = self._lbp_hist(image)
        hist_feat = self._intensity_hist(image)
        edge_feat = self._edge_stats(image)

        if self.feature_names_ is None:
            names  = [f"HOG_{i}"  for i in range(len(hog_feat))]
            names += [f"LBP_{i}"  for i in range(len(lbp_feat))]
            names += [f"Hist_{i}" for i in range(len(hist_feat))]
            names += ["Edge_Mean", "Edge_Std", "Edge_Q1", "Edge_Q3", "Edge_Density"]
            self.feature_names_ = names

        return np.concatenate([hog_feat, lbp_feat, hist_feat, edge_feat])

    def transform(self, images):
        feats = []
        for i, img in enumerate(images):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(images)} images")
            feats.append(self.extract_one(img))
        return np.array(feats)

# %% [markdown]
# ### 3.1 Extract Features for Train / Val / Test + Standardize

# %%
print("=" * 60)
print("FEATURE EXTRACTION (Train/Val/Test)")
print("=" * 60)

extractor = FeatureExtractor()

print("\nTrain features:")
X_train = extractor.transform(images_train)

print("\nValidation features:")
X_val = extractor.transform(images_val)

print("\nTest features:")
X_test = extractor.transform(images_test)

print("\nShapes:")
print("  X_train:", X_train.shape)
print("  X_val:  ", X_val.shape)
print("  X_test: ", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# %% [markdown]
# ## 4) Classification (Rubric Step 3)
# Train multiple classifiers using GridSearchCV:
# - SVM
# - Random Forest
# - k-NN
# - Logistic Regression

# %%
class ModelTrainer:
    def __init__(self):
        self.models = {
            "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        }
        self.best_params = {}
        self.training_times = {}
        self.results = {}

    def train_with_gridsearch(self, model_name, model, param_grid, X_train, y_train):
        grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
        t0 = time.time()
        grid.fit(X_train, y_train)
        t1 = time.time()
        self.best_params[model_name] = grid.best_params_
        self.training_times[model_name] = t1 - t0
        return grid.best_estimator_

    def evaluate(self, model, model_name, X_eval, y_eval):
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, average="binary")
        rec  = recall_score(y_eval, y_pred, average="binary")
        f1   = f1_score(y_eval, y_pred, average="binary")
        cm   = confusion_matrix(y_eval, y_pred)

        self.results[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        return self.results[model_name]

    def train_all(self, X_train, y_train, X_val, y_val, X_test, y_test):
        param_grids = {
            "SVM": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto", 0.01],
                "kernel": ["rbf", "poly"],
            },
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "k-NN": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            "Logistic Regression": {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            },
        }

        trained = {}
        val_acc = {}

        print("=" * 60)
        print("TRAINING + VALIDATION SELECTION")
        print("=" * 60)

        for model_name, model in self.models.items():
            print(f"\n--- {model_name} ---")
            best = self.train_with_gridsearch(model_name, model, param_grids[model_name], X_train, y_train)
            trained[model_name] = best

            val_res = self.evaluate(best, f"{model_name} (val)", X_val, y_val)
            val_acc[model_name] = val_res["accuracy"]

            print("Best params:", self.best_params[model_name])
            print(f"Val accuracy: {val_res['accuracy']:.4f}")
            print(f"Training time: {self.training_times[model_name]:.2f}s")

        best_model_name = max(val_acc, key=val_acc.get)
        best_model = trained[best_model_name]
        print(f"\n🏆 Best model by validation accuracy: {best_model_name}")

        print("\n" + "=" * 60)
        print("TEST EVALUATION (All Models)")
        print("=" * 60)

        for model_name, model in trained.items():
            self.evaluate(model, model_name, X_test, y_test)

        return trained, best_model_name, best_model

# %% [markdown]
# ### 4.1 Run Training

# %%
trainer = ModelTrainer()
trained_models, best_model_name, best_model = trainer.train_all(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    X_test_scaled, y_test
)

# %% [markdown]
# ## 5) Performance Evaluation (Rubric Step 4)
# Required:
# - Accuracy
# - Precision
# - Recall
# - F1-score
# - Confusion Matrix

# %% [markdown]
# ### 5.1 Metrics Table (Test Set)

# %%
results_df = pd.DataFrame(trainer.results).T

# Show only TEST rows (exclude "(val)")
results_df_test = results_df[~results_df.index.str.contains(r"\(val\)", regex=True)].copy()
results_df_test = results_df_test[["accuracy", "precision", "recall", "f1"]].round(4)

print("Model performance on TEST set:")
display(results_df_test)

# %% [markdown]
# ### 5.2 Confusion Matrices (Test Set)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, model_name in enumerate(results_df_test.index):
    cm = trainer.results[model_name]["confusion_matrix"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
        xticklabels=class_names, yticklabels=class_names
    )
    axes[idx].set_title(f"{model_name} - Confusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.3 Classification Report (Best Model)

# %%
best_pred = trainer.results[best_model_name]["y_pred"]
print(f"Classification report for best model: {best_model_name}\n")
print(classification_report(y_test, best_pred, target_names=class_names))

# %% [markdown]
# ## 6) Save Notebook Outputs (Recommended)
# This creates:
# - `reports/model_comparison_test.csv`
# - `reports/detailed_results_test.json`
# - `models/best_<ModelName>.pkl`, `models/scaler.pkl`, `models/feature_extractor.pkl`

# %%
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df_test.to_csv("reports/model_comparison_test.csv", index=True)

detailed_results = {}
for model_name in results_df_test.index:
    r = trainer.results[model_name]
    detailed_results[model_name] = {
        "accuracy": float(r["accuracy"]),
        "precision": float(r["precision"]),
        "recall": float(r["recall"]),
        "f1": float(r["f1"]),
        "confusion_matrix": r["confusion_matrix"].tolist(),
    }

with open("reports/detailed_results_test.json", "w") as f:
    json.dump(detailed_results, f, indent=4)

best_model_path = f"models/best_{best_model_name.replace(' ', '_')}.pkl"
joblib.dump(best_model, best_model_path)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(extractor, "models/feature_extractor.pkl")

print("✅ Saved reports and models.")
print("  -", "reports/model_comparison_test.csv")
print("  -", "reports/detailed_results_test.json")
print("  -", best_model_path)
print("  -", "models/scaler.pkl")
print("  -", "models/feature_extractor.pkl")

# %% [markdown]
# ## 7) Bonus: Predict on a New Image (Optional)

# %%
def predict_new_image(image_path, model, scaler, extractor, class_names, img_size=(128, 128)):
    img = imread(image_path)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    img = resize(img, img_size, anti_aliasing=True)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    feat = extractor.extract_one(img)
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    pred = model.predict(feat_scaled)[0]
    proba = model.predict_proba(feat_scaled)[0] if hasattr(model, "predict_proba") else None

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if proba is not None:
        plt.bar(class_names, proba)
        plt.title(f"Prediction: {class_names[pred]}")
        plt.ylabel("Probability")
    else:
        plt.text(0.1, 0.5, f"Prediction: {class_names[pred]}", fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return pred, proba

# Example usage (uncomment and set path):
# pred, proba = predict_new_image(
#     r"C:\Users\louis\Desktop\ML Project\brain_tumor_dataset\yes\image(1).jpg",
#     best_model, scaler, extractor, class_names
# )
# print("Prediction:", class_names[pred], "  Probabilities:", proba)
