#  Melanoma Skin Cancer Detection Using CNN

> A binary image classification system to detect benign vs. malignant melanoma lesions using a custom CNN, with two preprocessing pipelines compared — one with Histogram Equalization and one without.

---

##  Overview

Melanoma is one of the most dangerous forms of skin cancer, and early detection is critical to improving survival rates. Manual diagnosis by dermatologists is time-consuming and subject to inter-observer variability. This project builds an automated deep learning pipeline that classifies dermoscopic images as either **benign** or **malignant**, comparing the impact of histogram equalization on model performance.

---

## Objectives

- Preprocess dermoscopic images to remove noise, hair artifacts, and enhance contrast before model training.
- Train a custom CNN architecture on the preprocessed melanoma dataset.
- Compare two preprocessing variants: one without histogram equalization (Notebook 1) and one with adaptive histogram equalization on the Y channel in YCbCr space (Notebook 2).
- Evaluate model performance on a held-out test set using accuracy, precision, recall, and confusion matrix.

---

## Dataset

- **Source:** Melanoma cancer dataset, split into `train/` and `test/` folders, each containing `benign/` and `malignant/` subfolders.
- **Number of samples:** 11,000 total — 5,500 benign, 5,500 malignant (train + test combined). Training split: 10,000 images; test set: 1,000 images.
- **Data type:** RGB dermoscopic JPEG/PNG images.
- **Preprocessing:** Yes — hair removal, noise reduction, contrast stretching, and (in Notebook 2) adaptive histogram equalization. Class balance was verified; no duplicates were found between train and test sets.

---

##  Methodology

### 1. Data Preprocessing

**Both notebooks share the following pipeline:**

- **Hair removal:** Grayscale conversion → Gaussian blur (21×21 kernel) → arithmetic subtraction to isolate hair → binary threshold → morphological closing → inpainting on the original color image.
- **Noise reduction:** Median blur (3×3 kernel).
- **Contrast stretching:** Min-max normalization to the full [0, 255] range.

**Notebook 2 additionally applies:**
- **Adaptive Histogram Equalization:** Image converted BGR → RGB → YCbCr; histogram equalization applied to the Y (luminance) channel only using `ImageOps.equalize`; channels merged and converted back.

All processed images are saved to a `processed_dataset/` folder with a dated subfolder name (e.g., `benign_20260430`).

### 2. Segmentation (Notebook 1 only)

- **Method:** Grayscale mean-threshold segmentation followed by bounding-box cropping with a 5-pixel margin.
- **Purpose:** Isolate the lesion region and reduce background noise before feeding into the model. Cropped images are saved to `benign_cropped/` and `malignant_cropped/` subfolders.

### 3. Feature Extraction

Features are learned end-to-end by the CNN. No handcrafted features were extracted. Images are resized to the model's expected input size before training.

### 4. Model

This project evaluates four deep learning models for binary melanoma classification: CNN, ResNet50, ResNet50V2, and DenseNet121.

---

## Experiments

| Setting | Notebook 1 (No HE) | Notebook 2 (With HE) |
|---|---|---|
| Input size | 224×224 | 300×300 |
| Train samples | 7,684 | 8,000 |
| Validation samples | 1,921 | 2,000 |
| Test samples | 1,000 | 1,000 |
| Train/Val split | 80/20 (stratified) | 80/20 (stratified) |
| Epochs | 15 | 15 |
| Batch size | 32 | 32 |
| Total params | 12,938,561 (~49 MB) | 22,523,201 (~86 MB) |

---

##  Results

All models were evaluated on the same 1,000-image held-out test set (500 benign / 500 malignant).

| Model | Preprocessing | Test Accuracy | Test Loss | Precision | Recall | AUC |
|-------|--------------|:---:|:---:|:---:|:---:|:---:|
| CNN | Without HE (segmentation) | **91.40%** | 0.2696 | 0.9404 | 0.8840 | — |
| CNN | With HE (no segmentation) | 59.20% | 1.2321 | 0.5537 | 0.9480 | — |
| ResNet50 | Without HE (segmentation) | 81.30% | 0.4182 | 0.9003 | 0.7040 | 0.8981 |
| ResNet50 | With HE (no segmentation) | 86.60% | 0.3592 | 0.8720 | 0.8580 | 0.9257 |
| ResNet50V2 | Without HE (segmentation) | 90.10% | 0.2420 | 0.9203 | 0.8780 | 0.9589 |
| ResNet50V2 | With HE (no segmentation) | 90.80% | 0.2445 | 0.9340 | 0.8780 | 0.9615 |
| DenseNet121 | Without HE (segmentation) | 90.10% | 0.2592 | 0.9221 | 0.8760 | 0.9556 |
| DenseNet121 | With HE (no segmentation) | 89.00% | 0.2606 | 0.8996 | 0.8780 | 0.9545 |

> **Key finding:** The custom CNN with segmentation achieves the highest test accuracy (91.40%). Among transfer learning models, ResNet50V2 with histogram equalization performs best (90.80%, AUC 0.9615). CNN without histogram equalization suffers severe overfitting (train 99.19% → test 59.20%), while histogram equalization consistently benefits the transfer learning models.

---

##  Sample Outputs

The notebooks visualize:
- Before/after comparisons for each preprocessing step (hair removal, noise reduction, contrast stretching, histogram equalization).
- Class distribution plots (pie chart and bar chart) confirming balance.
- Training and validation accuracy/loss curves over 15 epochs.
- Confusion matrix heatmap on the test set.
- First 12 misclassified images with true vs. predicted labels.

---

##  How to Run

```bash
git clone <repo-link>
cd melanoma-detection
pip install -r requirements.txt
```

Edit the dataset paths at the top of each notebook:
```python
TRAIN_ROOT = r"path/to/melanoma_cancer_dataset/train"
TEST_ROOT  = r"path/to/melanoma_cancer_dataset/test"
OUTPUT_ROOT = r"path/to/processed_dataset"
```


**Dependencies:** `tensorflow`, `keras`, `opencv-python`, `Pillow`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`
