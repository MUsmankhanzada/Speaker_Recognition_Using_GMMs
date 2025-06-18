# 🎙️ Speaker Recognition using MFCC and Gaussian Mixture Models (GMM)

**Author:** Muhammad Usman Khanzada  
**Course:** Machine Learning – Class Activity 4  
**Notebook:** `Class_Activity_4_ML.ipynb`  
**Approach:** MFCC Feature Extraction + GMM Modeling  

---

## 📌 Objective

Build a speaker recognition system by extracting MFCC features from audio files and modeling speaker characteristics using Gaussian Mixture Models (GMM). This project focuses on **speaker identification**, where the system assigns a test utterance to one of several known speakers.

---

## 📁 Dataset

- **Source:** TIMIT Speech Corpus (or similarly structured dataset)
- **Structure:** `.WAV` files organized by speaker folders
- **Format:** PCM audio sampled at consistent rates

---

## 🔄 Project Pipeline

### 1. MFCC Feature Extraction
- Extracts 13 MFCCs per frame using `librosa`
- Features are saved as `.npy` files
- Folder structure of the dataset is preserved

### 2. Data Splitting
- Splits extracted features into:
  - `train/` (80%)
  - `test/` (20%)
- Uses `shutil` and `random` for shuffling and copying files

### 3. GMM Training
- Concatenates all training MFCCs per speaker
- Trains a Gaussian Mixture Model (`K=13`)
- Uses Expectation-Maximization (EM) algorithm
- Saves model parameters (`mean`, `covariance`, `weights`) in `.npy` format

### 4. Visualization
- MFCC heatmaps
- GMM clustering (scatter plot)
- Log-likelihood curve over iterations

---


