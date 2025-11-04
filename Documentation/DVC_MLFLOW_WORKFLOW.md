# DVC + MLflow Workflow Guide

## 🎯 Understanding the Complete MLOps Workflow

Here's the breakdown:

## 📊 Two-Tier Tracking System

### 1. **MLflow** - Experiment Tracking & Model Registry

- **Purpose**: Track hyperparameters, metrics, and model versions
- **Used with**: `main.py` (basic training)
- **Tracks**:
  - Hyperparameters (learning rate, batch size, epochs, etc.)
  - Metrics (accuracy, loss, precision, recall, F1)
  - Training curves (loss/accuracy per epoch)
  - Model artifacts (saved models)
  - Confusion matrices
- **Storage**: Local or remote (DagsHub, MLflow server, S3)

### 2. **DVC** - Data Version Control & Pipeline Orchestration

- **Purpose**:
  - Version control for **heavy files** (data, models, artifacts)
  - Track **experiment comparisons** (different hyperparameter configs)
  - Push large files to **S3** (not GitHub)
  - Reproduce exact experiments
- **Used with**: `dvc.yaml` (pipeline definition)
- **Tracks**:
  - Data files (images, datasets)
  - Model files (.pth)
  - Metrics files (scores.json)
  - Pipeline dependencies
- **Storage**: S3 (for large files), GitHub (only .dvc pointer files)

## 🔄 Complete Workflow

### **Step 1: Initialize DVC** (One-time setup)

```bash
# Initialize DVC repository
dvc init

# Configure remote storage (S3)
dvc remote add -d myremote s3://your-bucket-name/dvc-storage

# Or use DagsHub (if using it)
dvc remote add -d origin https://dagshub.com/username/repo.dvc
```

### **Step 2: Basic Training (MLflow Tracking)**

```bash
# Run training - automatically tracked by MLflow
python main.py
```

**What happens:**

- ✅ MLflow logs all metrics, hyperparameters, models
- ✅ Model saved to `artifacts/training/model.pth`
- ✅ Metrics saved to `scores.json`
- ✅ Training logs in `logs/running_logs.log`

**View results:**

```bash
mlflow ui
# Open http://localhost:5000
```

### **Step 3: Compare Experiments (DVC + MLflow)**

#### A. Experiment with Different Hyperparameters

```bash
# Edit params.yaml
# Change LEARNING_RATE, BATCH_SIZE, etc.

# Run training
python main.py

# Commit with DVC
dvc add artifacts/training/model.pth
dvc add artifacts/data_ingestion/kidney-ct-scan-image
dvc add scores.json
```

#### B. Compare Experiments

```bash
# Compare metrics between runs
dvc metrics diff

# Show all metrics
dvc metrics show

# Compare specific metrics
dvc metrics diff HEAD~1 HEAD
```

#### C. Push to Remote Storage

```bash
# Push large files to S3
dvc push

# Push code and DVC pointers to GitHub
git add .dvc/
git add dvc.yaml
git add params.yaml
git commit -m "Experiment: LR=0.001, BS=32"
git push
```

## 📁 File Structure After DVC Setup

```
project/
├── .dvc/                    # DVC configuration
│   ├── config               # DVC config (remote storage)
│   └── cache/               # Local cache
├── artifacts/
│   ├── data_ingestion/
│   │   └── data.zip.dvc     # Pointer to S3 location
│   └── training/
│       └── model.pth.dvc    # Pointer to S3 location
├── dvc.yaml                 # Pipeline definition
├── dvc.lock                 # Locked versions
├── scores.json              # Metrics file
└── params.yaml              # Hyperparameters
```

## 🎓 Key Concepts

### **GitHub (Git)**

- ✅ **Stores**: Code, config files, small files, `.dvc` pointer files
- ❌ **Does NOT store**: Large data files, model files directly

### **S3 (via DVC)**

- ✅ **Stores**: Large data files, model files, artifacts
- ✅ **Tracked by**: DVC pointer files in Git

### **MLflow**

- ✅ **Stores**: Metrics, hyperparameters, model metadata, training curves
- ✅ **Can store**: Model artifacts (if configured)

## 🔧 Complete Setup Commands

### 1. Initialize DVC (First Time)

```bash
# Initialize DVC
dvc init

# Configure S3 remote (replace with your bucket)
dvc remote add -d myremote s3://your-bucket-name/dvc-storage

# Configure AWS credentials (if using S3)
aws configure
# Or set environment variables:
# export AWS_ACCESS_KEY_ID=your_key
# export AWS_SECRET_ACCESS_KEY=your_secret

# Add .dvc files to Git
git add .dvc/
git commit -m "Initialize DVC"
```

### 2. Track Large Files

```bash
# Track data
dvc add artifacts/data_ingestion/data.zip

# Track model
dvc add artifacts/training/model.pth

# Track metrics (optional, but good for comparison)
dvc add scores.json

# Commit DVC files to Git
git add *.dvc .gitignore
git commit -m "Track data and model with DVC"
```

### 3. Push to Remote

```bash
# Push large files to S3
dvc push

# Push code to GitHub
git push
```

### 4. Pull from Remote (on another machine)

```bash
# Pull code from GitHub
git pull

# Pull large files from S3
dvc pull
```

## 📊 Experiment Comparison Workflow

### **Scenario: Compare Different Learning Rates**

```bash
# Experiment 1: LR = 0.001
# Edit params.yaml: LEARNING_RATE: 0.001
python main.py
dvc add artifacts/training/model.pth scores.json
dvc metrics show
git add params.yaml *.dvc
git commit -m "Experiment: LR=0.001"
dvc push && git push

# Experiment 2: LR = 0.01
# Edit params.yaml: LEARNING_RATE: 0.01
python main.py
dvc add artifacts/training/model.pth scores.json
dvc metrics diff HEAD~1  # Compare with previous
git add params.yaml *.dvc
git commit -m "Experiment: LR=0.01"
dvc push && git push

# View all experiments
dvc metrics show
mlflow ui  # Compare in MLflow UI
```

## 🎯 What Each Tool Does

| Task                              | Tool           | Why                                       |
| --------------------------------- | -------------- | ----------------------------------------- |
| **Track metrics during training** | MLflow         | Real-time logging, visualization          |
| **Compare hyperparameters**       | DVC + MLflow   | DVC for pipeline, MLflow for UI           |
| **Store large data files**        | DVC → S3       | GitHub can't handle large files           |
| **Version control models**        | DVC → S3       | Track model versions without bloating Git |
| **Reproduce experiments**         | DVC            | Exact data, code, and config versions     |
| **Model registry**                | MLflow         | Production model management               |
| **CI/CD**                         | GitHub Actions | Automated testing, deployment             |

## 🚀 Quick Reference

### **Daily Training**

```bash
python main.py              # Train with MLflow tracking
mlflow ui                   # View results
```

### **Experimenting**

```bash
# Edit params.yaml
python main.py
dvc add artifacts/training/model.pth scores.json
dvc metrics diff            # Compare
dvc push && git push        # Save
```

### **Sharing/Cloning**

```bash
git clone <repo>
dvc pull                    # Download large files from S3
```

## ✅ Summary

**Your Understanding:**

- ✅ `main.py` → MLflow tracking (metrics, hyperparameters)
- ✅ DVC → Compare hyperparameters
- ✅ DVC → Push heavy data/models to S3
- ✅ GitHub → Store code + DVC pointers

**This is EXACTLY the MLOps workflow!** 🎉

The key insight: **GitHub stores code, S3 stores data (via DVC), MLflow tracks experiments.**
