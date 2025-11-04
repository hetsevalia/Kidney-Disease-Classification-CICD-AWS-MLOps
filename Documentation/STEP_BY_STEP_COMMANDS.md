# Step-by-Step Commands - Complete Workflow

## 🎯 Goal

1. Clean GitHub repository (fresh start)
2. Run multiple experiments using DVC
3. Select final model
4. Test website with final model
5. Prepare for deployment (with MLflow tracking)

---

## 📋 STEP 1: Clean GitHub Repository

### Option A: Keep Code, Remove Git History (Recommended)

```powershell
# 1. Backup current work (optional but recommended)
cd ..
mkdir backup_kidney_project
xcopy /E /I "Kidney Disease Classification MLOps Project\*" "backup_kidney_project\"
cd "Kidney Disease Classification MLOps Project"

# 2. Remove .git folder (removes all commit history)
Remove-Item -Recurse -Force .git

# 3. Initialize fresh git repository
git init

# 4. Create .gitignore (already exists, verify it has these):
# - .dvc/cache
# - mlruns/
# - artifacts/
# - *.dvc (except data.zip.dvc)
# - dvc.lock
# - scores_*.json

# 5. Stage all files
git add .

# 6. Create initial commit
git commit -m "Initial commit: Clean MLOps pipeline setup"

# 7. Add/Update remote repository
# If you have a new GitHub repo:
git remote add origin https://github.com/yourusername/kidney-disease-mlops.git

# OR if you want to update existing remote:
git remote set-url origin https://github.com/yourusername/kidney-disease-mlops.git

# 8. Create main branch (if needed)
git branch -M main

# 9. Push to GitHub (force push since we're starting fresh)
git push -u origin main --force
```

### Option B: Create New Repository on GitHub

```powershell
# 1. Create new repo on GitHub via web interface
# 2. Then clone it:
cd ..
git clone https://github.com/yourusername/kidney-disease-mlops-clean.git
cd kidney-disease-mlops-clean

# 3. Copy all files from old project (excluding .git)
xcopy /E /I /Y "..\Kidney Disease Classification MLOps Project\*" .

# 4. Remove .git folder if copied
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue

# 5. Re-initialize git
git init
git add .
git commit -m "Initial commit: Clean MLOps pipeline setup"
git branch -M main
git push -u origin main
```

---

## 📋 STEP 2: Initialize DVC

```powershell
# 1. Check if DVC is installed
dvc --version

# 2. If not installed:
pip install dvc

# 3. Initialize DVC
dvc init

# 4. Verify .dvcignore exists (should already exist)
# Check if it has:
# - artifacts/
# - mlruns/

# 5. Commit DVC initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"
git push

# 6. Verify DVC setup
dvc doctor
```

---

## 📋 STEP 3: Verify MLflow Setup

```powershell
# 1. Check if MLflow is installed
pip show mlflow

# 2. If not installed:
pip install mlflow==2.2.2

# 3. Verify MLflow tracking URI (should be local by default)
# Check: src/cnnClassifier/config/configuration.py
# Should have: MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# 4. MLflow will automatically create mlruns/ directory when first run starts
# No additional setup needed - it works automatically!
```

---

## 📋 STEP 4: Run Multiple Experiments Using DVC

### Check DVC Version First

```powershell
# Check DVC version (need 2.0+ for experiments feature)
dvc --version

# If version < 2.0, upgrade:
pip install --upgrade dvc
```

### Queue and Run Experiments

```powershell
# Queue Experiment 1: Batch Size 16, Epochs 3
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3

# Queue Experiment 2: Batch Size 32, Epochs 3
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3

# Queue Experiment 3: Batch Size 8, Epochs 3
dvc exp run --queue -S params.yaml:BATCH_SIZE=8 -S params.yaml:EPOCHS=3

# Queue Experiment 4: Batch Size 64, Epochs 3
dvc exp run --queue -S params.yaml:BATCH_SIZE=64 -S params.yaml:EPOCHS=3

# Queue Experiment 5: Batch Size 24, Epochs 3
dvc exp run --queue -S params.yaml:BATCH_SIZE=24 -S params.yaml:EPOCHS=3

# Queue Experiment 6: Batch Size 48, Epochs 3
dvc exp run --queue -S params.yaml:BATCH_SIZE=48 -S params.yaml:EPOCHS=3

# If you get 100% accuracy, try lower epochs:
# Queue Experiment 7: Batch Size 16, Epochs 0
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=0

# Queue Experiment 8: Batch Size 32, Epochs 1
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=1

# Queue Experiment 9: Batch Size 16, Epochs 2
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=2

# Run ALL queued experiments
dvc exp run --run-all
```

**Note:** This will take time! Each experiment runs the full pipeline (data ingestion, training, evaluation).

### Monitor Experiments

```powershell
# While experiments are running, check status:
dvc exp show

# View all completed experiments:
dvc exp show --no-pager
```

---

## 📋 STEP 5: Compare Experiments and Select Final Model

### View Results

```powershell
# Method 1: DVC Metrics (after all experiments complete)
dvc metrics show

# Method 2: DVC Experiments Table (sorted by accuracy)
dvc exp show --sort-by accuracy

# Method 3: MLflow UI (Visual comparison - BEST OPTION)
mlflow ui
# Open http://localhost:5000 in browser
# You'll see all experiments with:
# - Hyperparameters
# - Metrics (accuracy, loss, f1, etc.)
# - Model artifacts
# - Confusion matrices
```

### Select Best Model

**Criteria:**

- ✅ Accuracy: ~97% (NOT 100% - that's overfitting)
- ✅ F1-score: > 0.95
- ✅ Train/Val gap: < 0.05 (small gap = good generalization)
- ✅ Loss: Low and stable

### Apply Best Experiment to Main Branch

```powershell
# 1. View all experiments
dvc exp show

# 2. Note the experiment name/ID of the best one
# Example output shows experiment names like: exp-xxxxx

# 3. Apply best experiment (replace <exp-name> with actual name)
dvc exp apply <exp-name>

# OR manually update params.yaml with best hyperparameters:
# Edit params.yaml:
# - BATCH_SIZE: 32 (or your best value)
# - EPOCHS: 3 (or your best value)

# 4. Run once more to get final model
dvc repro

# 5. Verify final model metrics
dvc metrics show
```

---

## 📋 STEP 6: Commit Final Model to Git

```powershell
# 1. Ensure params.yaml has best hyperparameters
# 2. Ensure scores.json has best model metrics
# 3. Ensure artifacts/training/model.pth exists

# 4. Track final model with DVC (recommended for large files)
dvc add artifacts/training/model.pth
dvc add artifacts/training/best_model.pth  # if exists

# This creates model.pth.dvc and best_model.pth.dvc files

# 5. Stage final model files
git add params.yaml
git add scores.json
git add model.pth.dvc
git add best_model.pth.dvc  # if exists
git add .gitignore

# 6. Commit final model
git commit -m "Final model: batch_size=32, epochs=3, accuracy=97%"

# 7. Push to GitHub
git push

# 8. Push model files to DVC remote (if configured)
# If you set up S3 remote:
dvc push
```

---

## 📋 STEP 7: Test Website with Final Model

### Verify Model Exists

```powershell
# 1. Check if model file exists
ls artifacts/training/model.pth

# 2. Test model loading
python check_model.py
```

### Test Website Locally

```powershell
# 1. Start Flask app
python app.py

# 2. Open browser
# Go to: http://localhost:8080

# 3. Test prediction:
# - Upload a kidney CT scan image
# - Click "Predict"
# - Should return: Normal or Tumor classification

# 4. Check model status endpoint:
# Go to: http://localhost:8080/model-status
# Should return: {"status": "ready", "model_path": "artifacts/training/model.pth"}
```

### Verify Prediction Pipeline

The app loads model from: `artifacts/training/model.pth` (already configured in `prediction.py`)

---

## 📋 STEP 8: Verify MLflow Integration

### Check MLflow Tracking

```powershell
# 1. Start MLflow UI
mlflow ui

# 2. Open browser: http://localhost:5000

# 3. Verify you see:
# - All experiment runs
# - Hyperparameters for each run
# - Metrics (accuracy, loss, f1, precision, recall)
# - Model artifacts (model.pth files)
# - Confusion matrix plots
# - Training curves

# 4. Compare experiments visually:
# - Select multiple runs
# - Compare metrics side by side
# - Download best model if needed
```

### Verify MLflow Logging in Code

MLflow is already integrated in:

- `src/cnnClassifier/components/model_training.py` - Logs training metrics
- `src/cnnClassifier/components/model_evaluation_mlflow.py` - Logs evaluation metrics

All experiments are automatically tracked!

---

## 📋 STEP 9: Prepare for Deployment

### Option A: Docker Deployment

```powershell
# 1. Check Dockerfile exists
ls Dockerfile

# 2. Build Docker image
docker build -t kidney-disease-classifier .

# 3. Test Docker image locally
docker run -p 5000:5000 kidney-disease-classifier

# 4. Test in browser: http://localhost:5000

# 5. Push to Docker Hub (if needed)
docker tag kidney-disease-classifier yourusername/kidney-disease-classifier
docker login
docker push yourusername/kidney-disease-classifier
```

### Option B: AWS Deployment

```powershell
# 1. Ensure requirements.txt is up to date
pip freeze > requirements.txt

# 2. Check deployment scripts (if exist)
ls *.sh
ls *.yaml

# 3. Test deployment locally first
python app.py

# 4. Deploy to AWS (follow your deployment guide)
```

### Option C: Local/VM Deployment

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with gunicorn (production server)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app

# 3. Access: http://localhost:8080
```

---

## 📋 STEP 10: Final Checklist

Before deployment, verify:

```powershell
# ✅ Clean GitHub repo (no unnecessary files)
git status

# ✅ DVC initialized
dvc doctor

# ✅ Multiple experiments completed
dvc exp show

# ✅ Final model selected (not overfitted)
dvc metrics show

# ✅ Final model committed
git log --oneline

# ✅ Website tested locally
# (Run: python app.py and test in browser)

# ✅ MLflow tracking working
# (Run: mlflow ui and verify all runs visible)

# ✅ Requirements.txt updated
pip freeze > requirements.txt

# ✅ Dockerfile tested (if using Docker)
docker build -t kidney-disease-classifier .
```

---

## 🚀 Quick Reference Commands

### DVC Experiments

```powershell
# Queue experiment
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3

# Run all queued
dvc exp run --run-all

# View experiments
dvc exp show

# Apply best experiment
dvc exp apply <exp-name>
```

### DVC Metrics

```powershell
# View metrics
dvc metrics show

# Compare
dvc metrics diff
```

### MLflow

```powershell
# Start UI
mlflow ui

# View runs
mlflow runs list
```

### Git

```powershell
# Status
git status

# Commit
git add .
git commit -m "message"

# Push
git push
```

### Website

```powershell
# Run app
python app.py

# Check model
python check_model.py
```

---

## 📝 Summary

**Complete Workflow:**

1. ✅ **Clean GitHub** - Remove history, start fresh
2. ✅ **Initialize DVC** - Set up version control
3. ✅ **Run Experiments** - `dvc exp run --queue ...` (multiple times)
4. ✅ **Run All** - `dvc exp run --run-all`
5. ✅ **Compare** - `mlflow ui` or `dvc exp show`
6. ✅ **Select Best** - Apply best experiment
7. ✅ **Commit Final** - Git commit final model only
8. ✅ **Test Website** - `python app.py` and test
9. ✅ **Verify MLflow** - `mlflow ui` check all runs
10. ✅ **Deploy** - Docker/AWS/Local

---

## 🎯 Important Notes

1. **Don't commit experiments** - Only commit final selected model
2. **MLflow tracks everything** - All runs automatically logged
3. **Target ~97% accuracy** - Not 100% (that's overfitting)
4. **Test locally first** - Always test website before deployment
5. **Keep repo clean** - Unnecessary files in `.gitignore`

---

**Follow these steps sequentially and you'll have a complete MLOps pipeline ready for deployment!** 🚀
