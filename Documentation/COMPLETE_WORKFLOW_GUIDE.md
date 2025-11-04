# Complete Workflow Guide - From Scratch to Deployment

## 🎯 Overview
This guide walks you through the complete workflow:
1. Clean GitHub repository (start fresh)
2. Set up DVC properly
3. Run multiple experiments using DVC
4. Select final model
5. Integrate model with website
6. Prepare for deployment

---

## Step 1: Clean GitHub Repository (Start Fresh)

### Option A: Keep Code, Remove Only Commits (Recommended)

```bash
# 1. Backup your current work (optional but recommended)
# Create a backup folder
mkdir ../backup_kidney_project
xcopy /E /I * ../backup_kidney_project

# 2. Remove git history but keep files
Remove-Item -Recurse -Force .git

# 3. Initialize fresh git repository
git init

# 4. Create/verify .gitignore (already exists)
# Make sure it has:
# - .dvc/cache
# - mlruns/
# - artifacts/
# - *.dvc (except data.zip.dvc)
# - dvc.lock
# - scores_*.json

# 5. Stage all files
git add .

# 6. Create initial commit
git commit -m "Initial commit: Clean repository setup"

# 7. Add remote (if you have GitHub repo)
git remote add origin <your-github-repo-url>
# OR if remote exists, update it
git remote set-url origin <your-github-repo-url>

# 8. Push to GitHub (force push since we're starting fresh)
git push -u origin main --force
```

### Option B: Create New Repository

```bash
# 1. Create new repo on GitHub (via web interface)
# 2. Clone it
cd ..
git clone <new-repo-url> kidney-project-clean
cd kidney-project-clean

# 3. Copy all files from old project
xcopy /E /I /Y "..\Kidney Disease Classification MLOps Project\*" .

# 4. Remove unnecessary files
# (Keep only code, configs, docs - not artifacts, models, etc.)
# They're already in .gitignore

# 5. Initial commit
git add .
git commit -m "Initial commit: Clean repository setup"
git push -u origin main
```

---

## Step 2: Set Up DVC Properly

```bash
# 1. Initialize DVC (if not already done)
dvc init

# 2. Configure DVC remote storage (S3 or local)
# Option A: S3 (for production)
dvc remote add -d myremote s3://your-bucket-name/kidney-disease-mlops
dvc remote modify myremote region us-east-1

# Option B: Local storage (for testing)
# Skip remote for now, can add later

# 3. Commit DVC config
git add .dvc .dvcignore
git commit -m "Initialize DVC"
git push

# 4. Verify DVC setup
dvc doctor
```

---

## Step 3: Set Up MLflow (Optional - Remote Tracking)

```bash
# Option A: Local MLflow (default - already set up)
# MLflow will create mlruns/ directory locally
# No setup needed - it works automatically

# Option B: Remote MLflow (if you have MLflow server)
# Edit src/cnnClassifier/config/configuration.py
# Set MLFLOW_TRACKING_URI to your MLflow server URL
```

---

## Step 4: Run Multiple Experiments Using DVC Experiments

### Prerequisites Check

```bash
# Check DVC version (need 2.0+ for experiments)
dvc --version

# If version < 2.0, upgrade:
pip install --upgrade dvc
```

### Run Experiments

```bash
# Method 1: Queue and Run Multiple Experiments

# Experiment 1: Batch Size 16
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3

# Experiment 2: Batch Size 32
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3

# Experiment 3: Batch Size 8
dvc exp run --queue -S params.yaml:BATCH_SIZE=8 -S params.yaml:EPOCHS=3

# Experiment 4: Batch Size 64
dvc exp run --queue -S params.yaml:BATCH_SIZE=64 -S params.yaml:EPOCHS=3

# Experiment 5: Batch Size 24
dvc exp run --queue -S params.yaml:BATCH_SIZE=24 -S params.yaml:EPOCHS=3

# Experiment 6: Batch Size 48
dvc exp run --queue -S params.yaml:BATCH_SIZE=48 -S params.yaml:EPOCHS=3

# If getting 100% accuracy, try EPOCHS=0,1,2
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=0
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=1
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=2

# Run all queued experiments
dvc exp run --run-all
```

### Monitor Experiments

```bash
# View running experiments
dvc exp show

# View all experiments with metrics
dvc exp show --no-pager

# View specific experiment
dvc exp show <experiment-name>
```

---

## Step 5: Compare Experiments and Select Final Model

### View Results

```bash
# Method 1: DVC Metrics
dvc metrics show
dvc metrics diff

# Method 2: DVC Experiments Table
dvc exp show --sort-by accuracy

# Method 3: MLflow UI (Visual)
mlflow ui
# Open http://localhost:5000
# Compare all runs visually
```

### Select Best Model

**Criteria for selecting best model:**
- ✅ Accuracy: ~97% (NOT 100% - that's overfitting)
- ✅ F1-score: > 0.95
- ✅ Train/Val gap: < 0.05 (small gap = good generalization)
- ✅ Loss: Low and stable

### Apply Best Experiment to Main Branch

```bash
# View all experiments
dvc exp show

# Apply best experiment (replace <exp-name> with actual name)
dvc exp apply <exp-name>

# OR manually update params.yaml with best hyperparameters
# Then run once more to get final model
dvc repro

# Verify final model
dvc metrics show
```

---

## Step 6: Commit Final Model to Git

```bash
# 1. Ensure params.yaml has best hyperparameters
# 2. Ensure scores.json has best model metrics

# 3. Stage final model files
git add params.yaml
git add scores.json
git add artifacts/training/model.pth
git add artifacts/training/best_model.pth

# 4. Commit final model
git commit -m "Final model: batch_size=32, epochs=3, accuracy=97%"

# 5. Push to GitHub
git push
```

---

## Step 7: Integrate Model with Website

### Verify Model Path in App

```bash
# Check app.py to ensure it loads the correct model
# Should point to: artifacts/training/model.pth or best_model.pth
```

### Test Website Locally

```bash
# 1. Ensure model exists
python check_model.py

# 2. Run Flask app
python app.py

# 3. Test in browser
# Open http://localhost:5000
# Upload an image and test classification
```

### Update Model Path if Needed

If your app.py needs to be updated to use the final model:

```python
# In app.py, ensure model path is:
MODEL_PATH = "artifacts/training/model.pth"
# OR
MODEL_PATH = "artifacts/training/best_model.pth"
```

---

## Step 8: Prepare for Deployment

### Option A: Docker Deployment

```bash
# 1. Build Docker image
docker build -t kidney-disease-classifier .

# 2. Test locally
docker run -p 5000:5000 kidney-disease-classifier

# 3. Push to Docker Hub (if needed)
docker tag kidney-disease-classifier yourusername/kidney-disease-classifier
docker push yourusername/kidney-disease-classifier
```

### Option B: AWS Deployment (EC2/ECS/Lambda)

```bash
# 1. Ensure requirements.txt is up to date
pip freeze > requirements.txt

# 2. Test deployment script (if exists)
# Run deployment script

# 3. Set up CI/CD with GitHub Actions (if needed)
# See .github/workflows/ for CI/CD configs
```

### Option C: Local/VM Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run app
python app.py

# 3. Use gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Step 9: Verify MLflow Integration

### Check MLflow Tracking

```bash
# 1. Start MLflow UI
mlflow ui

# 2. Verify all experiments are tracked
# Open http://localhost:5000
# You should see all runs with:
# - Hyperparameters
# - Metrics (accuracy, loss, f1, etc.)
# - Model artifacts
# - Confusion matrix plots

# 3. Register best model in MLflow (optional)
mlflow models serve -m runs:/<run-id>/model
```

### Verify MLflow Logging in Code

All experiments should be automatically logged to MLflow because:
- `model_training.py` logs to MLflow
- `model_evaluation_mlflow.py` logs to MLflow
- Metrics are stored in `mlruns/` directory

---

## Step 10: Final Checklist

### Before Deployment

- [ ] ✅ Clean GitHub repository (no unnecessary files)
- [ ] ✅ DVC initialized and configured
- [ ] ✅ Multiple experiments run and compared
- [ ] ✅ Final model selected (not overfitted)
- [ ] ✅ `params.yaml` has best hyperparameters
- [ ] ✅ `scores.json` has best model metrics
- [ ] ✅ Final model committed to git
- [ ] ✅ Website tested locally with final model
- [ ] ✅ MLflow tracking working (all experiments visible)
- [ ] ✅ Requirements.txt updated
- [ ] ✅ Dockerfile tested (if using Docker)
- [ ] ✅ Deployment scripts ready

---

## Quick Reference Commands

### DVC Experiments
```bash
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
```bash
# View metrics
dvc metrics show

# Compare
dvc metrics diff
```

### MLflow
```bash
# Start UI
mlflow ui

# View runs
mlflow runs list
```

### Git
```bash
# Status
git status

# Commit
git add .
git commit -m "message"

# Push
git push
```

---

## Troubleshooting

### DVC Experiments Not Working
```bash
# Upgrade DVC
pip install --upgrade dvc

# Check version
dvc --version  # Should be 2.0+
```

### MLflow Not Tracking
```bash
# Check MLflow URI in config
# Should be in src/cnnClassifier/config/configuration.py

# Check mlruns/ directory exists
ls mlruns/

# Start MLflow UI to verify
mlflow ui
```

### Model Not Loading in App
```bash
# Check model path in app.py
# Verify model file exists
ls artifacts/training/model.pth

# Test model loading
python check_model.py
```

---

## Summary

**Complete Workflow:**
1. ✅ Clean GitHub repo (fresh start)
2. ✅ Set up DVC
3. ✅ Run multiple experiments: `dvc exp run --queue ...`
4. ✅ Compare: `dvc exp show` or `mlflow ui`
5. ✅ Select best: `dvc exp apply <exp-name>`
6. ✅ Commit final model to git
7. ✅ Test website with final model
8. ✅ Deploy

**Remember:**
- Only commit final model to git (not all experiments)
- Use MLflow UI for visual comparison
- Target ~97% accuracy (not 100%)
- Test website locally before deployment

---

**You're all set! Follow these steps sequentially and you'll have a clean, production-ready MLOps pipeline!** 🚀

