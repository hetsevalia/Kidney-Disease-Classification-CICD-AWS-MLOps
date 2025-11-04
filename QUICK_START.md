# Quick Start - Complete Workflow

## 🚀 Fast Track Commands

### 1. Clean GitHub Repository
```powershell
Remove-Item -Recurse -Force .git
git init
git add .
git commit -m "Initial commit: Clean MLOps pipeline setup"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main --force
```

### 2. Initialize DVC
```powershell
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
git push
```

### 3. Run Multiple Experiments
```powershell
# Queue experiments
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=8 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=64 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=24 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=48 -S params.yaml:EPOCHS=3

# Run all
dvc exp run --run-all
```

### 4. Compare and Select Best Model
```powershell
# View results
mlflow ui
# Open http://localhost:5000

# OR
dvc exp show --sort-by accuracy

# Apply best experiment
dvc exp apply <exp-name>
# OR manually update params.yaml and run:
dvc repro
```

### 5. Commit Final Model
```powershell
# Track model with DVC
dvc add artifacts/training/model.pth

# Commit
git add params.yaml scores.json model.pth.dvc .gitignore
git commit -m "Final model: batch_size=32, epochs=3, accuracy=97%"
git push
```

### 6. Test Website
```powershell
python app.py
# Open http://localhost:8080
```

### 7. Verify MLflow
```powershell
mlflow ui
# Open http://localhost:5000
```

---

## 📚 Detailed Guide

For complete step-by-step instructions, see:
- **`Documentation/STEP_BY_STEP_COMMANDS.md`** - Complete workflow with all commands
- **`Documentation/COMPLETE_WORKFLOW_GUIDE.md`** - Detailed explanations
- **`Documentation/DVC_EXPERIMENTS_GUIDE.md`** - DVC experiments guide

---

**That's it! Follow these commands sequentially.** 🎯

