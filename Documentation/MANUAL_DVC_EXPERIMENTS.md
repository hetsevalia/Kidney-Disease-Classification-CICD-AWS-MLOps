# Manual DVC Experiments Guide

## Overview
This guide shows how to run experiments manually using DVC commands in terminal. No Python scripts - just pure DVC workflow.

**For running multiple experiments at once, see `Documentation/DVC_EXPERIMENTS_GUIDE.md`**

## Workflow

### Step 1: Run an Experiment

```bash
# 1. Edit params.yaml with your hyperparameters
# For example, change BATCH_SIZE to 16

# 2. Run the pipeline with DVC
dvc repro

# This will:
# - Run all stages (data ingestion, prepare base model, training, evaluation)
# - Track metrics in scores.json
# - Update dvc.lock file
```

### Step 2: View Current Metrics

```bash
# View current metrics
dvc metrics show

# Output shows:
# Path         accuracy    f1_score    loss     precision    recall
# scores.json  0.95        0.94        0.15     0.95         0.94
```

### Step 3: Run Another Experiment

```bash
# 1. Edit params.yaml (change BATCH_SIZE to 32, etc.)

# 2. Run again
dvc repro

# 3. View metrics again
dvc metrics show
```

### Step 4: Compare Experiments

#### Option A: Compare with Previous Run
```bash
# Compare current with previous (HEAD)
dvc metrics diff

# Compare with specific commit
dvc metrics diff HEAD~1

# Compare with 2 commits ago
dvc metrics diff HEAD~2
```

#### Option B: Compare Specific Commits
```bash
# First, see commit history
git log --oneline

# Compare two specific commits
dvc metrics diff <commit-hash-1> <commit-hash-2>
```

#### Option C: View All Metrics Across Commits
```bash
# Show metrics for all commits
dvc metrics show --all-branches

# Show metrics for specific commits
dvc metrics show HEAD HEAD~1 HEAD~2
```

### Step 5: View in MLflow

```bash
# Start MLflow UI
mlflow ui

# Open http://localhost:5000
# Compare all runs visually
```

## Complete Example Workflow

### Experiment 1: Batch Size 16
```bash
# Edit params.yaml: BATCH_SIZE: 16, EPOCHS: 3
dvc repro
dvc metrics show
```

### Experiment 2: Batch Size 32
```bash
# Edit params.yaml: BATCH_SIZE: 32, EPOCHS: 3
dvc repro
dvc metrics diff HEAD~1  # Compare with previous
```

### Experiment 3: Batch Size 8
```bash
# Edit params.yaml: BATCH_SIZE: 8, EPOCHS: 3
dvc repro
dvc metrics diff HEAD~1  # Compare with previous
```

### Compare All Experiments
```bash
# View all metrics
dvc metrics show HEAD HEAD~1 HEAD~2

# Or view in MLflow
mlflow ui
```

## Git Workflow (Optional - Only for Final Model)

**Important:** Only commit to git when you've selected the final model for deployment.

```bash
# After selecting best model:

# 1. Ensure params.yaml has best hyperparameters
# 2. Ensure scores.json has best model's metrics

# 3. Commit only final model
git add params.yaml scores.json
git commit -m "Final model: batch_size=32, epochs=3, accuracy=97%"

# 4. Push to GitHub
git push
```

## DVC Commands Reference

### Basic Commands
```bash
dvc repro              # Run pipeline
dvc metrics show       # Show current metrics
dvc metrics diff       # Compare with previous
dvc status             # Check pipeline status
dvc dag                # View pipeline graph
```

### Metrics Commands
```bash
# View current metrics
dvc metrics show

# Compare with previous commit
dvc metrics diff

# Compare specific commits
dvc metrics diff HEAD~1 HEAD~2

# View all metrics across commits
dvc metrics show --all-branches

# Show specific commits
dvc metrics show HEAD HEAD~1 HEAD~2 HEAD~3
```

### Pipeline Commands
```bash
# Check what needs to run
dvc status

# Visualize pipeline
dvc dag

# Run specific stage
dvc repro training
dvc repro evaluation

# Force rerun everything
dvc repro --force
```

## Tips

1. **After each experiment**, check `dvc metrics show` to see current metrics
2. **Compare experiments** using `dvc metrics diff HEAD~1` after each run
3. **Use MLflow UI** for visual comparison: `mlflow ui`
4. **Only commit to git** when you've selected the final model
5. **Keep experiment files local** - they're in `.gitignore`

## Example Session

```bash
# Experiment 1
# Edit params.yaml: BATCH_SIZE: 16, EPOCHS: 3
dvc repro
dvc metrics show
# Output: accuracy: 0.96, f1_score: 0.95

# Experiment 2
# Edit params.yaml: BATCH_SIZE: 32, EPOCHS: 3
dvc repro
dvc metrics diff HEAD~1
# Shows: accuracy improved from 0.96 to 0.97

# Experiment 3
# Edit params.yaml: BATCH_SIZE: 8, EPOCHS: 3
dvc repro
dvc metrics diff HEAD~1
# Shows comparison

# Compare all three
dvc metrics show HEAD HEAD~1 HEAD~2
# Or
mlflow ui  # Visual comparison
```

## Summary

**No scripts needed!** Just:
1. Edit `params.yaml`
2. Run `dvc repro`
3. Check `dvc metrics show` or `dvc metrics diff`
4. Repeat
5. Compare in MLflow UI
6. Commit only final model to git

---

## 🚀 Running Multiple Experiments at Once

**Want to run multiple experiments automatically?** See `Documentation/DVC_EXPERIMENTS_GUIDE.md` for:
- DVC Experiments feature (native DVC)
- Shell scripts (PowerShell/Bash)
- One-liner commands

**Quick example:**
```powershell
# PowerShell script (Windows)
.\run_experiments.ps1
```

```bash
# Bash script (Linux/Mac/Git Bash)
chmod +x run_experiments.sh
./run_experiments.sh
```

---

**That's it! Pure DVC workflow using terminal commands.**

