# Git + DVC Commands Guide

## Fix Git Line Ending Warnings (Windows)

If you see warnings about `LF will be replaced by CRLF`, this is normal on Windows. You can configure Git to handle this:

```bash
# Configure Git for Windows (recommended)
git config --global core.autocrlf true

# This will automatically convert line endings when committing/checking out
```

**Note:** The warnings are harmless - Git is just telling you it will normalize line endings. The `.gitattributes` file is configured to handle this properly.

## Complete Workflow: Git + DVC + Experiments

### Step 1: Initialize Git (if not already done)

```bash
# Check if git is initialized
git status

# If not initialized, run:
git init
git add .
git commit -m "Initial commit"
```

### Step 2: Initialize DVC

```bash
# Initialize DVC
dvc init

# Add DVC files to git
git add .dvc/
git add .dvcignore
git commit -m "Initialize DVC"
```

### Step 3: Configure DVC Remote (Optional - for S3/Cloud Storage)

```bash
# For S3 (if you have AWS)
dvc remote add -d myremote s3://your-bucket-name/dvc-storage

# For DagsHub (alternative)
dvc remote add -d origin https://dagshub.com/username/repo.dvc

# Commit DVC config
git add .dvc/config
git commit -m "Configure DVC remote"
```

### Step 4: Run Experiments with DVC

```bash
# Run all experiments (automatically commits to git)
python run_experiments.py
```

**What happens:**
- Script updates `params.yaml`
- Runs `dvc repro`
- Commits to git automatically

### Step 5: Manual Git Commands (if needed)

#### After Each Experiment (if running manually)

```bash
# Stage changed files
git add params.yaml
git add dvc.lock
git add scores.json

# Commit experiment
git commit -m "Experiment: batch_size=16, epochs=3"

# View commit history
git log --oneline
```

#### View Git History

```bash
# View all commits
git log

# View commits on one line
git log --oneline

# View last 5 commits
git log --oneline -5

# View changes in a file
git log -p params.yaml
```

#### Compare Experiments

```bash
# Compare current with previous commit
git diff HEAD~1 params.yaml

# Compare with specific commit
git diff <commit-hash> params.yaml

# View what changed in last commit
git show HEAD
```

#### Checkout Specific Experiment

```bash
# List all commits
git log --oneline

# Checkout specific experiment
git checkout <commit-hash>

# View metrics at that commit
dvc metrics show

# Go back to latest
git checkout main
# or
git checkout master
```

### Step 6: View DVC Metrics

```bash
# Show all metrics (current state)
dvc metrics show

# Show metrics at specific commit
git checkout <commit-hash>
dvc metrics show

# Compare metrics between commits
dvc metrics diff HEAD~1

# Compare specific commits
dvc metrics diff <commit-hash-1> <commit-hash-2>

# Compare with previous experiment
dvc metrics diff HEAD~6 HEAD
```

### Step 7: Push to Remote (if using GitHub)

```bash
# Add remote (first time only)
git remote add origin https://github.com/username/repo.git

# Push code to GitHub
git push -u origin main
# or
git push -u origin master

# Push DVC files to remote storage (if configured)
dvc push
```

### Step 8: Pull from Remote (on another machine)

```bash
# Clone repository
git clone https://github.com/username/repo.git
cd repo

# Pull DVC files from remote storage
dvc pull
```

## Complete Example Workflow

### First Time Setup

```bash
# 1. Initialize Git
git init
git add .
git commit -m "Initial commit"

# 2. Initialize DVC
dvc init
git add .dvc/ .dvcignore
git commit -m "Initialize DVC"

# 3. Configure remote (optional)
dvc remote add -d myremote s3://bucket/dvc-storage
git add .dvc/config
git commit -m "Configure DVC remote"

# 4. Add GitHub remote (optional)
git remote add origin https://github.com/username/repo.git
```

### Running Experiments

```bash
# Option 1: Automated (recommended)
python run_experiments.py
# This automatically commits to git

# Option 2: Manual
# Edit params.yaml
dvc repro
git add params.yaml dvc.lock scores.json
git commit -m "Experiment: batch_size=16"
```

### Comparing Experiments

```bash
# View all metrics
dvc metrics show

# Compare with previous
dvc metrics diff HEAD~1

# Compare all experiments
dvc metrics diff HEAD~6 HEAD

# View in MLflow
mlflow ui
```

## Useful Git Commands

### Basic Commands

```bash
# Check status
git status

# View changes
git diff

# View staged changes
git diff --staged

# Add files
git add <file>
git add .                    # Add all files
git add params.yaml          # Add specific file

# Commit
git commit -m "Message"

# View history
git log
git log --oneline
git log --graph --oneline

# View specific file history
git log params.yaml
```

### Working with Branches

```bash
# Create branch for experiment
git checkout -b experiment-batch-16

# Switch branches
git checkout main
git checkout experiment-batch-16

# List branches
git branch

# Merge branch
git checkout main
git merge experiment-batch-16
```

### Undo Changes

```bash
# Unstage files
git reset HEAD <file>

# Discard changes (careful!)
git checkout -- <file>

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View reflog (find lost commits)
git reflog
```

## DVC + Git File Structure

```
project/
├── .dvc/                    # DVC config (tracked by git)
│   ├── config               # DVC configuration
│   └── cache/               # Local cache (NOT in git)
├── .git/                    # Git repository
├── .gitignore               # Git ignore file
├── .dvcignore               # DVC ignore file
├── params.yaml              # Tracked by git
├── dvc.yaml                 # Tracked by git
├── dvc.lock                 # Tracked by git (auto-generated)
├── scores.json              # Tracked by git (metrics)
├── *.dvc                    # DVC pointer files (tracked by git)
└── artifacts/               # Large files (NOT in git, tracked by DVC)
    ├── data.zip.dvc         # Pointer file (in git)
    └── training/
        └── model.pth.dvc    # Pointer file (in git)
```

## Quick Reference

### Before Running Experiments

```bash
# Check if git is initialized
git status

# Check if DVC is initialized
ls .dvc/

# If not, initialize:
dvc init
git add .dvc/ .dvcignore
git commit -m "Initialize DVC"
```

### After Running Experiments

```bash
# View metrics
dvc metrics show

# Compare experiments
dvc metrics diff

# View git history
git log --oneline

# Compare parameters
git diff HEAD~1 params.yaml
```

### If Something Goes Wrong

```bash
# Check DVC status
dvc status

# Check git status
git status

# View DVC pipeline
dvc dag

# View what changed
dvc diff
```

## Complete Workflow Example

```bash
# 1. Setup (one-time)
git init
dvc init
git add .dvc/ .dvcignore
git commit -m "Initialize DVC"

# 2. Run experiments
python run_experiments.py
# This runs 6 experiments and commits each to git

# 3. Compare results
dvc metrics show              # View all metrics
dvc metrics diff HEAD~6 HEAD  # Compare all experiments
python compare_experiments.py # Get recommendations

# 4. View in MLflow
mlflow ui                     # Visual comparison

# 5. Select best model
git log --oneline             # Find best experiment
git checkout <commit-hash>    # Checkout best experiment
dvc metrics show              # Verify metrics
```

---

**Summary:** 
- Git tracks: code, configs, params.yaml, .dvc files, dvc.lock
- DVC tracks: large files (data, models) via pointer files
- Both work together: Git for code/config, DVC for data/models

