# DVC Experiments - Run Multiple Experiments at Once

## Overview
DVC has an **experiments** feature that allows you to run multiple experiments with different parameter combinations in a single command or queue.

## Method 1: DVC Experiments (Recommended)

### Prerequisites
```bash
# Check if DVC experiments is available (DVC 2.0+)
dvc --version

# If not available, upgrade DVC
pip install --upgrade dvc
```

### Run Multiple Experiments at Once

#### Option A: Queue Multiple Experiments
```bash
# Queue experiment 1: BATCH_SIZE=16, EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3

# Queue experiment 2: BATCH_SIZE=32, EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3

# Queue experiment 3: BATCH_SIZE=8, EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=8 -S params.yaml:EPOCHS=3

# Run all queued experiments
dvc exp run --run-all
```

#### Option B: Run Experiments Sequentially
```bash
# Run experiment 1
dvc exp run -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3

# Run experiment 2
dvc exp run -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3

# Run experiment 3
dvc exp run -S params.yaml:BATCH_SIZE=8 -S params.yaml:EPOCHS=3
```

#### Option C: Using Experiments File
Create `experiments.yaml`:
```yaml
experiments:
  exp_batch_16:
    params.yaml:
      BATCH_SIZE: 16
      EPOCHS: 3
  exp_batch_32:
    params.yaml:
      BATCH_SIZE: 32
      EPOCHS: 3
  exp_batch_8:
    params.yaml:
      BATCH_SIZE: 8
      EPOCHS: 3
  exp_batch_64:
    params.yaml:
      BATCH_SIZE: 64
      EPOCHS: 3
```

Then run:
```bash
dvc exp run experiments.yaml
```

### View Experiment Results
```bash
# List all experiments
dvc exp show

# Compare experiments
dvc metrics diff

# Show metrics for all experiments
dvc exp show --no-pager
```

### Compare Experiments
```bash
# Compare all experiments
dvc exp show --sort-by accuracy

# Compare specific experiments
dvc metrics diff exp_batch_16 exp_batch_32
```

## Method 2: Shell Script (Simple Loop)

### Windows PowerShell Script (`run_experiments.ps1`)
```powershell
# Run multiple experiments with different batch sizes
$batchSizes = @(8, 16, 24, 32, 48, 64)
$epochs = 3

foreach ($batchSize in $batchSizes) {
    Write-Host "Running experiment: BATCH_SIZE=$batchSize, EPOCHS=$epochs"
    
    # Edit params.yaml (using PowerShell)
    (Get-Content params.yaml) -replace 'BATCH_SIZE: \d+', "BATCH_SIZE: $batchSize" | Set-Content params.yaml
    
    # Run DVC
    dvc repro
    
    # Show metrics
    dvc metrics show
    
    Write-Host "Completed: BATCH_SIZE=$batchSize`n"
}
```

Run it:
```powershell
.\run_experiments.ps1
```

### Bash Script (Linux/Mac/Git Bash) (`run_experiments.sh`)
```bash
#!/bin/bash

# Run multiple experiments with different batch sizes
batch_sizes=(8 16 24 32 48 64)
epochs=3

for batch_size in "${batch_sizes[@]}"; do
    echo "Running experiment: BATCH_SIZE=$batch_size, EPOCHS=$epochs"
    
    # Edit params.yaml (using sed)
    sed -i "s/BATCH_SIZE: .*/BATCH_SIZE: $batch_size/" params.yaml
    sed -i "s/EPOCHS: .*/EPOCHS: $epochs/" params.yaml
    
    # Run DVC
    dvc repro
    
    # Show metrics
    dvc metrics show
    
    echo "Completed: BATCH_SIZE=$batch_size"
    echo ""
done
```

Run it:
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## Method 3: One-Liner (PowerShell)

```powershell
# Run 6 experiments with different batch sizes
8,16,24,32,48,64 | ForEach-Object { (Get-Content params.yaml) -replace 'BATCH_SIZE: \d+', "BATCH_SIZE: $_" | Set-Content params.yaml; dvc repro; dvc metrics show }
```

## Comparison: Which Method to Use?

| Method | Pros | Cons |
|--------|------|------|
| **DVC Experiments** | Native DVC feature, better tracking, parallel execution | Requires DVC 2.0+, learning curve |
| **Shell Script** | Simple, works everywhere, easy to customize | Manual git commits, sequential only |
| **One-Liner** | Quick, no files needed | Hard to read, no error handling |

## Recommended Workflow

### For Quick Testing (Shell Script)
```powershell
# Use PowerShell script for quick batch runs
.\run_experiments.ps1
```

### For Production (DVC Experiments)
```bash
# Use DVC experiments for better tracking
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3
dvc exp run --run-all
dvc exp show
```

## View Results

### DVC Metrics
```bash
# View all metrics
dvc metrics show

# Compare with previous
dvc metrics diff
```

### MLflow UI
```bash
# Start MLflow UI to see all experiments
mlflow ui

# Open http://localhost:5000
```

## Example: Complete Workflow

```bash
# 1. Queue multiple experiments
dvc exp run --queue -S params.yaml:BATCH_SIZE=16 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=32 -S params.yaml:EPOCHS=3
dvc exp run --queue -S params.yaml:BATCH_SIZE=8 -S params.yaml:EPOCHS=3

# 2. Run all queued experiments
dvc exp run --run-all

# 3. View results
dvc exp show

# 4. Compare best experiments
dvc metrics diff

# 5. Visual comparison
mlflow ui
```

## Notes

- **DVC Experiments** stores results separately from main branch
- **Shell Script** modifies params.yaml and runs sequentially
- Both methods track metrics in `scores.json`
- MLflow also tracks all runs automatically
- Only commit final selected model to git

---

**Choose the method that best fits your workflow!**

