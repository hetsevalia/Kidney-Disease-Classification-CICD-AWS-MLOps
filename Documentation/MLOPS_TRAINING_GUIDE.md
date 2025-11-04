# MLOps Training Guide - Kidney Disease Classification

## 🎯 Overview

This guide demonstrates MLOps best practices for training and selecting the best model with proper overfitting/underfitting detection.

## 📋 Step-by-Step Training Process

### Step 1: Install Package (if not done)
```bash
pip install -e .
```

### Step 2: Configure Hyperparameters

Edit `params.yaml` to experiment with different configurations:

```yaml
EPOCHS: 50                    # Maximum epochs (early stopping will stop earlier if needed)
LEARNING_RATE: 0.001          # Initial learning rate
BATCH_SIZE: 32                # Batch size for training
EARLY_STOPPING_PATIENCE: 10   # Stop if no improvement for 10 epochs
LEARNING_RATE_SCHEDULER: "ReduceLROnPlateau"  # Learning rate scheduling
```

### Step 3: Run Complete Pipeline

#### Option A: Using Main Script (Recommended)
```bash
python main.py
```

This runs all stages:
1. **Data Ingestion** - Downloads and prepares data
2. **Prepare Base Model** - Loads VGG16 with transfer learning
3. **Training** - Trains with early stopping, checkpointing, and MLflow tracking
4. **Evaluation** - Comprehensive metrics and MLflow logging

#### Option B: Using DVC (For Experiment Tracking)
```bash
# Run complete pipeline
dvc repro

# Run specific stage
dvc repro training

# Compare experiments
dvc metrics diff
```

### Step 4: Monitor Training

During training, you'll see:
- ✅ **Per-epoch metrics**: Train/Val loss and accuracy
- ✅ **Early stopping**: Automatic stop if no improvement
- ✅ **Best model checkpointing**: Saves best model based on validation loss
- ✅ **Overfitting detection**: Warns if train loss >> val loss
- ✅ **Underfitting detection**: Warns if val loss >> train loss
- ✅ **Learning rate scheduling**: Automatically reduces LR when stuck

### Step 5: View Results

#### Check Metrics
```bash
# View scores.json
cat scores.json
```

Includes:
- Accuracy, Precision, Recall, F1-Score
- Per-class metrics
- Confusion matrix

#### MLflow Dashboard
```bash
# Start MLflow UI
mlflow ui

# Or if using DagsHub
# Visit your MLflow URI
```

View:
- Training curves (loss/accuracy per epoch)
- Hyperparameters
- Model artifacts
- Confusion matrix
- Compare different runs

### Step 6: Compare Experiments

#### Using DVC
```bash
# Compare metrics between runs
dvc metrics diff

# Show metrics table
dvc metrics show
```

#### Using MLflow
```bash
mlflow ui
# Compare runs in the web UI
```

## 🔍 Model Selection Criteria

### ✅ Good Model (Target: ~97% Accuracy)
- **Validation Accuracy**: 95-97% (NOT 100% - that's overfitting!)
- **Low Validation Loss**: Decreasing trend
- **Close Train/Val Loss**: Gap < 0.05 (good generalization)
- **High F1-Score**: > 0.95 (balanced precision and recall)
- **Stable Training**: No sudden spikes
- **No Overfitting Warnings**: Train loss close to val loss

### ❌ Overfitting (Detected Automatically)
- **Signs**: Train loss < Val loss by > 0.1
- **Solution**: 
  - Increase dropout
  - Add more data augmentation
  - Reduce model complexity
  - Increase early stopping patience

### ❌ Underfitting (Detected Automatically)
- **Signs**: Val loss >> Train loss by > 0.5
- **Solution**:
  - Increase model capacity
  - Train for more epochs
  - Reduce regularization
  - Adjust learning rate

## 📊 Understanding Metrics

### Accuracy
- Overall correctness: `(Correct Predictions / Total) * 100`
- **Good**: > 85%

### Precision
- Of predictions, how many are correct
- **Good**: > 0.85

### Recall
- Of actual positives, how many were found
- **Good**: > 0.85

### F1-Score
- Harmonic mean of precision and recall
- **Good**: > 0.85

### Confusion Matrix
- Shows per-class predictions
- Helps identify which classes are confused

## 🚀 Experimentation Workflow - Reducing Overfitting

### Problem: 100% Accuracy = Overfitting
If you see 100% accuracy, your model is **overfitting**. The goal is to achieve **~97% accuracy** with good generalization.

### Solution: Manual DVC Experiments (Terminal Commands)

**No scripts needed!** Use DVC commands directly in terminal.

#### Step 1: Run First Experiment

```bash
# Edit params.yaml (change BATCH_SIZE, EPOCHS, etc.)
# For example: BATCH_SIZE: 16, EPOCHS: 3

# Run pipeline with DVC
dvc repro

# View metrics
dvc metrics show
```

#### Step 2: Run More Experiments

```bash
# Experiment 2: Edit params.yaml (BATCH_SIZE: 32, EPOCHS: 3)
dvc repro
dvc metrics diff HEAD~1  # Compare with previous

# Experiment 3: Edit params.yaml (BATCH_SIZE: 8, EPOCHS: 3)
dvc repro
dvc metrics diff HEAD~1  # Compare with previous

# Continue for more experiments...
```

#### Step 3: Compare All Experiments

```bash
# Compare with previous
dvc metrics diff

# View all metrics across commits
dvc metrics show HEAD HEAD~1 HEAD~2 HEAD~3

# Or view in MLflow (visual comparison)
mlflow ui
```

#### Step 4: Select Best Model

- Check `dvc metrics show` for all experiments
- Review MLflow dashboard: `mlflow ui`
- Choose model with **~97% accuracy** (not 100%)
- Good F1-score (> 0.95)
- Low train/val gap (< 0.05)

#### Step 5: Commit Final Model (Only When Selected)

```bash
# Only commit the final selected model
git add params.yaml scores.json
git commit -m "Final model: batch_size=32, epochs=3, accuracy=97%"
```

**Note:** Experiment files are NOT committed to git (they're in `.gitignore`). Only commit when you've selected the final model for deployment.

### Recommended Experiment Configurations

Try these combinations to find ~97% accuracy:

1. **BATCH_SIZE: 16, EPOCHS: 3**
2. **BATCH_SIZE: 32, EPOCHS: 3**
3. **BATCH_SIZE: 8, EPOCHS: 3**
4. **BATCH_SIZE: 64, EPOCHS: 3**
5. **BATCH_SIZE: 24, EPOCHS: 3**
6. **BATCH_SIZE: 48, EPOCHS: 3**

Or try **EPOCHS: 0, 1, or 2** if still getting 100% accuracy.

See `Documentation/MANUAL_DVC_EXPERIMENTS.md` for complete manual workflow guide.

## 🎯 Changes Made to Reduce Overfitting

### 1. Train/Val Split: 70/30 (was 80/20)
- More validation data = better generalization check
- Less training data = less overfitting

### 2. More Aggressive Data Augmentation
- Added RandomVerticalFlip
- Increased rotation (45°)
- Added ColorJitter (brightness, contrast)
- More scale variation (0.7-1.3)

### 3. Stricter Early Stopping
- Patience: 5 (was 10)
- Min Delta: 0.002 (was 0.001)
- Max epochs: 30 (was 50)
- LR Scheduler Patience: 3 (was 5)

## 📁 Key Files & Artifacts

```
artifacts/
├── training/
│   ├── model.pth          # Final trained model
│   └── best_model.pth     # Best model (by validation loss)
├── data_ingestion/
│   └── kidney-ct-scan-image/
└── prepare_base_model/
    └── base_model.pth

scores.json                 # Evaluation metrics
logs/running_logs.log       # Training logs
confusion_matrix.png        # Evaluation visualization
```

## 🎓 MLOps Features Demonstrated

### ✅ Experiment Tracking
- **MLflow**: Tracks all hyperparameters, metrics, and models
- **DVC**: Version control for data and pipelines

### ✅ Model Management
- **Checkpointing**: Saves best model automatically
- **Versioning**: Each run creates a new model version
- **Registry**: MLflow model registry for production

### ✅ Reproducibility
- **DVC**: Ensures same data and code versions
- **MLflow**: Tracks exact configuration used
- **Git**: Version control for code

### ✅ Monitoring
- **Early Stopping**: Prevents overfitting
- **Metrics Tracking**: Comprehensive evaluation
- **Visualization**: Training curves, confusion matrix

### ✅ Automation
- **Pipeline**: Automated end-to-end workflow
- **DVC**: Handles dependencies automatically
- **MLflow**: Automatic logging

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in params.yaml
BATCH_SIZE: 16  # Instead of 32
```

### Training Too Slow
- Reduce epochs (but let early stopping handle it)
- Use smaller image size
- Reduce batch size

### Model Not Improving
- Check learning rate (try 0.0001 or 0.01)
- Increase epochs
- Check data quality

## 📝 Next Steps

1. **Run Experiments**: Edit `params.yaml`, then run `dvc repro`
2. **Compare Results**: Use `dvc metrics diff` or `mlflow ui`
3. **Select Best Model**: Choose model with ~97% accuracy, not 100%
4. **Commit Final Model**: Only commit selected model to git (not all experiments)
5. **Deploy**: Use selected model for inference

**See `Documentation/MANUAL_DVC_EXPERIMENTS.md` for detailed manual workflow.**

## 🔗 Useful Commands

### Setup Commands
```bash
# Install package
pip install -e .

# Initialize DVC (first time)
dvc init
git add .dvc/ .dvcignore
git commit -m "Initialize DVC"

# Check setup
python verify_setup.py
```

### Training Commands
```bash
# Run single training (manual)
python main.py

# Run with DVC (recommended for experiments)
dvc repro

# View metrics after training
dvc metrics show
```

### DVC Commands
```bash
dvc repro              # Run pipeline
dvc metrics show       # Show all metrics
dvc metrics diff       # Compare metrics
dvc metrics diff HEAD~6 HEAD  # Compare all experiments
dvc status             # Check pipeline status
dvc dag                # View pipeline graph
```

### Git Commands (for DVC workflow)
```bash
git status             # Check status
git log --oneline      # View commit history
git diff HEAD~1        # Compare with previous
git checkout <commit>  # Checkout specific experiment
```

### Comparison Commands
```bash
# Compare with previous experiment
dvc metrics diff

# Compare specific commits
dvc metrics diff HEAD~1 HEAD~2

# View all metrics
dvc metrics show HEAD HEAD~1 HEAD~2

# Visual comparison
mlflow ui
```

### MLflow Commands
```bash
mlflow ui              # Start MLflow UI (compare all experiments)
mlflow runs list       # List all runs
```

**See `GIT_DVC_COMMANDS.md` for complete Git + DVC workflow guide.**

---

**Happy Training! 🚀**

For questions, check the logs in `logs/running_logs.log` or MLflow dashboard.

