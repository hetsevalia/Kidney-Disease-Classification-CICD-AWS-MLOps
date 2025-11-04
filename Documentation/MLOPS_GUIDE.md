# MLOps Practices Guide

## Table of Contents
1. [Introduction to MLOps](#introduction-to-mlops)
2. [DVC (Data Version Control)](#dvc-data-version-control)
3. [MLflow Integration](#mlflow-integration)
4. [Pipeline Orchestration](#pipeline-orchestration)
5. [CI/CD with GitHub Actions](#cicd-with-github-actions)
6. [Best Practices](#best-practices)

---

## Introduction to MLOps

### What is MLOps?

MLOps (Machine Learning Operations) is the practice of combining DevOps principles with Machine Learning to streamline the ML lifecycle from development to deployment.

### Why MLOps for This Project?

**Without MLOps:**
- ❌ Hard to track experiments
- ❌ Difficult to reproduce results
- ❌ No version control for data/models
- ❌ Manual deployment process
- ❌ Limited collaboration

**With MLOps:**
- ✅ Track all experiments automatically
- ✅ Reproducible results
- ✅ Version control for data and models
- ✅ Automated deployment
- ✅ Better team collaboration

### MLOps Tools in This Project

1. **DVC** - Data and model version control
2. **MLflow** - Experiment tracking and model registry
3. **Docker** - Containerization for consistent environments
4. **GitHub Actions** - Automated CI/CD pipeline
5. **Flask** - API deployment

---

## DVC (Data Version Control)

### What is DVC?

DVC is like Git for data science. It helps you version control large files, datasets, and models that don't fit in Git.

### How DVC Works

```
┌─────────────────────────────────────────┐
│           Git Repository                │
│                                         │
│  src/         # Python code             │
│  config/      # Configuration files     │
│  .dvc/        # DVC metadata            │
│                                         │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│     DVC Storage (Remote)                │
│                                         │
│  - Large datasets (data.zip)            │
│  - Model files (model.pth)             │
│  - All artifacts                       │
│                                         │
│  Only stores diffs (efficient)          │
└─────────────────────────────────────────┘
```

### DVC Concepts

#### 1. Stages (`dvc.yaml`)

**What is a stage?**
A stage is a single step in your ML pipeline (e.g., data ingestion, training).

**Example Stage**:
```yaml
data_ingestion:
  cmd: python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
  deps:
    - src/cnnClassifier/pipeline/stage_01_data_ingestion.py
    - config/config.yaml
  outs:
    - artifacts/data_ingestion/kidney-ct-scan-image
```

**Breaking it down**:
- `cmd`: Command to run
- `deps`: Dependencies (triggers re-run when changed)
- `outs`: Output files (tracked by DVC)
- `params`: Hyperparameters (tracked for changes)

#### 2. Dependencies (deps)

**What happens?**
DVC watches these files. If they change, the stage re-runs automatically.

**Example**:
```yaml
deps:
  - src/cnnClassifier/pipeline/stage_01_data_ingestion.py  # If code changes
  - config/config.yaml                                    # If config changes
```

#### 3. Outputs (outs)

**What happens?**
DVC tracks these files. If outputs don't exist or are outdated, the stage re-runs.

**Example**:
```yaml
outs:
  - artifacts/data_ingestion/kidney-ct-scan-image  # Track this directory
```

#### 4. Parameters (params)

**What happens?**
DVC tracks parameter changes. If params change, dependent stages re-run.

**Example**:
```yaml
params:
  - IMAGE_SIZE    # If this changes in params.yaml
  - BATCH_SIZE    # If this changes
  - EPOCHS        # Training re-runs if this changes
```

### DVC Commands Explained

#### Initialize DVC
```bash
dvc init
```
**What it does**: Creates `.dvc/` directory with configuration files.

**When to use**: Once at the beginning of the project.

---

#### Run Pipeline
```bash
dvc repro
```
**What it does**: Runs the entire pipeline, respecting dependencies.

**How it works**:
1. Check if code/config changed → Run stage 1
2. Check if stage 1 outputs exist → Skip or run
3. Check if params changed → Decide to re-run subsequent stages
4. Run only what's necessary

**Example**:
```bash
# First time: runs all stages
$ dvc repro
> Running 'data_ingestion'...
> Running 'prepare_base_model'...
> Running 'training'...
> Running 'evaluation'...

# Second time: nothing runs (nothing changed)
$ dvc repro
Data pipelines are up to date.

# After changing BATCH_SIZE in params.yaml
$ dvc repro
> Running 'training'...      # Only training re-runs
> Running 'evaluation'...
```

---

#### Visualize Pipeline
```bash
dvc dag
```
**What it shows**: Dependency graph of pipeline stages

**Output**:
```
+------------------+
| data_ingestion    |
+--------+---------+
         |
         v
+------------------+
| prepare_base     |
+--------+---------+
         |
         v
+------------------+
| training         |
+--------+---------+
         |
         v
+------------------+
| evaluation       |
+------------------+
```

---

#### Check Status
```bash
dvc status
```
**What it shows**: Which stages are up-to-date and which need to run

**Output**:
```
data_ingestion:
    changed deps:
        config/config.yaml
prepare_base_model:
    changed deps:
        config/config.yaml
```

---

#### Track File
```bash
dvc add artifacts/training/model.pth
```
**What it does**: Start tracking this file with DVC

**Why use it**: For individual files not in dvc.yaml

---

### DVC Workflow in This Project

#### 1. Initial Setup
```bash
# Initialize DVC
dvc init

# Define pipeline
# Edit dvc.yaml to define stages

# Commit to Git
git add dvc.yaml .dvc
git commit -m "Add DVC pipeline"
```

#### 2. Running Experiments
```bash
# Modify params.yaml (e.g., change BATCH_SIZE)
# Edit params.yaml

# Run pipeline
dvc repro

# Check status
dvc status

# View results
ls artifacts/
```

#### 3. Comparing Experiments
```bash
# Check different commits
dvc diff HEAD~1  # Compare with previous commit

# See what changed
dvc params diff HEAD~1
```

---

## MLflow Integration

### What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle:
- **Tracking**: Log parameters, metrics, and artifacts
- **Registry**: Centralized model storage
- **Models**: Package models for deployment

### MLflow Components Used

#### 1. Tracking

**Purpose**: Log experiments and compare results

**What Gets Logged**:
- Hyperparameters (epochs, batch size, learning rate)
- Metrics (accuracy, loss)
- Model artifacts (model.pth)
- Git commit hash
- Environment info

**Code Example**:
```python
import mlflow

# Start an experiment run
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'augmentation': True
    })
    
    # Log metrics
    mlflow.log_metrics({
        'train_accuracy': 0.92,
        'val_accuracy': 0.88,
        'train_loss': 0.15,
        'val_loss': 0.22
    })
    
    # Log artifacts (model, plots, etc.)
    mlflow.log_artifact('model.pth')
    mlflow.log_artifact('confusion_matrix.png')
```

#### 2. UI

**Start MLflow UI**:
```bash
mlflow ui
# Visit http://localhost:5000
```

**What You Can Do**:
- View all experiments
- Compare metrics
- Download models
- Visualize metrics over time

---

### MLflow Integration in This Project

#### Setup (Optional - Remote Tracking)

**Using Dagshub** (Cloud MLflow):
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_token
```

**Benefits**:
- Access from anywhere
- Team collaboration
- Backup and security

---

#### Local Setup

**Default location**: `./mlruns/`

**Start UI**:
```bash
mlflow ui
# Open http://localhost:5000
```

---

### MLflow in Evaluation Stage

**File**: `src/cnnClassifier/components/model_evaluation_mlflow.py`

**What it does**:
```python
def log_into_mlflow(self):
    mlflow.set_registry_uri(self.config.mlflow_uri)
    experiment = mlflow.create_experiment("Kidney Disease Classification")
    
    with mlflow.start_run(experiment_id=experiment):
        # Log all parameters
        mlflow.log_params(self.all_params)
        
        # Log metrics
        mlflow.log_metrics({
            'test_accuracy': accuracy
        })
        
        # Log model
        mlflow.pytorch.log_model(model, 'model')
```

---

## Pipeline Orchestration

### Pipeline Flow

```
┌─────────────────────────────────────────┐
│         main.py (Orchestrator)          │
└─────────────────────────────────────────┘
         │
         ├──► Stage 1: Data Ingestion
         │    ├─ Download data
         │    └─ Extract data
         │
         ├──► Stage 2: Prepare Base Model
         │    ├─ Load VGG16
         │    ├─ Modify classifier
         │    └─ Save model
         │
         ├──► Stage 3: Training
         │    ├─ Load data
         │    ├─ Data augmentation
         │    ├─ Train model
         │    └─ Save trained model
         │
         └──► Stage 4: Evaluation
              ├─ Load trained model
              ├─ Evaluate on test set
              ├─ Log to MLflow
              └─ Save metrics
```

### Orchestration with DVC

**How it works**:
```yaml
# dvc.yaml defines the pipeline
stages:
  data_ingestion:
    cmd: python stage_01.py
    deps: [config.yaml, code]
    outs: [data/]
  
  prepare_base_model:
    cmd: python stage_02.py
    deps: [code]
    deps: [params.yaml]  # Re-run if params change
    outs: [base_model.pth]
  
  training:
    cmd: python stage_03.py
    deps: [data/, base_model.pth, code, params.yaml]
    outs: [model.pth]
```

**Benefits**:
- Automatic dependency resolution
- Only re-runs what's necessary
- Reproducible results
- Clear pipeline visualization

---

## CI/CD with GitHub Actions

### What is CI/CD?

**CI (Continuous Integration)**: Automatically test code on every commit

**CD (Continuous Deployment)**: Automatically deploy to production

### CI/CD Pipeline for This Project

```
┌─────────────────────────────────────────┐
│      Developer pushes to GitHub        │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      GitHub Actions Triggered           │
│   (.github/workflows/main.yml)          │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Build Docker Image                 │
│   docker build -t model:latest          │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Push to ECR                        │
│   aws ecr push model:latest             │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Deploy to EC2                      │
│   pull image and run container          │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Application Running                │
│   http://ec2-ip:8080                    │
└─────────────────────────────────────────┘
```

### GitHub Actions Workflow

**File**: `.github/workflows/main.yml`

**Steps**:
1. Checkout code
2. Set up Python environment
3. Build Docker image
4. Login to AWS ECR
5. Push image to ECR
6. Deploy to EC2

**Benefits**:
- Automated deployment
- Consistency across environments
- Quick rollback if needed
- Team collaboration

---

## Best Practices

### 1. Version Control

**What to version control**:
- ✅ Code (Python files)
- ✅ Configuration files (config.yaml, params.yaml)
- ✅ Documentation
- ❌ Large files (use DVC)

**Git + DVC**:
```bash
# Small files → Git
git add src/ config/ params.yaml

# Large files → DVC
dvc add artifacts/training/model.pth
dvc push  # Push to remote storage
```

---

### 2. Experiment Tracking

**Always log**:
- Hyperparameters
- Metrics
- Model artifacts
- Code version (Git hash)
- Environment details

**Why**:
- Compare experiments
- Reproduce results
- Debug issues
- Make data-driven decisions

---

### 3. Configuration Management

**Separate concerns**:
- `config.yaml`: Paths and directories
- `params.yaml`: Hyperparameters
- `requirements.txt`: Dependencies

**Benefits**:
- Easy to modify without code changes
- Clear documentation
- Version control for experiments

---

### 4. Reproducibility

**Ensure**:
- Fixed random seeds
- Version controlled code
- Tracked parameters
- Locked dependencies
- Documented environment

**Code**:
```python
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

### 5. Testing

**Test**:
- Individual components
- Data pipelines
- Model training
- API endpoints
- Deployment

**Before deploying**:
- Run full pipeline locally
- Test API endpoints
- Validate model performance
- Check logs for errors

---

### 6. Monitoring

**Monitor**:
- API response times
- Error rates
- Model performance
- Resource usage
- User feedback

**Tools**:
- MLflow UI (experiments)
- Application logs
- Server metrics
- Custom dashboards

---

### 7. Documentation

**Document**:
- Project setup
- Configuration changes
- Experiment results
- Known issues
- Deployment process

**Keep updated**:
- README.md
- Code comments
- Configuration files
- Experiment logs

---

## Summary

### MLOps in This Project

1. **DVC**: Version control for data and models
2. **MLflow**: Track experiments and metrics
3. **Docker**: Consistent environments
4. **GitHub Actions**: Automated deployment
5. **Flask**: Production-ready API

### Benefits

- ✅ Reproducible experiments
- ✅ Track model performance
- ✅ Automated deployment
- ✅ Version control for data/models
- ✅ Team collaboration
- ✅ Production-ready

### Next Steps

1. Set up MLflow tracking
2. Configure DVC remote storage
3. Set up CI/CD pipeline
4. Implement monitoring
5. Document your experiments

---

For more details, see:
- [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md)
- [TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md)
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)

