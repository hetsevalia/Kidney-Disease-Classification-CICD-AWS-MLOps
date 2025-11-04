# Kidney Disease Classification - Complete Documentation

**Comprehensive Guide for Kidney Disease Classification MLOps Project**

---

**Table of Contents**

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Technology Stack](#technology-stack)
4. [Project Architecture](#project-architecture)
5. [Pipeline Stages Explained](#pipeline-stages-explained)
6. [Technical Architecture](#technical-architecture)
7. [MLOps Practices](#mlops-practices)
8. [Setup and Installation](#setup-and-installation)
9. [API Documentation](#api-documentation)
10. [Deployment Guide](#deployment-guide)
11. [Troubleshooting](#troubleshooting)
12. [Quick Reference](#quick-reference)
13. [Best Practices](#best-practices)

---

# Project Overview

## What is This Project?

This project implements an **end-to-end MLOps pipeline** for classifying kidney CT scan images into two categories:
- **Normal**: Healthy kidney tissue
- **Tumor**: Abnormal kidney tissue with tumors

## Key Objectives

1. **Automated Data Pipeline**: Automated data ingestion from Google Drive
2. **Transfer Learning**: Use pre-trained VGG16 model for classification
3. **Model Training**: Train with data augmentation and validation
4. **Model Evaluation**: Evaluate model performance with metrics tracking
5. **Model Deployment**: Deploy via Flask API with web interface
6. **MLOps Integration**: Use DVC for version control and MLflow for experiment tracking

## Dataset Information

- **Source**: Kidney CT scan images from Google Drive
- **Classes**: 2 (Normal, Tumor)
- **Images**: ~465 total images (240 Normal + 225 Tumor)
- **Format**: JPEG images
- **Resolution**: Variable (processed to 224x224)

---

# Quick Start

## Setup Environment

```bash
# 1. Create conda environment
conda create -n cnncls python=3.8 -y
conda activate cnncls

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python main.py

# 4. Start web application
python app.py
```

Visit: http://localhost:8080

---

# Technology Stack

## Deep Learning & ML Framework
- **PyTorch** (≥2.0.0): Deep learning framework for model development
- **Torchvision** (≥0.15.0): Image processing and pretrained models

## Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **PIL (Pillow)**: Image processing
- **gdown**: Download files from Google Drive

## MLOps Tools
- **DVC (Data Version Control)**: Version control for datasets and models
- **MLflow** (2.2.2): Experiment tracking and model registry
- **Dagshub**: MLflow tracking backend

## Web Framework
- **Flask**: Web application framework for API
- **Flask-CORS**: Cross-origin resource sharing support

## Development & Configuration
- **PyYAML**: Configuration file management
- **python-box**: Configuration management
- **Jupyter Notebook**: Research and experimentation

## Cloud & Deployment
- **Docker**: Containerization for deployment
- **AWS**: Cloud infrastructure (EC2, ECR)
- **GitHub Actions**: CI/CD automation

## Visualization
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations

---

# Project Architecture

## Directory Structure

```
Kidney-Disease-Classification-Deep-Learning-Project/
│
├── config/
│   └── config.yaml              # Configuration for all stages
│
├── src/cnnClassifier/
│   ├── components/               # Core functionality modules
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── model_training.py
│   │   └── model_evaluation_mlflow.py
│   │
│   ├── pipeline/                 # Pipeline orchestration
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py
│   │
│   ├── config/
│   │   └── configuration.py     # Configuration manager
│   │
│   └── entity/
│       └── config_entity.py     # Data classes for config
│
├── artifacts/                    # Generated files and models
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   └── training/
│
├── research/                     # Jupyter notebooks for exploration
├── templates/                     # HTML templates for web UI
├── main.py                       # Main training script
├── app.py                        # Flask API application
├── params.yaml                   # Hyperparameters
├── dvc.yaml                      # DVC pipeline definition
└── Dockerfile                    # Container definition
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Flask API (app.py)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │
│  │   Train  │  │ Predict  │  │  Model Status    │     │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘     │
└───────┼─────────────┼──────────────────────────────────┘
        │             │
        ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                  Training Pipeline (main.py)             │
│                                                          │
│  Stage 1: Data Ingestion                                │
│         └──> Download & Extract                         │
│                                                          │
│  Stage 2: Prepare Base Model                            │
│         └──> Load VGG16 + Custom Head                   │
│                                                          │
│  Stage 3: Model Training                                │
│         └──> Train with Augmentation                    │
│                                                          │
│  Stage 4: Evaluation                                    │
│         └──> Metrics & MLflow Logging                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                      Artifacts                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │    Data      │  │ Base Model  │  │ Trained Model│  │
│  └──────────────┘  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
        │             │
        ▼             ▼
┌─────────────────────────────────────────────────────────┐
│              MLOps Tools                                 │
│  ┌──────────┐                      ┌──────────┐       │
│  │   DVC    │  Version Control      │  MLflow  │       │
│  │ Pipeline │  & Data Tracking      │ Tracking │       │
│  └──────────┘                      └──────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

# Pipeline Stages Explained

## Stage 1: Data Ingestion

**File**: `src/cnnClassifier/components/data_ingestion.py`

**What it does:**
1. Downloads the dataset from Google Drive using the provided URL
2. Extracts the ZIP file to the local file system
3. Organizes data for training

**Why:**
- Ensures data reproducibility by downloading from a stable source
- Automates data preparation process
- Version control with DVC tracks changes

**How it works:**
```python
def download_file(self):
    # Extract file ID from Google Drive URL
    # Use gdown to download the ZIP file
    # Save to local_data_file path
    
def extract_zip_file(self):
    # Unzip the downloaded file
    # Extract to kidney-ct-scan-image/ directory
    # Contains Normal/ and Tumor/ subdirectories
```

**Output**: 
- `artifacts/data_ingestion/kidney-ct-scan-image/`
  - `Normal/` (240 images)
  - `Tumor/` (225 images)

---

## Stage 2: Prepare Base Model

**File**: `src/cnnClassifier/components/prepare_base_model.py`

**What it does:**
1. Loads pre-trained VGG16 model (ImageNet weights)
2. Removes the original classifier head (1000 ImageNet classes)
3. Adds a custom classifier for binary classification (Normal/Tumor)
4. Saves the base model and updated model

**Why:**
- **Transfer Learning**: Leverages features learned on ImageNet
- **Efficiency**: Pre-trained features reduce training time
- **Performance**: ImageNet features generalize well to medical imaging

**Architecture Details:**

```python
# Original VGG16 Structure
Features: Conv layers (conv1_1 -> pool5)
Classifier: 3 FC layers (4096 -> 4096 -> 1000 classes)

# Modified for Kidney Classification
Features: Same VGG16 features (frozen)
New Classifier:
  - AdaptiveAvgPool2d (7x7)
  - Flatten
  - Linear(512*7*7 -> 4096) + ReLU + Dropout(0.5)
  - Linear(4096 -> 4096) + ReLU + Dropout(0.5)
  - Linear(4096 -> 2 classes)
```

**Outputs**:
- `base_model.pth`: VGG16 features only
- `base_model_updated.pth`: Features + custom classifier

---

## Stage 3: Model Training

**File**: `src/cnnClassifier/components/model_training.py`

**What it does:**
1. Loads the prepared base model
2. Splits data into train/validation (80/20)
3. Applies data augmentation to training set
4. Trains the model using transfer learning
5. Saves the trained model

**Why Data Augmentation?**
- Increases dataset diversity without collecting more data
- Prevents overfitting
- Improves model generalization

**Augmentation Techniques**:
```python
RandomRotation(40 degrees)      # Rotation invariance
RandomHorizontalFlip()         # Reflection invariance
RandomAffine(translate=0.2)   # Position invariance
RandomAffine(scale=0.8-1.2)   # Size invariance
```

**Training Process**:
1. **Forward Pass**: Input images → VGG16 features → Classifier → Predictions
2. **Loss Calculation**: Cross-entropy loss
3. **Backward Pass**: Gradient computation and optimization
4. **Validation**: Evaluate on hold-out set

**Optimizer**: SGD (Stochastic Gradient Descent)
- Learning Rate: 0.001
- Momentum: 0.9
- Batch Size: 32

**Loss Function**: CrossEntropyLoss
- Suitable for multi-class classification

**Output**:
- `artifacts/training/model.pth`: Trained model weights

---

## Stage 4: Model Evaluation

**File**: `src/cnnClassifier/components/model_evaluation_mlflow.py`

**What it does:**
1. Loads the trained model
2. Evaluates on validation data
3. Calculates metrics (accuracy, precision, recall, F1)
4. Logs metrics to MLflow
5. Saves scores to `scores.json`

**Why MLflow?**
- **Experiment Tracking**: Logs all hyperparameters
- **Model Versioning**: Tracks different model versions
- **Reproducibility**: Compare experiments easily
- **Model Registry**: Centralized model management

**Metrics Calculated**:
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

**Outputs**:
- `scores.json`: Model performance metrics
- MLflow tracking: All experiments logged

---

# Technical Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Web Browser  │  │ Mobile App   │  │ API Client   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask API Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  app.py: HTTP Server                                    │   │
│  │  ├─ GET  /               (Web UI)                       │   │
│  │  ├─ POST /train          (Trigger training)             │   │
│  │  ├─ POST /predict        (Make predictions)             │   │
│  │  └─ GET  /model-status   (Check model state)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Business Logic Layer                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PredictionPipeline (prediction.py)                      │   │
│  │  ├─ Load trained model                                   │   │
│  │  ├─ Preprocess image                                     │   │
│  │  ├─ Make prediction                                      │   │
│  │  └─ Return results                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## VGG16 Architecture

### Original VGG16 Structure

```
Input: 224×224×3 RGB Image
│
├─ Conv Block 1 (64 filters)
├─ Conv Block 2 (128 filters)
├─ Conv Block 3 (256 filters)
├─ Conv Block 4 (512 filters)
├─ Conv Block 5 (512 filters)
└─ Classifier (4096 → 4096 → 1000 classes)
```

### Modified for Kidney Classification

```
Features (Frozen):
├─ Conv Block 1-5 (Same as Original)
│  Output: 7×7×512 feature maps
│
└─ Custom Classifier:
   ├─ AdaptiveAvgPool2d(7×7) → 7×7×512
   ├─ Flatten → 25088
   ├─ Linear(25088→4096) + ReLU + Dropout(0.5)
   ├─ Linear(4096→4096) + ReLU + Dropout(0.5)
   └─ Linear(4096→2)  # Normal/Tumor
```

### Transfer Learning Strategy

**Why Transfer Learning?**

1. **Limited Data**: Only ~465 images total
   - Not enough for training from scratch
   - Would lead to severe overfitting

2. **Pre-trained Features**: ImageNet features are general
   - Edge detection (early layers)
   - Pattern recognition (middle layers)
   - High-level features (late layers)

3. **Efficiency**:
   - Faster training (only train classifier)
   - Lower computational cost
   - Better performance with less data

**Freezing Strategy**:
```python
# Freeze all feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False  # Don't update weights

# Only train classifier
for param in model.classifier.parameters():
    param.requires_grad = True   # Update weights
```

## Image Preprocessing

### Training Preprocessing
```python
transforms.Compose([
    # Resize to match VGG16 input
    transforms.Resize((224, 224)),
    
    # Augmentation (only on training)
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(translate=(0.2, 0.2)),
    transforms.RandomAffine(scale=(0.8, 1.2)),
    
    # Convert to tensor
    transforms.ToTensor(),
    
    # Normalize for ImageNet pretraining
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

---

# MLOps Practices

## DVC (Data Version Control)

**Purpose**: Track dataset and model versions like Git tracks code

**Benefits**:
- Version control for large files
- Reproducible experiments
- Track data lineage
- Efficient storage (only stores diffs)

**Pipeline Definition** (`dvc.yaml`):
```yaml
stages:
  data_ingestion:        # Stage 1
    cmd: python script
    deps: [source files]  # What triggers the stage
    outs: [output files]  # Outputs to track
    
  prepare_base_model:    # Stage 2
    depends on: Stage 1
    
  training:              # Stage 3
    depends on: Stage 1, 2
    
  evaluation:            # Stage 4
    depends on: Stage 3
```

**Commands**:
```bash
dvc init              # Initialize DVC repo
dvc repro             # Run entire pipeline
dvc dag               # Visualize pipeline
dvc status            # Check pipeline status
```

## MLflow

**Purpose**: Machine Learning Lifecycle Management

**Components Used**:
1. **Tracking**: Log experiments and metrics
2. **Registry**: Centralized model storage

**Features**:
- Track all hyperparameters
- Log metrics across experiments
- Compare model versions
- Register best models

**Setup**:
```bash
mlflow ui              # Start MLflow UI
# Visit http://localhost:5000
```

**Why MLflow?**
- Centralized experiment tracking
- Compare multiple experiments
- Reproducible research
- Model governance

## Docker Containerization

**Purpose**: Consistent environment across development and production

**Dockerfile Overview**:
```dockerfile
FROM python:3.10.18-slim-buster

RUN apt-get update && apt-get install awscli

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]
```

**Why Docker?**
- Consistent environment
- Isolated dependencies
- Easy deployment
- Scalability

---

# Setup and Installation

## Prerequisites

- Python 3.8+
- Conda (recommended)
- Git
- Docker (optional, for deployment)

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Kidney-Disease-Classification-Deep-Learning-Project
```

### 2. Create Conda Environment
```bash
# Create environment
conda create -n cnncls python=3.8 -y

# Activate environment
conda activate cnncls
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training Pipeline
```bash
# Option 1: Run all stages
python main.py

# Option 2: Run with DVC
dvc repro

# Option 3: Run individual stage
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
```

### 5. Start Web Application
```bash
python app.py

# Visit http://localhost:8080
```

---

# API Documentation

## Flask Application Endpoints

### 1. Home Endpoint

**GET /** 
- **Purpose**: Display web interface for model interaction
- **Response**: HTML page (`templates/index.html`)

**Example**:
```
GET http://localhost:8080/
```

### 2. Train Endpoint

**POST /train**
- **Purpose**: Trigger model training pipeline
- **Method**: GET, POST
- **Response**: Success message or error

**Example Request**:
```bash
curl http://localhost:8080/train
```

**Example Response** (Success):
```json
"Training done successfully!"
```

**What Happens**:
1. Executes `main.py`
2. Runs all 4 pipeline stages
3. Saves trained model to `artifacts/training/model.pth`
4. Logs all outputs

### 3. Predict Endpoint

**POST /predict**
- **Purpose**: Classify kidney CT scan image
- **Request Body**: JSON with base64-encoded image
- **Response**: Prediction and confidence score

**Request Format**:
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response Format**:
```json
[
  {
    "image": "Tumor",        # or "Normal"
    "confidence": 0.9456     # confidence score
  }
]
```

**Example**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_string"}'
```

**What Happens**:
1. Receives base64 image
2. Decodes and saves to `inputImage.jpg`
3. Loads trained model from `artifacts/training/model.pth`
4. Preprocesses image (resize, normalize)
5. Makes prediction
6. Returns class and confidence

### 4. Model Status Endpoint

**GET /model-status**
- **Purpose**: Check if model is trained and ready
- **Response**: Model status and path

**Example Response** (Model Ready):
```json
{
  "status": "ready",
  "model_path": "artifacts/training/model.pth"
}
```

**Example Response** (Not Trained):
```json
{
  "status": "not_found",
  "message": "Model needs to be trained"
}
```

---

# Deployment Guide

## Docker Deployment

### 1. Build Docker Image
```bash
docker build -t kidney-classifier:latest .
```

### 2. Run Container
```bash
docker run -p 8080:8080 kidney-classifier:latest
```

### 3. Access Application
```bash
# Open browser to
http://localhost:8080
```

## AWS Deployment

### Prerequisites
1. AWS Account
2. IAM User with EC2 and ECR access
3. GitHub Secrets configured

### Setup Steps

#### 1. Create IAM User
- Create user with policies:
  - `AmazonEC2ContainerRegistryFullAccess`
  - `AmazonEC2FullAccess`
- Save Access Key ID and Secret Access Key

#### 2. Create ECR Repository
```bash
aws ecr create-repository --repository-name kidney-classifier
```
- Save the URI

#### 3. Create EC2 Instance
- Launch Ubuntu EC2 instance
- Configure security group (allow port 8080)
- Save EC2 IP address

#### 4. Install Docker on EC2
```bash
sudo apt-get update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 5. Setup GitHub Secrets
Add to GitHub repository settings:
```
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
AWS_REGION=us-east-1
AWS_ECR_LOGIN_URI=<ecr_uri>
ECR_REPOSITORY_NAME=kidney-classifier
```

---

# Troubleshooting

## Common Issues and Solutions

### Issue 1: Model Not Found
**Error**: `FileNotFoundError: artifacts/training/model.pth`

**Solution**:
1. Train model first: `python main.py` or visit `/train`
2. Check if file exists: `ls artifacts/training/`
3. Verify training completed successfully

### Issue 2: CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce batch size in `params.yaml`
2. Use CPU: `device = 'cpu'`
3. Reduce input image size
4. Close other applications using GPU

### Issue 3: Data Download Fails
**Error**: `Error downloading data`

**Solution**:
1. Check internet connection
2. Verify Google Drive URL is accessible
3. Try manual download:
   ```bash
   gdown <file_id> -O artifacts/data_ingestion/data.zip
   ```
4. Check Google Drive permissions

### Issue 4: DVC Pipeline Issues
**Error**: `dvc repro fails`

**Solution**:
1. Check DVC version: `dvc --version`
2. Verify dependencies in `dvc.yaml`
3. Run stages individually
4. Clean artifacts: `rm -rf artifacts/`

---

# Quick Reference

## Common Commands

### Training Commands
```bash
# Run entire pipeline
python main.py

# Run with DVC
dvc repro

# Run individual stage
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
```

### DVC Commands
```bash
dvc init              # Initialize DVC
dvc repro             # Run pipeline
dvc dag               # Show pipeline graph
dvc status            # Check pipeline status
```

### MLflow Commands
```bash
mlflow ui              # Start MLflow UI (http://localhost:5000)
mlflow runs list       # List runs
```

### Docker Commands
```bash
# Build image
docker build -t kidney-classifier:latest .

# Run container
docker run -p 8080:8080 kidney-classifier:latest

# View logs
docker logs <container_id>
```

## API Endpoints

### 1. Train Model
```bash
curl http://localhost:8080/train
```

### 2. Make Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image"}'
```

### 3. Check Model Status
```bash
curl http://localhost:8080/model-status
```

---

# Best Practices

## Code Organization
- ✅ Modular design with clear separation of concerns
- ✅ Configuration centralized in YAML files
- ✅ Consistent error handling and logging
- ✅ Type hints for better code readability

## Data Management
- ✅ Version control with DVC
- ✅ Keep raw data separate from processed data
- ✅ Document data sources and transformations
- ✅ Use reproducible seeds for train/test splits

## Model Development
- ✅ Use transfer learning for small datasets
- ✅ Implement data augmentation
- ✅ Track experiments with MLflow
- ✅ Validate on hold-out set
- ✅ Save model checkpoints

## Deployment
- ✅ Containerize with Docker
- ✅ Use CI/CD pipelines
- ✅ Implement health checks
- ✅ Monitor model performance
- ✅ Version model artifacts

## Security
- ✅ Don't commit credentials
- ✅ Use environment variables for secrets
- ✅ Validate input data
- ✅ Implement rate limiting
- ✅ Use HTTPS in production

---

## Performance Metrics

### Model Performance
- **Expected Accuracy**: 85-95% (depending on data quality)
- **Training Time**: ~10-30 minutes per epoch (GPU)
- **Inference Time**: <1 second per image

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **GPU**: Optional (CUDA or MPS)
- **Disk**: 5GB for data + models
- **Network**: For downloading data

---

## References

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)

### Resources
- Original Dataset: [Google Drive](https://drive.google.com/file/d/1vlhZ5c7abUKF8xXERIw6m9Te8fW7ohw3/view)
- MLflow Tutorials: [Official Tutorials](https://mlflow.org/docs/latest/tutorials.html)
- Docker Guides: [Docker Documentation](https://docs.docker.com/)

---

## Conclusion

This project demonstrates a complete MLOps pipeline for medical image classification using:
- **Deep Learning** (PyTorch + Transfer Learning)
- **MLOps Tools** (DVC + MLflow)
- **Web Deployment** (Flask + Docker)
- **Cloud Infrastructure** (AWS)

Follow this documentation to understand, set up, and deploy the kidney disease classification system.

---

**Last Updated**: 2024
**Version**: 1.0
**Maintainer**: Project Team
**License**: Check LICENSE file


