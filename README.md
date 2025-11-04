# Kidney Disease Classification - MLOps Project

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-green.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.2-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.0-blueviolet.svg)](https://dvc.org/)

An end-to-end MLOps pipeline for classifying kidney CT scan images using deep learning, transfer learning, and modern MLOps practices.

## 📚 Documentation

- **[PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md)** - Complete project documentation with step-by-step explanations
- **[TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md)** - In-depth technical architecture details
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick reference guide for developers

## Features

- ✅ Automated data ingestion from Google Drive
- ✅ Transfer learning with VGG16 (ImageNet pre-trained)
- ✅ Data augmentation for improved generalization
- ✅ Experiment tracking with MLflow
- ✅ Version control for data/models with DVC
- ✅ Flask REST API for predictions
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions
- ✅ AWS deployment support

## Quick Start

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

Visit http://localhost:8080

## Pipeline Stages

1. **Data Ingestion** - Download and extract kidney CT scan dataset from Google Drive
2. **Prepare Base Model** - Load VGG16, modify for binary classification (Normal/Tumor)
3. **Model Training** - Train model with data augmentation on 80/20 train/val split
4. **Evaluation** - Evaluate model performance and log metrics to MLflow

Each stage is orchestrated with DVC for reproducibility.

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```






## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow \
MLFLOW_TRACKING_USERNAME=entbappy \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow

export MLFLOW_TRACKING_USERNAME=entbappy 

export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0

```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


## API Endpoints

### Web Interface
- `GET /` - Home page with interactive UI

### Training
- `POST /train` - Trigger model training pipeline

### Prediction
- `POST /predict` - Classify kidney CT scan image
  - Request: `{"image": "base64_encoded_image"}`
  - Response: `[{"image": "Normal/Tumor", "confidence": 0.95}]`

### Status
- `GET /model-status` - Check if model is trained and ready

## Technology Stack

- **Deep Learning**: PyTorch, Torchvision
- **MLOps**: DVC (Data Version Control), MLflow (Experiment Tracking)
- **Web Framework**: Flask, Flask-CORS
- **Data Processing**: Pandas, NumPy, PIL
- **Deployment**: Docker, AWS (EC2, ECR)
- **CI/CD**: GitHub Actions

## Model Details

- **Architecture**: VGG16 (Transfer Learning)
- **Input Size**: 224×224×3 RGB images
- **Classes**: Normal (0) and Tumor (1)
- **Training**: SGD optimizer, CrossEntropyLoss, batch size 32
- **Augmentation**: Random rotation, flip, translation, scaling

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details

## Acknowledgments

- Dataset from Google Drive
- VGG16 architecture by Karen Simonyan and Andrew Zisserman
- MLflow and DVC communities
