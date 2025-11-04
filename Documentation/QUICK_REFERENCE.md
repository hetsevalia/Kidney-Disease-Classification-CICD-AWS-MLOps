# Quick Reference Guide

## 🚀 Quick Start

### Setup Environment
```bash
# 1. Create conda environment
conda create -n cnncls python=3.8 -y
conda activate cnncls

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python main.py

# 4. Start web app
python app.py
```

Visit: http://localhost:8080

---

## 📁 Project Structure

```
├── config/           → Configuration files
├── src/cnnClassifier/ 
│   ├── components/   → Core functionality
│   ├── pipeline/     → Stage orchestrators
│   ├── config/       → Config manager
│   └── entity/       → Data classes
├── artifacts/        → Generated files
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   └── training/
└── research/         → Jupyter notebooks
```

---

## 🔧 Configuration Files

### config/config.yaml
```yaml
data_ingestion:
  source_URL: <google_drive_url>
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  base_model_path: artifacts/prepare_base_model/base_model.pth
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.pth

training:
  trained_model_path: artifacts/training/model.pth
```

### params.yaml
```yaml
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 32
EPOCHS: 2
LEARNING_RATE: 0.001
CLASSES: 2
AUGMENTATION: true
```

---

## 🔄 Common Commands

### Training Commands
```bash
# Run entire pipeline
python main.py

# Run with DVC
dvc repro

# Run individual stage
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
python src/cnnClassifier/pipeline/stage_03_model_training.py
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
```

### DVC Commands
```bash
dvc init              # Initialize DVC
dvc repro             # Run pipeline
dvc dag               # Show pipeline graph
dvc status            # Check pipeline status
dvc add <file>        # Start tracking file
```

### MLflow Commands
```bash
mlflow ui              # Start MLflow UI (http://localhost:5000)
mlflow runs list       # List runs
mlflow runs describe   # Show run details
```

### Docker Commands
```bash
# Build image
docker build -t kidney-classifier:latest .

# Run container
docker run -p 8080:8080 kidney-classifier:latest

# Check running containers
docker ps

# View logs
docker logs <container_id>
```

---

## 🌐 API Endpoints

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

## 🔍 Troubleshooting

### Problem: Model not found
**Solution**: 
```bash
python main.py  # Train the model first
```

### Problem: Import errors
**Solution**:
```bash
pip install -e .  # Install package in development mode
```

### Problem: Port 8080 in use
**Solution**:
```bash
# Find process
lsof -i :8080

# Kill process
kill -9 <PID>
```

### Problem: CUDA out of memory
**Solution**:
1. Reduce `BATCH_SIZE` in `params.yaml`
2. Use CPU mode
3. Close other GPU applications

---

## 📊 Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Main entry point - runs all pipeline stages |
| `app.py` | Flask web application with API endpoints |
| `config/config.yaml` | Paths and directory configuration |
| `params.yaml` | Hyperparameters (learning rate, batch size, etc.) |
| `dvc.yaml` | DVC pipeline definition |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition |

---

## 🧪 Testing Individual Stages

### Test Data Ingestion
```bash
cd src/cnnClassifier/pipeline
python stage_01_data_ingestion.py
# Check: artifacts/data_ingestion/kidney-ct-scan-image/
```

### Test Base Model
```bash
python stage_02_prepare_base_model.py
# Check: artifacts/prepare_base_model/
```

### Test Training
```bash
python stage_03_model_training.py
# Check: artifacts/training/model.pth
```

### Test Evaluation
```bash
python stage_04_model_evaluation.py
# Check: scores.json
```

---

## 📈 Monitoring

### View Logs
```bash
# View training logs
tail -f logs/running_logs.log

# View DVC logs
dvc repro --verbose

# View Docker logs
docker logs -f <container_id>
```

### Check Model Performance
```bash
# View scores
cat scores.json

# Check MLflow
# Visit: http://localhost:5000
```

---

## 🐛 Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Package not installed | `pip install -r requirements.txt` |
| `FileNotFoundError` | Model not trained | Run `python main.py` |
| `CUDA out of memory` | GPU memory full | Reduce batch size |
| `Port already in use` | Port 8080 occupied | Kill process on port |
| `Data download failed` | Network issue | Check internet/URL |

---

## 🔐 Environment Variables

### For MLflow (Dagshub)
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password
```

### For AWS Deployment
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

---

## 📝 Code Examples

### Load Trained Model
```python
import torch
import torchvision.models as models
import torch.nn as nn

# Create model architecture
vgg16 = models.vgg16(pretrained=False)
features = nn.Sequential(*list(vgg16.features.children()))
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Flatten(),
    nn.Linear(512*7*7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 2)
)
model = nn.Sequential(features, classifier)

# Load weights
model.load_state_dict(torch.load('artifacts/training/model.pth'))
```

### Make Prediction
```python
from cnnClassifier.pipeline.prediction import PredictionPipeline

pipeline = PredictionPipeline('image.jpg')
result = pipeline.predict()
# Returns: [{"image": "Normal", "confidence": 0.9456}]
```

---

## 📚 Useful Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)

---

## 💡 Tips

1. **Use GPU**: Training is much faster on GPU
2. **Adjust Epochs**: More epochs may improve accuracy
3. **Check Metrics**: Use MLflow UI to compare experiments
4. **Version Control**: Use DVC to track data and models
5. **Monitor Logs**: Check logs for errors and warnings
6. **Test Locally**: Test API endpoints before deployment
7. **Backup Models**: Keep good model versions
8. **Document Changes**: Document hyperparameter changes

---

## 🎯 Next Steps

1. ✅ Set up environment
2. ✅ Train model (`python main.py`)
3. ✅ Test API (`python app.py`)
4. ✅ Evaluate performance
5. ✅ Deploy to production (optional)

---

## 📞 Support

For issues or questions:
- Check project documentation
- Review logs for errors
- Search GitHub issues
- Contact project maintainers

---

**Last Updated**: 2024
**Version**: 1.0

