# Kidney Disease Classification - Startup & Verification Guide

## 📋 Project Overview

This is an MLOps project for classifying kidney CT scan images (Normal/Tumor) using:
- **PyTorch** with VGG16 transfer learning
- **DVC** for pipeline orchestration and data versioning
- **MLflow** for experiment tracking
- **Flask** for REST API predictions

## ✅ Quick Verification Checklist

Before starting, verify your setup:

### 1. **Check Environment**
```bash
# Activate your conda environment
conda activate cnncls

# Verify Python version (should be 3.8)
python --version

# Verify you're in the project directory
pwd  # Should be: /Users/shubh/Desktop/kidney_Disease_MLOPS/Kidney-Disease-Classification-Deep-Learning-Project
```

### 2. **Verify Dependencies**
```bash
# Check if all packages are installed
pip list | grep -E "torch|mlflow|dvc|flask"

# If missing, install:
pip install -r requirements.txt
```

### 3. **Verify Package Installation**
```bash
# Check if cnnClassifier package is installed
python -c "from cnnClassifier import logger; print('Package installed successfully!')"

# If not installed, install in editable mode:
pip install -e .
```

### 4. **Check DVC Status**
```bash
# Verify DVC is initialized
dvc status

# Check DVC pipeline
dvc dag

# View DVC stages
cat dvc.yaml
```

### 5. **Verify Required Files Exist**
```bash
# Check config files
ls -la config/config.yaml params.yaml

# Check if data exists (should be already extracted)
ls -la artifacts/data_ingestion/kidney-ct-scan-image/

# Check if models exist
ls -la artifacts/training/model.pth
ls -la artifacts/prepare_base_model/base_model.pth
```

---

## 🚀 Starting the Project

### **Option 1: Run Full Pipeline (Recommended for First Time)**

This will run all 4 stages: Data Ingestion → Prepare Base Model → Training → Evaluation

```bash
# Make sure you're in the project root
cd /Users/shubh/Desktop/kidney_Disease_MLOPS/Kidney-Disease-Classification-Deep-Learning-Project

# Activate environment
conda activate cnncls

# Run the full pipeline
python main.py
```

**OR using DVC (recommended for reproducibility):**
```bash
# Run entire DVC pipeline
dvc repro

# Or run a specific stage
dvc repro training
```

**Expected Output:**
- Data will be downloaded/extracted (if not already present)
- Base model (VGG16) will be prepared
- Model will train for 2 epochs (as per params.yaml)
- Evaluation will run and log to MLflow
- Final model saved at: `artifacts/training/model.pth`

**Check logs:**
```bash
tail -f logs/running_logs.log
```

---

### **Option 2: Test with Existing Model (Quick Start)**

If models already exist, you can directly start the Flask app:

```bash
# Verify model exists
ls -la artifacts/training/model.pth

# Start Flask application
python app.py
```

Then open in browser:
```
http://localhost:8080
```

---

## 🧪 Testing Each Component

### **1. Test Individual Pipeline Stages**

```bash
# Test Data Ingestion
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py

# Test Prepare Base Model
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py

# Test Training
python src/cnnClassifier/pipeline/stage_03_model_training.py

# Test Evaluation
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
```

### **2. Test Flask API**

```bash
# Start the Flask app
python app.py

# In another terminal, test the endpoints:
```

**Check Model Status:**
```bash
curl http://localhost:8080/model-status
```

**Expected Response:**
```json
{
  "status": "ready",
  "model_path": "artifacts/training/model.pth"
}
```

**Test Training Endpoint (if needed):**
```bash
curl -X POST http://localhost:8080/train
```

**Test Prediction (via Web UI):**
1. Open http://localhost:8080 in browser
2. Upload a kidney CT scan image
3. Click "Predict"

---

## 📊 Monitor MLflow (Optional)

If you want to track experiments:

```bash
# Start MLflow UI
mlflow ui

# Open in browser
# http://localhost:5000
```

**Note:** The project is configured to use Dagshub MLflow tracking URI by default. Check `src/cnnClassifier/config/configuration.py` for MLflow settings.

---

## 🔍 Verification Script

Run this comprehensive check:

```bash
#!/bin/bash
echo "=== Project Verification ==="

# Check Python
echo "1. Python version:"
python --version

# Check Package
echo "2. Testing package import:"
python -c "from cnnClassifier import logger; print('✓ Package OK')" || echo "✗ Package NOT installed"

# Check Files
echo "3. Checking required files:"
[ -f "config/config.yaml" ] && echo "✓ config.yaml" || echo "✗ config.yaml missing"
[ -f "params.yaml" ] && echo "✓ params.yaml" || echo "✗ params.yaml missing"
[ -f "artifacts/training/model.pth" ] && echo "✓ Model exists" || echo "✗ Model missing"
[ -d "artifacts/data_ingestion/kidney-ct-scan-image" ] && echo "✓ Data exists" || echo "✗ Data missing"

# Check DVC
echo "4. DVC status:"
dvc status

echo "=== Verification Complete ==="
```

---

## 🐛 Common Issues & Solutions

### **Issue 1: ModuleNotFoundError: No module named 'cnnClassifier'**
**Solution:**
```bash
pip install -e .
```

### **Issue 2: Model not found error**
**Solution:**
```bash
# Train the model first
python main.py
```

### **Issue 3: DVC errors**
**Solution:**
```bash
# Since DVC is working fine (as you mentioned), just verify:
dvc status
dvc dag
```

### **Issue 4: CUDA/MPS errors**
**Solution:**
- The code auto-detects device (CUDA/MPS/CPU)
- For Mac with Apple Silicon, MPS will be used automatically
- For CPU, it will fallback gracefully

### **Issue 5: Port 8080 already in use**
**Solution:**
```bash
# Change port in app.py (line 82) or kill existing process:
lsof -ti:8080 | xargs kill -9
```

---

## 📝 What to Check After Running

1. **Logs:** Check `logs/running_logs.log` for any errors
2. **Model:** Verify `artifacts/training/model.pth` exists and has reasonable size (>10MB)
3. **Scores:** Check `scores.json` for accuracy metrics
4. **DVC:** Run `dvc status` to verify pipeline state
5. **Web App:** Test prediction at http://localhost:8080

---

## 🎯 Recommended First Steps

1. **Verify Setup:**
   ```bash
   python -c "from cnnClassifier import logger; print('Setup OK')"
   ```

2. **Check if model exists:**
   ```bash
   ls -la artifacts/training/model.pth
   ```

3. **If model exists, start app:**
   ```bash
   python app.py
   ```

4. **If model doesn't exist, run training:**
   ```bash
   python main.py
   ```

5. **After training, start app:**
   ```bash
   python app.py
   ```

6. **Test in browser:**
   - Open http://localhost:8080
   - Upload a kidney CT scan image
   - Verify prediction works

---

## ✅ Success Indicators

You'll know everything is working when:
- ✅ `python main.py` runs without errors
- ✅ Model file exists at `artifacts/training/model.pth`
- ✅ `python app.py` starts Flask server
- ✅ Web UI loads at http://localhost:8080
- ✅ Predictions return Normal/Tumor with confidence scores
- ✅ DVC status shows pipeline is up-to-date

---

## 📞 Next Steps

After verifying everything works:
1. Try modifying `params.yaml` (epochs, batch_size) and retrain
2. Experiment with different hyperparameters
3. Check MLflow for experiment tracking
4. Deploy using Docker: `docker build -t kidney-classifier .`
5. Set up CI/CD with GitHub Actions (as documented in README)

---

**Good luck! 🚀**

