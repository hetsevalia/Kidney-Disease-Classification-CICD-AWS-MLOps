# Technical Architecture Documentation

## Overview
This document provides in-depth technical details about the Kidney Disease Classification MLOps project architecture, design decisions, and implementation details.

---

## System Architecture

### High-Level Architecture

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
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Training Pipeline Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Stage 1:     │→ │ Stage 2:     │→ │ Stage 3:          │→ │
│  │ Data         │  │ Base Model   │  │ Training          │  │
│  │ Ingestion    │  │ Preparation  │  │                   │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
│                                                     │           │
│                          ┌─────────────────────────────────┐   │
│                          ▼                                 │   │
│                   ┌──────────────┐                        │   │
│                   │ Stage 4:     │                        │   │
│                   │ Evaluation   │◄───────────────────────┘   │
│                   └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION STAGE                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Google Drive ──┐                                            │
│                 │                                            │
│                 ▼                                            │
│        data.zip (downloaded)                                │
│                 │                                            │
│                 ▼                                            │
│        Extract ZIP                                          │
│                 │                                            │
│                 ▼                                            │
│    kidney-ct-scan-image/                                    │
│          ├─ Normal/ (240 images)                             │
│          └─ Tumor/ (225 images)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPARE BASE MODEL STAGE                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PyTorch VGG16                                              │
│       │                                                      │
│       ▼                                                      │
│  Download Pre-trained Weights (ImageNet)                    │
│       │                                                      │
│       ▼                                                      │
│  Remove Classifier Head (1000 → Custom)                     │
│       │                                                      │
│       ▼                                                      │
│  Add Custom Binary Classifier                               │
│       │                                                      │
│       ├─ base_model.pth                                      │
│       └─ base_model_updated.pth                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING STAGE                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Dataset Split (80/20)                                      │
│       │                                                      │
│       ├─ Train Set (80%)                                    │
│       │   ├─ Data Augmentation                              │
│       │   └─ Batch Processing                               │
│       │                                                      │
│       └─ Validation Set (20%)                               │
│           └─ No Augmentation                                │
│                                                              │
│  Training Loop                                              │
│       │                                                      │
│       ├─ Forward Pass: Image → Model → Prediction           │
│       ├─ Loss Calculation: CrossEntropyLoss                 │
│       ├─ Backward Pass: Gradient Computation                │
│       ├─ Optimization: SGD Update                          │
│       └─ Validation: Evaluate on Val Set                    │
│                                                              │
│       Save: model.pth                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MODEL EVALUATION STAGE                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Load Trained Model                                         │
│       │                                                      │
│       ▼                                                      │
│  Evaluate on Validation Set                                 │
│       │                                                      │
│       ├─ Calculate Metrics:                                │
│       │  ├─ Accuracy                                        │
│       │  ├─ Precision                                       │
│       │  ├─ Recall                                          │
│       │  └─ F1 Score                                       │
│       │                                                      │
│       ├─ Log to MLflow                                      │
│       └─ Save to scores.json                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Prediction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Receive Image (Base64 Encoded)                          │
│       │                                                      │
│       ▼                                                      │
│  2. Decode Image                                            │
│       │                                                      │
│       ▼                                                      │
│  3. Load Trained Model (model.pth)                          │
│       │                                                      │
│       ▼                                                      │
│  4. Preprocess Image                                        │
│       ├─ Resize to 224x224                                  │
│       ├─ Convert to Tensor                                  │
│       └─ Normalize [0.485, 0.456, 0.406]                    │
│       │                                                      │
│       ▼                                                      │
│  5. Forward Pass                                            │
│       ├─ Extract Features (VGG16)                           │
│       ├─ Classify (Custom Head)                             │
│       └─ Get Logits                                         │
│       │                                                      │
│       ▼                                                      │
│  6. Post-process                                            │
│       ├─ Softmax (Probabilities)                            │
│       ├─ Argmax (Prediction)                                │
│       └─ Confidence Score                                   │
│       │                                                      │
│       ▼                                                      │
│  7. Return JSON Response                                    │
│       {                                                      │
│         "image": "Normal" or "Tumor",                       │
│         "confidence": 0.9456                                │
│       }                                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### VGG16 Architecture

#### Original VGG16 Structure
```
Input: 224×224×3 RGB Image
│
├─ Conv Block 1
│  ├─ Conv2d(3→64, 3×3, padding=1) + ReLU
│  ├─ Conv2d(64→64, 3×3, padding=1) + ReLU
│  └─ MaxPool2d(2×2, stride=2) → 112×112×64
│
├─ Conv Block 2
│  ├─ Conv2d(64→128, 3×3, padding=1) + ReLU
│  ├─ Conv2d(128→128, 3×3, padding=1) + ReLU
│  └─ MaxPool2d(2×2, stride=2) → 56×56×128
│
├─ Conv Block 3
│  ├─ Conv2d(128→256, 3×3, padding=1) + ReLU
│  ├─ Conv2d(256→256, 3×3, padding=1) + ReLU
│  ├─ Conv2d(256→256, 3×3, padding=1) + ReLU
│  └─ MaxPool2d(2×2, stride=2) → 28×28×256
│
├─ Conv Block 4
│  ├─ Conv2d(256→512, 3×3, padding=1) + ReLU
│  ├─ Conv2d(512→512, 3×3, padding=1) + ReLU
│  ├─ Conv2d(512→512, 3×3, padding=1) + ReLU
│  └─ MaxPool2d(2×2, stride=2) → 14×14×512
│
├─ Conv Block 5
│  ├─ Conv2d(512→512, 3×3, padding=1) + ReLU
│  ├─ Conv2d(512→512, 3×3, padding=1) + ReLU
│  ├─ Conv2d(512→512, 3×3, padding=1) + ReLU
│  └─ MaxPool2d(2×2, stride=2) → 7×7×512
│
└─ Classifier (Original)
   ├─ Linear(25088→4096) + ReLU + Dropout(0.5)
   ├─ Linear(4096→4096) + ReLU + Dropout(0.5)
   └─ Linear(4096→1000)  # ImageNet classes
```

#### Modified for Kidney Classification
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
   - Medical images share some visual patterns

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

**Why Freeze Features?**
- VGG16 features are already trained on millions of images
- More robust to overfitting
- Focus training on task-specific classifier
- Faster training

---

## Configuration Management

### Configuration Hierarchy

```
├─ config/config.yaml (Global Configuration)
│  ├─ data_ingestion:
│  │  ├─ root_dir
│  │  ├─ source_URL
│  │  ├─ local_data_file
│  │  └─ unzip_dir
│  │
│  ├─ prepare_base_model:
│  │  ├─ root_dir
│  │  ├─ base_model_path
│  │  └─ updated_base_model_path
│  │
│  └─ training:
│     ├─ root_dir
│     └─ trained_model_path
│
├─ params.yaml (Hyperparameters)
│  ├─ IMAGE_SIZE: [224, 224, 3]
│  ├─ BATCH_SIZE: 32
│  ├─ EPOCHS: 2
│  ├─ LEARNING_RATE: 0.001
│  ├─ CLASSES: 2
│  ├─ AUGMENTATION: true
│  ├─ INCLUDE_TOP: false
│  └─ WEIGHTS: imagenet
│
└─ dvc.yaml (Pipeline Definition)
   ├─ data_ingestion stage
   ├─ prepare_base_model stage
   ├─ training stage
   └─ evaluation stage
```

### Why This Architecture?

**Separation of Concerns**:
- Config: Paths and directories
- Params: Hyperparameters (easy to tune)
- DVC: Pipeline dependencies

**Benefits**:
- Easy to modify without code changes
- Version control for experiments
- Reproducible results
- Clear documentation

---

## Data Processing Pipeline

### Image Preprocessing

#### Training Preprocessing
```python
transforms.Compose([
    # Resize to match VGG16 input
    transforms.Resize((224, 224)),
    
    # Augmentation (only on training)
    transforms.RandomRotation(40),         # Rotate ±40 degrees
    transforms.RandomHorizontalFlip(),     # Mirror horizontally
    transforms.RandomAffine(translate=(0.2, 0.2)),  # Shift position
    transforms.RandomAffine(scale=(0.8, 1.2)),      # Scale images
    
    # Convert to tensor
    transforms.ToTensor(),  # PIL Image → Tensor [0, 1]
    
    # Normalize for ImageNet pretraining
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

#### Validation Preprocessing
```python
transforms.Compose([
    transforms.Resize((224, 224)),  # No augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Why ImageNet Normalization?**
- VGG16 was trained on ImageNet with these statistics
- Ensures features are in expected range
- Improves model performance

---

## Training Strategy

### Training Configuration

```python
Optimizer: SGD(
    lr=0.001,      # Learning rate
    momentum=0.9   # Momentum for stable training
)

Loss Function: CrossEntropyLoss()
  # Good for multi-class classification
  # Handles class imbalance

Batch Size: 32
  # Balance between memory and stability
  # Larger batch = more stable gradients
  # Smaller batch = more gradient noise (regularization)

Epochs: 2
  # Can be increased for better performance
  # Watch for overfitting
```

### Training Loop Details

```python
for epoch in range(num_epochs):
    # Training Phase
    model.train()  # Enable dropout, batch norm training mode
    for batch in train_loader:
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        accuracy = calculate_accuracy(predictions, labels)
    
    # Validation Phase
    model.eval()  # Disable dropout, batch norm eval mode
    with torch.no_grad():  # No gradient computation
        for batch in val_loader:
            predictions = model(images)
            loss = criterion(predictions, labels)
            accuracy = calculate_accuracy(predictions, labels)
    
    # Log metrics
    log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
```

**Why Eval Mode?**
- Dropout disabled (uses all neurons)
- BatchNorm uses running statistics
- Consistent with inference mode
- Accurate validation metrics

---

## MLOps Integration

### DVC Pipeline

```yaml
stages:
  data_ingestion:       # Dependencies: source code + config
    cmd: python script
    deps: [code, config]
    outs: [dataset]
    # Runs when code/config changes

  prepare_base_model:   # Dependencies: config
    cmd: python script
    deps: [code, config]
    params: [IMAGE_SIZE, CLASSES, LEARNING_RATE]
    outs: [base_models]
    # Runs when params change

  training:             # Dependencies: dataset + base model
    cmd: python script
    deps: [dataset, base_model, code]
    params: [BATCH_SIZE, EPOCHS, AUGMENTATION]
    outs: [trained_model]
    # Runs when any dependency changes

  evaluation:           # Dependencies: trained model
    cmd: python script
    deps: [trained_model, code]
    params: [IMAGE_SIZE, BATCH_SIZE]
    metrics: [scores.json]
    # Runs after training
```

**Benefits**:
- Automatic dependency resolution
- Reproducible experiments
- Clear pipeline visualization
- Efficient caching

### MLflow Integration

```python
# Training
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    })
    
    # Log metrics during training
    mlflow.log_metrics({
        'train_accuracy': 0.92,
        'val_accuracy': 0.88
    })
    
    # Log model
    mlflow.pytorch.log_model(model, 'model')
    
    # Save artifacts
    mlflow.log_artifact('model.pth')
```

**Tracking**:
- All hyperparameters
- Training metrics per epoch
- Model artifacts
- Code version (Git hash)
- Environment details

---

## API Design

### RESTful API Design

#### Endpoint Structure
```
Base URL: http://localhost:8080

├─ GET  /               - Web UI
├─ POST /train          - Trigger training
├─ POST /predict        - Make prediction
└─ GET  /model-status   - Check model
```

#### Request/Response Formats

**Train Endpoint**:
```python
# Request: None (triggered via button)
POST /train

# Response
"Training done successfully!"
```

**Predict Endpoint**:
```python
# Request
POST /predict
Content-Type: application/json
{
  "image": "base64_encoded_string"
}

# Response
[
  {
    "image": "Normal",      # or "Tumor"
    "confidence": 0.9456
  }
]
```

**Model Status Endpoint**:
```python
# Request
GET /model-status

# Response (Trained)
{
  "status": "ready",
  "model_path": "artifacts/training/model.pth"
}

# Response (Not Trained)
{
  "status": "not_found",
  "message": "Model needs to be trained"
}
```

---

## Error Handling

### Error Types

1. **Model Not Found**:
   - Status: 404
   - Message: "Model file not found"
   - Solution: Train model first

2. **Invalid Image**:
   - Status: 500
   - Message: "Invalid image format"
   - Solution: Check image encoding

3. **Training Failure**:
   - Status: 500
   - Message: Error details
   - Solution: Check logs

4. **Server Error**:
   - Status: 500
   - Message: Exception details
   - Solution: Check stack trace

---

## Security Considerations

### Input Validation
- Validate image format before processing
- Check file size limits
- Sanitize all inputs

### Authentication (Future)
- Add API key authentication
- Implement rate limiting
- Log all requests

### Model Security
- Don't expose model weights publicly
- Use secure storage (encrypted)
- Version control for audits

---

## Performance Optimization

### Model Optimization
- Use GPU when available
- Batch inference for multiple images
- Quantize model (reduce precision)
- Model pruning (remove unnecessary weights)

### API Optimization
- Implement caching for frequent predictions
- Use async processing for long operations
- Add response compression
- Connection pooling

---

## Monitoring and Logging

### Logging Strategy
```python
# Structured logging
logger.info(f"Stage {STAGE_NAME} started")
logger.debug(f"Parameters: {params}")
logger.error(f"Error occurred: {e}", exc_info=True)
```

### Metrics to Monitor
- Request rate
- Response time
- Error rate
- Model accuracy over time
- System resource usage

---

## Conclusion

This architecture provides:
- **Scalability**: Modular design allows easy expansion
- **Maintainability**: Clear separation of concerns
- **Reproducibility**: DVC + MLflow ensure consistent results
- **Production-Ready**: Docker + CI/CD for deployment

For questions or improvements, refer to the main project documentation.

