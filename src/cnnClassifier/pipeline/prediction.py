import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
from cnnClassifier import logger


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    
    def predict(self):
        try:
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Recreate model architecture (VGG16) - match training architecture exactly
            # Using weights=None since we're loading from state_dict anyway
            try:
                # Try new API (torchvision >= 0.13)
                vgg16 = models.vgg16(weights=None)
            except TypeError:
                # Fall back to old API (torchvision < 0.13)
                vgg16 = models.vgg16(pretrained=False)
            features = nn.Sequential(*list(vgg16.features.children()))
            
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 2)  # 2 classes: Normal and Tumor
            )
            
            model = nn.Sequential(features, classifier)
            
            # FIXED PATH - Changed from "model/model.pth" to "artifacts/training/model.pth"
            model_path = os.path.join("artifacts", "training", "model.pth")
            
            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. "
                    "Please train the model first by running 'python main.py' "
                    "or clicking the 'Train Model' button."
                )
            
            logger.info(f"Loading model from {model_path}")
            
            # Load state dict with explicit weights_only=False to suppress warning
            # Note: weights_only=False is safe here since we control the model file
            try:
                loaded_data = torch.load(model_path, map_location=device, weights_only=False)
                
                # Handle both state_dict and full model saves
                if isinstance(loaded_data, dict) and all(isinstance(k, str) for k in loaded_data.keys()):
                    # It's a state_dict
                    logger.info("Loading state_dict...")
                    model.load_state_dict(loaded_data)
                elif hasattr(loaded_data, 'state_dict'):
                    # It's a full model object
                    logger.info("Extracting state_dict from model object...")
                    model.load_state_dict(loaded_data.state_dict())
                else:
                    # Try to load as state_dict anyway
                    logger.info("Attempting to load as state_dict...")
                    model.load_state_dict(loaded_data)
                
                model.to(device)
                model.eval()
                logger.info("Model loaded successfully")
            except Exception as load_error:
                logger.error(f"Failed to load model: {str(load_error)}")
                # Try alternative loading method
                logger.info("Attempting alternative loading method...")
                try:
                    # Load on CPU first, then move to device
                    loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    if isinstance(loaded_data, dict):
                        model.load_state_dict(loaded_data)
                    elif hasattr(loaded_data, 'state_dict'):
                        model.load_state_dict(loaded_data.state_dict())
                    else:
                        model.load_state_dict(loaded_data)
                    model.to(device)
                    model.eval()
                    logger.info("Model loaded successfully with alternative method")
                except Exception as alt_error:
                    logger.error(f"Alternative loading also failed: {str(alt_error)}")
                    raise RuntimeError(
                        f"Failed to load model from {model_path}. "
                        f"Error: {str(load_error)}. "
                        f"The model file may be corrupted. Please retrain the model using 'python main.py'."
                    )

            # Load and preprocess image
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"Image file not found: {self.filename}")
            
            logger.info(f"Loading image from {self.filename}")
            image = Image.open(self.filename).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(device)
            logger.info(f"Image preprocessed, shape: {image_tensor.shape}")
            
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

            if predicted_class == 1:
                prediction = 'Tumor'
            else:
                prediction = 'Normal'
            
            result = [{"image": prediction, "confidence": float(confidence)}]
            logger.info(f"Prediction result: {result}")
            return result
            
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError in prediction: {str(e)}")
            raise
        except RuntimeError as e:
            logger.error(f"RuntimeError in prediction: {str(e)}")
            # This might be a model architecture mismatch
            raise RuntimeError(f"Error loading/running model: {str(e)}. The model architecture might not match.")
        except Exception as e:
            logger.exception(f"Unexpected error in prediction: {str(e)}")
            raise