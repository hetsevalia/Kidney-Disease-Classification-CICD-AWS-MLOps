import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from pathlib import Path
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier import logger


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        val_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=val_transform
        )
        
        # Split for evaluation (30% for evaluation)
        eval_size = int(0.3 * len(val_dataset))
        train_size = len(val_dataset) - eval_size
        _, eval_dataset = torch.utils.data.random_split(
            val_dataset, [train_size, eval_size]
        )
        
        # Set num_workers=0 on Windows to avoid multiprocessing issues
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        
        self.valid_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=num_workers
        )


    @staticmethod
    def load_model(path: Path, num_classes: int) -> nn.Module:
        # Recreate model architecture
        try:
            # Try new API (torchvision >= 0.13)
            vgg16 = models.vgg16(weights=None)
        except (TypeError, AttributeError):
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
            nn.Linear(4096, num_classes)
        )
        
        model = nn.Sequential(features, classifier)
        loaded_data = torch.load(path, weights_only=False)
        
        # Handle both state_dict and full model saves
        if isinstance(loaded_data, dict):
            model.load_state_dict(loaded_data)
        elif hasattr(loaded_data, 'state_dict'):
            model.load_state_dict(loaded_data.state_dict())
        else:
            model.load_state_dict(loaded_data)
        
        return model
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model, self.config.params_classes)
        self._valid_generator()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Evaluation using device: {device}")
        self.model.to(device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Store predictions and targets for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate basic metrics
        accuracy = correct / total
        avg_loss = total_loss / len(self.valid_loader)
        
        # Calculate comprehensive metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Store scores
        self.score = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "confusion_matrix": cm.tolist()
        }
        
        # Log detailed metrics
        logger.info("=" * 60)
        logger.info("Model Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy*100:.2f}%")
        logger.info(f"Precision (Weighted): {precision:.4f}")
        logger.info(f"Recall (Weighted): {recall:.4f}")
        logger.info(f"F1-Score (Weighted): {f1:.4f}")
        logger.info(f"\nPer-Class Metrics:")
        class_names = ['Normal', 'Tumor'] if self.config.params_classes == 2 else [f'Class_{i}' for i in range(self.config.params_classes)]
        for i, name in enumerate(class_names):
            logger.info(f"  {name}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info("=" * 60)
        
        # Classification report
        report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
        logger.info(f"\nClassification Report:\n{report}")
        
        self.save_score()

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.score)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # Log all parameters
            if isinstance(self.config.all_params, dict):
                mlflow.log_params(self.config.all_params)
            else:
                # Handle params as object with attributes
                params_dict = {}
                for key in dir(self.config.all_params):
                    if not key.startswith('_'):
                        try:
                            value = getattr(self.config.all_params, key)
                            if not callable(value):
                                params_dict[key] = value
                        except:
                            pass
                mlflow.log_params(params_dict)
            
            # Log all metrics
            mlflow.log_metrics({
                "loss": self.score["loss"],
                "accuracy": self.score["accuracy"],
                "precision": self.score["precision"],
                "recall": self.score["recall"],
                "f1_score": self.score["f1_score"]
            })
            
            # Log per-class metrics
            class_names = ['Normal', 'Tumor'] if self.config.params_classes == 2 else [f'Class_{i}' for i in range(self.config.params_classes)]
            for i, name in enumerate(class_names):
                mlflow.log_metrics({
                    f"{name}_precision": self.score["precision_per_class"][i],
                    f"{name}_recall": self.score["recall_per_class"][i],
                    f"{name}_f1": self.score["f1_per_class"][i]
                })
            
            # Log confusion matrix as artifact
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.score["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            plt.close()
            
            mlflow.log_artifact('confusion_matrix.png')
            
            # Model registry does not work with file store
            try:
                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.pytorch.log_model(self.model, "model")
                logger.info("[OK] Model logged to MLflow successfully")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to log model to MLflow: {str(e)}")
                logger.info("Evaluation metrics logged, but model artifact logging failed")
            
            logger.info("[OK] Metrics logged to MLflow successfully")
