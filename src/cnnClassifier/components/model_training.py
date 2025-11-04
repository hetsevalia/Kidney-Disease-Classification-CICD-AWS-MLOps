import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from pathlib import Path
import mlflow
import mlflow.pytorch
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        # Load the updated base model
        model = nn.Sequential()
        
        # Load VGG16 features
        try:
            # Try new API (torchvision >= 0.13)
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except (TypeError, AttributeError):
            # Fall back to old API (torchvision < 0.13)
            vgg16 = models.vgg16(pretrained=True)
        features = nn.Sequential(*list(vgg16.features.children()))
        
        # Load classifier
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.config.params_classes)
        )
        
        self.model = nn.Sequential(features, classifier)
        
        # Load state dict if it exists
        if os.path.exists(self.config.updated_base_model_path):
            self.model.load_state_dict(torch.load(self.config.updated_base_model_path, weights_only=False))

    def train_valid_generator(self):
        # Define transforms
        if self.config.params_is_augmentation:
            # More aggressive augmentation to reduce overfitting
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(45),  # Increased rotation
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),  # Added vertical flip
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.RandomAffine(degrees=0, scale=(0.7, 1.3)),  # More scale variation
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Added color jitter
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        full_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=train_transform
        )
        
        # Split dataset - Use 70/30 split to reduce overfitting (more validation data)
        train_size = int(0.7 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Update val dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        # Set num_workers=0 on Windows to avoid multiprocessing issues
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        self.valid_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)


    
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        self.model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate, momentum=0.9)
        
        # Setup learning rate scheduler
        scheduler = None
        if self.config.params_learning_rate_scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.config.params_learning_rate_scheduler_factor,
                patience=self.config.params_learning_rate_scheduler_patience
            )
        elif self.config.params_learning_rate_scheduler == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=self.config.params_learning_rate_scheduler_factor
            )
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = self.config.trained_model_path.parent / "best_model.pth"
        
        # MLflow tracking
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        # Training history for overfitting detection
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "epochs": self.config.params_epochs,
                "batch_size": self.config.params_batch_size,
                "learning_rate": self.config.params_learning_rate,
                "augmentation": self.config.params_is_augmentation,
                "early_stopping_patience": self.config.params_early_stopping_patience,
                "lr_scheduler": self.config.params_learning_rate_scheduler,
                "model": "VGG16"
            })
            
            for epoch in range(self.config.params_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                    
                    if batch_idx % 10 == 0:
                        logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in self.valid_loader:
                        data, target = data.to(device), target.to(device)
                        output = self.model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                # Calculate metrics
                train_loss_avg = train_loss / len(self.train_loader)
                val_loss_avg = val_loss / len(self.valid_loader)
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total
                
                # Store history
                train_losses.append(train_loss_avg)
                val_losses.append(val_loss_avg)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                # Log to MLflow (per epoch)
                mlflow.log_metrics({
                    "train_loss": train_loss_avg,
                    "val_loss": val_loss_avg,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                logger.info(f'Epoch {epoch}/{self.config.params_epochs}: '
                          f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Learning rate scheduler step
                if scheduler:
                    old_lr = optimizer.param_groups[0]['lr']
                    if self.config.params_learning_rate_scheduler == "ReduceLROnPlateau":
                        scheduler.step(val_loss_avg)
                    else:
                        scheduler.step()
                    new_lr = optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                
                # Early stopping and model checkpointing
                if val_loss_avg < best_val_loss - self.config.params_early_stopping_min_delta:
                    best_val_loss = val_loss_avg
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    self.save_model(path=best_model_path, model=self.model)
                    logger.info(f"[OK] New best model saved! Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
                else:
                    patience_counter += 1
                    logger.info(f"Early stopping patience: {patience_counter}/{self.config.params_early_stopping_patience}")
                
                # Check for overfitting
                if epoch > 5 and len(train_losses) > 5:
                    train_val_gap = train_loss_avg - val_loss_avg
                    if train_val_gap < -0.1:  # Train loss significantly higher than val (overfitting)
                        logger.warning(f"[WARNING] Potential overfitting detected! Train-Val gap: {train_val_gap:.4f}")
                    elif train_val_gap > 0.5:  # Val loss significantly higher (underfitting)
                        logger.warning(f"[WARNING] Potential underfitting detected! Train-Val gap: {train_val_gap:.4f}")
                
                # Early stopping
                if patience_counter >= self.config.params_early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%")
                    break
            
            # Load best model
            if best_model_path.exists():
                self.model.load_state_dict(torch.load(best_model_path, weights_only=False))
                logger.info(f"Loaded best model with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%")
            
            # Log final metrics
            mlflow.log_metrics({
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
                "final_train_loss": train_losses[-1] if train_losses else 0,
                "final_val_loss": val_losses[-1] if val_losses else 0,
                "final_train_accuracy": train_accs[-1] if train_accs else 0,
                "final_val_accuracy": val_accs[-1] if val_accs else 0,
                "epochs_trained": epoch + 1
            })
            
            # Log model (with error handling for distutils compatibility issues)
            try:
                mlflow.pytorch.log_model(self.model, "model")
                logger.info("[OK] Model logged to MLflow successfully")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to log model to MLflow: {str(e)}")
                logger.info("Model saved locally at: " + str(self.config.trained_model_path))
            
            # Log overfitting analysis
            if len(train_losses) > 5:
                final_overfitting_score = train_losses[-1] - val_losses[-1]
                mlflow.log_metric("overfitting_score", final_overfitting_score)
                logger.info(f"Overfitting Analysis - Final Train-Val Loss Gap: {final_overfitting_score:.4f}")
                if final_overfitting_score < -0.1:
                    logger.warning("Model shows signs of overfitting (train loss > val loss)")
                elif final_overfitting_score > 0.5:
                    logger.warning("Model shows signs of underfitting (val loss >> train loss)")
                else:
                    logger.info("[OK] Model shows good generalization (train and val losses are close)")

        # Save final model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

