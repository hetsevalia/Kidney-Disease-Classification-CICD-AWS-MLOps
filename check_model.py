#!/usr/bin/env python3
"""
Script to check and diagnose model file issues
"""
import os
import torch
from pathlib import Path

def check_model_file(model_path):
    """Check if model file is valid"""
    print("=" * 60)
    print("Model File Diagnostic Tool")
    print("=" * 60)
    print()
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✓ File exists: {model_path}")
    print(f"✓ File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    # Check if it's a zip file
    try:
        import zipfile
        if zipfile.is_zipfile(model_path):
            print("✓ File is a valid ZIP archive")
        else:
            print("⚠ WARNING: File is not a valid ZIP archive")
    except Exception as e:
        print(f"⚠ Could not verify ZIP format: {e}")
    
    # Try to load the model
    print("\nAttempting to load model...")
    try:
        # Try loading on CPU first
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✓ Model loaded successfully!")
        
        # Check what type of data was loaded
        if isinstance(loaded_data, dict):
            print(f"✓ Loaded data type: dict (state_dict)")
            print(f"✓ Number of keys: {len(loaded_data)}")
            if len(loaded_data) > 0:
                print(f"✓ Sample keys: {list(loaded_data.keys())[:5]}")
                # Check sizes of first few tensors
                for i, (key, value) in enumerate(list(loaded_data.items())[:3]):
                    if hasattr(value, 'shape'):
                        print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
        elif hasattr(loaded_data, 'state_dict'):
            print(f"✓ Loaded data type: Model object (has state_dict method)")
        else:
            print(f"⚠ Loaded data type: {type(loaded_data)}")
        
        # Try to load into a model architecture
        print("\nAttempting to load into model architecture...")
        try:
            import torch.nn as nn
            import torchvision.models as models
            
            # Recreate model architecture
            try:
                vgg16 = models.vgg16(weights=None)
            except (TypeError, AttributeError):
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
                nn.Linear(4096, 2)
            )
            model = nn.Sequential(features, classifier)
            
            # Try to load state dict
            if isinstance(loaded_data, dict):
                model.load_state_dict(loaded_data)
            elif hasattr(loaded_data, 'state_dict'):
                model.load_state_dict(loaded_data.state_dict())
            else:
                model.load_state_dict(loaded_data)
            
            print("✓ Successfully loaded into model architecture!")
            print("✓ Model is ready to use!")
            return True
            
        except Exception as arch_error:
            print(f"❌ ERROR: Failed to load into model architecture")
            print(f"   Error: {str(arch_error)}")
            return False
        
    except RuntimeError as e:
        error_msg = str(e)
        if "corrupted" in error_msg.lower() or "invalid header" in error_msg.lower():
            print(f"❌ ERROR: Model file appears to be CORRUPTED")
            print(f"   Error: {error_msg}")
            print("\n💡 SOLUTION: The model file is corrupted. You need to retrain:")
            print("   1. Delete the corrupted model: rm artifacts/training/model.pth")
            print("   2. Retrain the model: python main.py")
            return False
        else:
            print(f"❌ ERROR: Failed to load model")
            print(f"   Error: {error_msg}")
            return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return False
    
    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    model_path = "artifacts/training/model.pth"
    
    if check_model_file(model_path):
        print("\n✅ Model file is valid and ready to use!")
    else:
        print("\n❌ Model file has issues. Please retrain the model.")
        print("\nTo retrain:")
        print("  python main.py")

