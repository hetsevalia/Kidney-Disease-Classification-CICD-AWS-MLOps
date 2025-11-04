#!/usr/bin/env python3
"""
Verification script for Kidney Disease Classification MLOps Project
Run this to check if everything is set up correctly.
"""

import os
import sys
from pathlib import Path

def print_check(name, condition, message=""):
    """Print check status"""
    status = "✓" if condition else "✗"
    print(f"{status} {name}")
    if message and not condition:
        print(f"  ⚠ {message}")

def main():
    print("=" * 60)
    print("Kidney Disease Classification - Setup Verification")
    print("=" * 60)
    print()
    
    checks_passed = 0
    total_checks = 0
    
    # 1. Check Python version
    print("1. Python Environment:")
    total_checks += 1
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print_check("Python version", True, f"{python_version.major}.{python_version.minor}")
        checks_passed += 1
    else:
        print_check("Python version", False, f"Found {python_version.major}.{python_version.minor}, need >= 3.8")
    
    # 2. Check package installation
    print("\n2. Package Installation:")
    total_checks += 1
    try:
        from cnnClassifier import logger
        print_check("cnnClassifier package", True)
        checks_passed += 1
    except ImportError:
        print_check("cnnClassifier package", False, "Run: pip install -e .")
    
    # 3. Check required packages
    print("\n3. Required Dependencies:")
    required_packages = ['torch', 'mlflow', 'dvc', 'flask', 'pandas', 'numpy']
    for pkg in required_packages:
        total_checks += 1
        try:
            __import__(pkg)
            print_check(pkg, True)
            checks_passed += 1
        except ImportError:
            print_check(pkg, False, f"Install: pip install {pkg}")
    
    # 4. Check configuration files
    print("\n4. Configuration Files:")
    config_files = [
        ("config/config.yaml", "config/config.yaml"),
        ("params.yaml", "params.yaml"),
    ]
    for name, path in config_files:
        total_checks += 1
        exists = Path(path).exists()
        print_check(name, exists, f"Missing: {path}")
        if exists:
            checks_passed += 1
    
    # 5. Check artifacts
    print("\n5. Artifacts & Data:")
    artifacts = [
        ("Data directory", "artifacts/data_ingestion/kidney-ct-scan-image"),
        ("Base model", "artifacts/prepare_base_model/base_model.pth"),
        ("Trained model", "artifacts/training/model.pth"),
    ]
    for name, path in artifacts:
        total_checks += 1
        exists = Path(path).exists()
        if exists:
            size = os.path.getsize(path) if Path(path).is_file() else None
            size_str = f" ({size/1024/1024:.2f} MB)" if size else ""
            print_check(name, True, f"{path}{size_str}")
            checks_passed += 1
        else:
            print_check(name, False, f"Missing: {path} (Run: python main.py)")
    
    # 6. Check DVC
    print("\n6. DVC Configuration:")
    total_checks += 1
    dvc_yaml = Path("dvc.yaml").exists()
    print_check("dvc.yaml", dvc_yaml)
    if dvc_yaml:
        checks_passed += 1
    
    total_checks += 1
    dvc_lock = Path("dvc.lock").exists()
    print_check("dvc.lock", dvc_lock, "Generated after running: dvc repro")
    if dvc_lock:
        checks_passed += 1
    
    # 7. Check logs directory
    print("\n7. Logs Directory:")
    total_checks += 1
    logs_dir = Path("logs")
    if logs_dir.exists():
        print_check("logs/ directory", True)
        checks_passed += 1
    else:
        print_check("logs/ directory", False, "Will be created on first run")
    
    # 8. Check scores
    print("\n8. Evaluation Results:")
    total_checks += 1
    scores_file = Path("scores.json")
    if scores_file.exists():
        print_check("scores.json", True)
        checks_passed += 1
        # Try to read scores
        try:
            import json
            with open(scores_file) as f:
                scores = json.load(f)
            print(f"   📊 Accuracy: {scores.get('accuracy', 'N/A')}")
            print(f"   📉 Loss: {scores.get('loss', 'N/A')}")
        except:
            pass
    else:
        print_check("scores.json", False, "Generated after evaluation stage")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {checks_passed}/{total_checks} checks passed")
    print("=" * 60)
    
    if checks_passed == total_checks:
        print("\n🎉 All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. If model exists: python app.py")
        print("  2. If model missing: python main.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  - Install package: pip install -e .")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Train model: python main.py")
    
    print()

if __name__ == "__main__":
    main()

