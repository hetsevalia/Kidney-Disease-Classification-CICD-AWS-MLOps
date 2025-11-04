#!/bin/bash
# DVC Setup Script for Kidney Disease Classification MLOps Project

echo "=========================================="
echo "DVC Setup for MLOps Project"
echo "=========================================="

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC is not installed. Installing..."
    pip install dvc dvc-s3
else
    echo "✅ DVC is already installed"
fi

# Initialize DVC
if [ ! -d ".dvc" ]; then
    echo ""
    echo "📦 Initializing DVC..."
    dvc init
    echo "✅ DVC initialized"
else
    echo "✅ DVC is already initialized"
fi

echo ""
echo "📝 Next Steps:"
echo "1. Configure S3 remote (if using S3):"
echo "   dvc remote add -d myremote s3://your-bucket-name/dvc-storage"
echo ""
echo "2. Or configure DagsHub remote (if using DagsHub):"
echo "   dvc remote add -d origin https://dagshub.com/username/repo.dvc"
echo ""
echo "3. Add large files to DVC:"
echo "   dvc add artifacts/data_ingestion/data.zip"
echo "   dvc add artifacts/training/model.pth"
echo ""
echo "4. Commit DVC files to Git:"
echo "   git add .dvc/ *.dvc .gitignore"
echo "   git commit -m 'Initialize DVC'"
echo ""
echo "5. Push to remote:"
echo "   dvc push    # Push large files to S3"
echo "   git push    # Push code to GitHub"
echo ""
echo "=========================================="
echo "Setup complete! See DVC_MLFLOW_WORKFLOW.md for details"
echo "=========================================="

