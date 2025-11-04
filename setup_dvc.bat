@echo off
REM DVC Setup Script for Windows - Kidney Disease Classification MLOps Project

echo ==========================================
echo DVC Setup for MLOps Project
echo ==========================================

REM Check if DVC is installed
python -c "import dvc" 2>nul
if errorlevel 1 (
    echo [91mDVC is not installed. Installing...[0m
    pip install dvc dvc-s3
) else (
    echo [92mDVC is already installed[0m
)

REM Initialize DVC
if not exist ".dvc" (
    echo.
    echo [94mInitializing DVC...[0m
    dvc init
    echo [92mDVC initialized[0m
) else (
    echo [92mDVC is already initialized[0m
)

echo.
echo [94mNext Steps:[0m
echo 1. Configure S3 remote (if using S3):
echo    dvc remote add -d myremote s3://your-bucket-name/dvc-storage
echo.
echo 2. Or configure DagsHub remote (if using DagsHub):
echo    dvc remote add -d origin https://dagshub.com/username/repo.dvc
echo.
echo 3. Add large files to DVC:
echo    dvc add artifacts/data_ingestion/data.zip
echo    dvc add artifacts/training/model.pth
echo.
echo 4. Commit DVC files to Git:
echo    git add .dvc/ *.dvc .gitignore
echo    git commit -m "Initialize DVC"
echo.
echo 5. Push to remote:
echo    dvc push    # Push large files to S3
echo    git push    # Push code to GitHub
echo.
echo ==========================================
echo Setup complete! See DVC_MLFLOW_WORKFLOW.md for details
echo ==========================================
pause

