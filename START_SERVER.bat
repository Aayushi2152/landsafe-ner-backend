@echo off
cd /d "%~dp0"
echo Installing dependencies...
python -m pip install -r requirements.txt

if not exist "model\risk_classifier.pkl" (
    echo.
    echo No trained model found - training it now, this takes ~10 seconds...
    python train_model.py
)

echo.
echo Starting LandSafe NER backend on http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
