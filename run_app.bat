@echo off
REM Launch CS2-Predictor GUI on Windows — opens browser automatically
cd /d "%~dp0"
echo Installing/updating requirements...
pip install -q -r requirements.txt
echo.
echo Starting Streamlit server...
echo If browser doesn't open automatically, go to: http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo.

REM Open browser after 3-second delay (server needs time to boot)
start "" /B cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

REM Run streamlit (this blocks)
streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
pause
