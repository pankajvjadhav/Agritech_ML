@echo off
REM This batch file runs all tests automatically WITHOUT requiring server
REM It activates the virtual environment and runs test scripts in sequence

cd C:\Agritech_ML
call venv\Scripts\activate.bat
echo ========================================
echo Running Dataset Functionality Test
echo ========================================
python test_dataset_functionality.py
echo.
echo ========================================
echo Running test_predict.py (Default - Row 0)
echo ========================================
python test_predict.py --format pretty
echo.
echo ========================================
echo Running test_predict.py (Realtime - Local Mode)
echo ========================================
REM Using local mode (no --http flag) so no server needed
python test_predict.py --realtime --lat 28.6 --lon 77.2 --format pretty
echo.
echo ========================================
echo All Local Tests Complete!
echo ========================================
echo.
echo Note: API endpoint tests require server to be running.
echo To test API endpoints, start server first: python main.py
echo Then run: python test_api_endpoints.py
echo.
pause


