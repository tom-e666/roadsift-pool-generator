@echo off
setlocal
cd /d "%~dp0"

set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
  echo Conda executable not found at "%CONDA_EXE%".
  echo Edit run.bat to point to your Conda installation.
  pause
  exit /b 1
)

if not exist ".conda-env\python.exe" (
  "%CONDA_EXE%" env create --yes --prefix "%CD%\.conda-env" --file environment.yml
  if errorlevel 1 (
    echo Conda could not create the tool environment.
    pause
    exit /b 1
  )
)

"%CONDA_EXE%" run --no-capture-output --prefix "%CD%\.conda-env" streamlit run streamlit_app.py
