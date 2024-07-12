@echo off
setlocal

:: Request administrative privileges for setting the PATH
net session >nul 2>nul
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb runAs"
    goto end
)

:: Determine the current script directory
set "SCRIPT_DIR=%~dp0"
set "TOOLS_DIR=%SCRIPT_DIR%Tools"

:: Check if Git is already installed
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo Git is already installed.
) else (
    :: Download Git installer
    echo Downloading Git installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/Git-2.41.0.3-64-bit.exe' -OutFile '%TEMP%\GitInstaller.exe'"

    :: Install Git
    echo Installing Git...
    %TEMP%\GitInstaller.exe /SILENT /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"

    :: Delete the installer
    del %TEMP%\GitInstaller.exe
)

:: Check if FFmpeg is already installed
where ffmpeg >nul 2>nul
if %errorlevel% == 0 (
    echo FFmpeg is already installed.
) else (
    :: Create a directory for FFmpeg
    echo Creating directory for FFmpeg...
    if not exist "%TOOLS_DIR%\ffmpeg" mkdir "%TOOLS_DIR%\ffmpeg"
    cd "%TOOLS_DIR%\ffmpeg"

    :: Download FFmpeg
    echo Downloading FFmpeg...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-6.0-essentials_build.zip' -OutFile 'ffmpeg.zip'"

    :: Unzip FFmpeg
    echo Unzipping FFmpeg...
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath ."

    :: Clean up
    del ffmpeg.zip
    cd "%SCRIPT_DIR%"
    echo FFmpeg installation complete!
)

:: Check if Poppler is already installed
if exist "%TOOLS_DIR%\poppler\Library\bin\pdfinfo.exe" (
    echo Poppler is already installed.
) else (
    echo Downloading Poppler...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/oschwartz10612/poppler-windows/releases/download/v23.01.0-0/Release-23.01.0-0.zip' -OutFile '%TEMP%\Poppler.zip'"

    echo Unzipping Poppler...
    powershell -Command "Expand-Archive -Path '%TEMP%\Poppler.zip' -DestinationPath '%TEMP%\Poppler'"

    echo Moving Poppler to %TOOLS_DIR%\poppler...
    if not exist "%TOOLS_DIR%\poppler" mkdir "%TOOLS_DIR%\poppler"
    for /d %%i in ("%TEMP%\Poppler\poppler-*") do xcopy "%%i\*" "%TOOLS_DIR%\poppler" /E /H /C /I

    :: Clean up
    del %TEMP%\Poppler.zip
    rmdir /S /Q %TEMP%\Poppler
    echo Poppler installation complete!
)

:: Check if Tesseract is already installed
if exist "%TOOLS_DIR%\tesseract\tesseract.exe" (
    echo Tesseract is already installed.
) else (
    :: Ensure no previous Tesseract installer exists
    if exist "%TEMP%\TesseractInstaller.exe" del "%TEMP%\TesseractInstaller.exe"

    echo Downloading Tesseract...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe' -OutFile '%TEMP%\TesseractInstaller.exe'"

    echo Installing Tesseract...
    %TEMP%\TesseractInstaller.exe /SILENT

    echo Moving Tesseract to %TOOLS_DIR%\tesseract...
    if not exist "%TOOLS_DIR%\tesseract" mkdir "%TOOLS_DIR%\tesseract"
    xcopy "%ProgramFiles%\Tesseract-OCR\*" "%TOOLS_DIR%\tesseract" /E /H /C /I

    :: Clean up
    del %TEMP%\TesseractInstaller.exe
    echo Tesseract installation complete!
)

:install_venv
echo Installing virtual environment and dependencies...

:: Enable delayed variable expansion
setlocal enabledelayedexpansion

cd /d "%SCRIPT_DIR%"

:: Create a virtual environment
echo Creating virtual environment...
python -m venv "venv"
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    goto end
)

:: Install project dependencies
echo Installing project dependencies...
"venv\Scripts\python" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    goto end
)

echo Virtual environment setup complete.

echo Press any key to exit...
pause >nul
goto :EOF

:end
echo Script ended.
