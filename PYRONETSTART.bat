@echo off
setlocal enabledelayedexpansion
title Instalador y Lanzador - Pyronet
color 0B

REM Cambia el directorio actual a la carpeta exacta donde esta este .bat (soporta espacios en el nombre)
cd /d "%~dp0"

echo ========================================================
echo      INSTALADOR Y LANZADOR AUTOMATICO - PYRONET
echo ========================================================
echo.
echo Ruta detectada: %CD%
echo.

REM 1. Verificar e instalar Python si no existe
echo [1/5] Verificando Python en el sistema...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0E
    echo [AVISO] Python no esta instalado. Descargando Python 3.11...
    curl.exe -L --fail --progress-bar -o "python_installer.exe" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    if exist "python_installer.exe" (
        echo Instalando Python en segundo plano ^(1 a 2 minutos^)...
        start /wait "" python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0
        del /f /q python_installer.exe
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
        python --version >nul 2>&1
        if !errorlevel! neq 0 (
            color 0C
            echo [ERROR] No se pudo configurar Python automaticamente.
            pause
            exit /b
        )
        echo [OK] Python instalado correctamente.
        color 0B
    ) else (
        color 0C
        echo [ERROR] Fallo la descarga de Python. Revisa tu conexion a internet.
        pause
        exit /b
    )
) else (
    echo [OK] Python ya esta instalado.
)
echo.

REM 2. Crear y activar entorno virtual
echo [2/5] Configurando entorno virtual ^(venv^)...
if not exist "venv\Scripts\activate.bat" (
    python -m venv venv
)
call "venv\Scripts\activate.bat"
echo [OK] Entorno virtual activo.
echo.

REM 3. Instalar librerias necesarias
echo [3/5] Instalando librerias de trabajo...
python -m pip install --upgrade pip >nul 2>&1
pip install "fastapi[standard]" uvicorn pandas numpy openpyxl streamlit pydeck requests altair plotly
echo.

REM 4. Descargar los 8 archivos de cache desde GitHub
echo [4/5] Verificando datos en Cachecountryperyear...
if not exist "Cachecountryperyear" mkdir "Cachecountryperyear"

set "BASE_RELEASE_URL=https://github.com/axellevine/api-incendios-latam/releases/download/ALLDATA"

for %%Y in (2018 2019 2020 2021 2022 2023 2024 2025) do (
    set "ARCHIVO=Cachecountryperyear\cache_%%Y.pkl.gz"
    if exist "!ARCHIVO!" (
        for %%F in ("!ARCHIVO!") do if %%~zF equ 0 del /f /q "!ARCHIVO!"
    )
    if not exist "!ARCHIVO!" (
        echo Descargando cache_%%Y.pkl.gz...
        curl.exe -L --fail --progress-bar -o "!ARCHIVO!" "!BASE_RELEASE_URL!/cache_%%Y.pkl.gz"
        if !errorlevel! neq 0 (
            color 0C
            echo [ERROR] No se pudo descargar cache_%%Y.pkl.gz.
            echo Verifica que el Release ALLDATA en GitHub este PUBLICADO.
            if exist "!ARCHIVO!" del /f /q "!ARCHIVO!"
            pause
            exit /b
        )
    ) else (
        echo [OK] cache_%%Y.pkl.gz ya existe localmente.
    )
)
echo.

REM 5. Verificar archivos principales y arrancar
echo [5/5] Iniciando servidores locales...
if not exist "main.py" (
    color 0C
    echo [ERROR] Falta el archivo main.py en esta carpeta.
    pause
    exit /b
)
if not exist "app48.py" (
    color 0C
    echo [ERROR] Falta el archivo app48.py en esta carpeta.
    pause
    exit /b
)

echo Arrancando Backend ^(main.py^) en http://127.0.0.1:8000 ...
start "Pyronet Backend" cmd /k "venv\Scripts\activate.bat && echo n | uvicorn main:app --host 127.0.0.1 --port 8000"

echo Esperando 8 segundos a que carguen los datos en RAM...
timeout /t 8 /nobreak >nul

echo Abriendo Mapa ^(app48.py^) en el navegador...
set "INCENDIOS_API_URL=http://127.0.0.1:8000/api/incendios-grid"
"venv\Scripts\streamlit.exe" run app48.py

pause