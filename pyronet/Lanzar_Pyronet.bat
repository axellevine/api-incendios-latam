@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Instalador y Lanzador - Pyronet
color 0B

cd /d "%~dp0"

REM ========================================================
REM CONFIGURACION DE SERVIDORES PYRONET
REM ========================================================
set "CLOUD_API_BASE=https://pyronet-api-ed48bf68.fastapicloud.dev"
set "CLOUD_API_GRID=%CLOUD_API_BASE%/api/incendios-grid"
set "LOCAL_API_GRID=http://127.0.0.1:8000/api/incendios-grid"
set "BASE_RELEASE_URL=https://github.com/axellevine/api-incendios-latam/releases/download/ALLDATA"

echo ========================================================
echo      INSTALADOR Y LANZADOR AUTOMATICO - PYRONET
echo ========================================================
echo.
echo Ruta detectada: %CD%
echo %CD% | findstr /I "OneDrive" >nul
if !errorlevel! equ 0 (
    color 0E
    echo [AVISO] Carpeta dentro de OneDrive detectada. Si tarda mucho, muevela a C:\Pyronet
    color 0B
)
echo.

REM 1. Verificar Python real (evita el acceso directo falso de Microsoft Store)
echo [1/5] Verificando Python en el sistema...
python -c "import sys" >nul 2>&1
if !errorlevel! neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
    )
)

python -c "import sys" >nul 2>&1
if !errorlevel! neq 0 (
    color 0E
    echo [AVISO] Python no esta instalado. Descargando Python 3.11...
    curl.exe -L --fail --progress-bar -o "python_installer.exe" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    if exist "python_installer.exe" (
        echo Instalando Python en segundo plano ^(1 a 2 minutos^)...
        start /wait "" python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0
        del /f /q python_installer.exe
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
        python -c "import sys" >nul 2>&1
        if !errorlevel! neq 0 (
            color 0C
            echo [ERROR] No se pudo configurar Python automaticamente.
            pause
            exit /b 1
        )
        echo [OK] Python instalado correctamente.
        color 0B
    ) else (
        color 0C
        echo [ERROR] Fallo la descarga de Python. Revisa tu conexion a internet.
        pause
        exit /b 1
    )
) else (
    echo [OK] Python detectado correctamente.
)
echo.

REM 2. Verificar integridad del entorno virtual
echo [2/5] Configurando entorno virtual ^(venv^)...
if exist "venv" (
    venv\Scripts\python.exe -m pip --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [AVISO] Se detecto un venv danado o copiado de otra PC. Reconstruyendo...
        rmdir /s /q "venv"
    )
)

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual limpio...
    python -m venv venv
)
echo [OK] Entorno virtual listo.
echo.

REM 3. Verificar librerias con timeout extendido y reintentos automaticos
echo [3/5] Verificando librerias de trabajo...
venv\Scripts\python.exe -c "import streamlit, pydeck, pandas, numpy, requests, altair, plotly, openpyxl" >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Librerias ya instaladas.
) else (
    echo Instalando librerias necesarias ^(solo ocurre la primera vez^)...
    venv\Scripts\python.exe -m pip install --default-timeout=120 --retries 10 --prefer-binary --disable-pip-version-check streamlit pydeck pandas numpy requests altair plotly openpyxl
    if !errorlevel! neq 0 (
        echo [AVISO] Corte de red detectado. Reintentando descarga...
        venv\Scripts\python.exe -m pip install --default-timeout=180 --retries 15 --prefer-binary --disable-pip-version-check streamlit pydeck pandas numpy requests altair plotly openpyxl
    )
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] No se pudieron descargar las librerias por inestabilidad en la conexion a internet.
        echo Verifica tu Wi-Fi y vuelve a ejecutar este archivo para reanudar la descarga.
        pause
        exit /b 1
    )
    echo [OK] Librerias instaladas correctamente.
)
echo.

REM 4. Comprobar estado de pyronet-api en FastAPI Cloud
echo [4/5] Conectando con Pyronet API ^(FastAPI Cloud^)...
set "MODO_NUBE=0"
curl.exe -s --fail --max-time 10 "%CLOUD_API_BASE%/health" >nul 2>&1
if !errorlevel! equ 0 (
    set "MODO_NUBE=1"
    set "INCENDIOS_API_URL=%CLOUD_API_GRID%"
    echo [OK] Pyronet API en linea: %CLOUD_API_GRID%
) else (
    color 0E
    echo [AVISO] No se pudo contactar FastAPI Cloud. Activando respaldo local...
    color 0B
)
echo.

REM 5. Iniciar backend local (solo si FastAPI Cloud no responde) y lanzar Streamlit
echo [5/5] Iniciando interfaz de Pyronet...
if not exist "app48.py" (
    color 0C
    echo [ERROR] Falta el archivo app48.py en %CD%.
    pause
    exit /b 1
)

if "!MODO_NUBE!"=="0" (
    set "BACKEND_DIR="
    if exist "main.py" set "BACKEND_DIR=%~dp0"
    if exist "server\main.py" set "BACKEND_DIR=%~dp0server"

    if "!BACKEND_DIR!"=="" (
        color 0C
        echo [ERROR] No hay conexion con FastAPI Cloud y tampoco existe main.py ni server\main.py local.
        pause
        exit /b 1
    )

    venv\Scripts\python.exe -c "import fastapi, uvicorn, pyarrow" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Instalando dependencias del servidor local...
        venv\Scripts\python.exe -m pip install --default-timeout=120 --retries 10 --prefer-binary --disable-pip-version-check fastapi uvicorn pyarrow
    )

    if not exist "!BACKEND_DIR!\Cachecountryperyear" mkdir "!BACKEND_DIR!\Cachecountryperyear"
    for %%Y in (2018 2019 2020 2021 2022 2023 2024 2025) do (
        set "PARQUET_FILE=!BACKEND_DIR!\Cachecountryperyear\cache_%%Y_diario.parquet"
        set "PKL_FILE=!BACKEND_DIR!\Cachecountryperyear\cache_%%Y.pkl.gz"
        if not exist "!PARQUET_FILE!" if not exist "!PKL_FILE!" (
            echo Descargando respaldo cache_%%Y.pkl.gz...
            curl.exe -L --fail --progress-bar -o "!PKL_FILE!" "%BASE_RELEASE_URL%/cache_%%Y.pkl.gz"
        )
    )

    echo Arrancando Backend local en http://127.0.0.1:8000 ...
    start "Pyronet Backend Local" cmd /k "cd /d "!BACKEND_DIR!" && "%~dp0venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000"
    timeout /t 5 /nobreak >nul
    set "INCENDIOS_API_URL=%LOCAL_API_GRID%"
)

echo.
echo ========================================================
echo  API Activa : !INCENDIOS_API_URL!
echo  Frontend   : http://localhost:8501
echo ========================================================
echo.

venv\Scripts\python.exe -m streamlit run app48.py --server.port 8501

pause