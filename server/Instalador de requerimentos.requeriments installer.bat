@echo off
title Instalador Proyecto Incendios LATAM
color 0B

echo ========================================================
echo      INSTALADOR AUTOMATICO - BACKEND Y FRONTEND
echo ========================================================
echo.

:: 1. Revisar si Python esta instalado
echo [1/4] Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR CRITICO] Python no esta instalado o no esta en el PATH de Windows.
    echo 1. Descarga Python desde: https://www.python.org/downloads/
    echo 2. Al instalar, MARCA LA CASILLA "Add Python.exe to PATH" abajo.
    echo 3. Reinicia tu computadora y vuelve a ejecutar este archivo.
    pause
    exit /b
)
echo [OK] Python detectado correctamente.
echo.

:: 2. Crear y activar Entorno Virtual
echo [2/4] Verificando Entorno Virtual (venv)...
if not exist "venv\Scripts\activate" (
    echo Creando un entorno virtual nuevo para aislar las librerias...
    python -m venv venv
) else (
    echo [OK] Entorno virtual ya existe. No se creara de nuevo.
)

echo Activando entorno virtual...
call venv\Scripts\activate
echo.

:: 3. Actualizar PIP
echo [3/4] Actualizando instalador de paquetes (pip)...
python -m pip install --upgrade pip >nul 2>&1
echo.

:: 4. Instalar todas las librerias
echo [4/4] Instalando modulos de Backend y Frontend...
echo (Si ya estan instalados, dira "Requirement already satisfied" y no los bajara 2 veces)
echo.

:: Dependencias Backend
pip install fastapi uvicorn pandas numpy openpyxl

:: Dependencias Frontend
pip install streamlit pydeck requests plotly

echo.
echo ========================================================
echo   ¡INSTALACION COMPLETADA CON EXITO!
echo ========================================================
echo.
echo Para encender tu servidor BACKEND, abre una terminal y usa:
echo   venv\Scripts\python.exe main.py
echo.
echo Para encender tu mapa FRONTEND, abre otra terminal y usa:
echo   venv\Scripts\streamlit.exe run app21.py
echo.
pause