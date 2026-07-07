#!/bin/bash

echo "Configurando entorno local para LegoGPT (Apple Silicon)..."

# 1. Crear entorno virtual
python3 -m venv legogpt_env
source legogpt_env/bin/activate

# 2. Actualizar gestores
pip install --upgrade pip setuptools wheel

# 3. Instalar PyTorch optimizado para Apple Silicon
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 4. Instalar PyTorch Geometric y el resto de dependencias
pip install -r prd/requirements.txt

echo "Entorno configurado. Para activar: source legogpt_env/bin/activate"