"""
Test script for MLX-VLM with Llama-3.2-11B-Vision-Instruct-4bit on Apple Silicon M4.
"""
import sys
import os

print("🚀 Verificando MLX-VLM...")
try:
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    print("✅ MLX y MLX-VLM cargados correctamente.")
    print("Apple Silicon Metal Device:", mx.default_device())
except Exception as e:
    print(f"❌ Error importando MLX-VLM: {e}")
    sys.exit(1)
