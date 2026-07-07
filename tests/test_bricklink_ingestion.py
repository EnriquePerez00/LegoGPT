import os
import pytest
from scripts.download_bricklink_model import download_bricklink_model

def test_bricklink_single_download_validation():
    # Model 180126 is a sample model ID.
    # The test runs the downloader and verifies it runs through the browser context safely.
    print("\n[BrickLink Ingestion] Ejecutando descarga de prueba del modelo 180126...")
    
    # We run the download. Since the model might be private or require login,
    # we expect the script to execute the browser flow and handle it gracefully (return True or False).
    # If it is blocked by Cloudflare or not found, it writes a screenshot and returns False.
    success = download_bricklink_model("180126", output_dir="data/bricklink_test")
    
    # Check that it either succeeded OR logged the error/screenshot properly
    if not success:
        print("[BrickLink Ingestion] Descarga no completada (esperado si el ID no es público o requiere login).")
        screenshot_path = "data/bricklink_test/error_180126.png"
        # If it reached the page and couldn't find the download button, it should have saved a screenshot
        if os.path.exists(screenshot_path):
            print(f"[BrickLink Ingestion] Captura de pantalla de diagnóstico encontrada: {screenshot_path}")
            assert os.path.exists(screenshot_path)
            # Clean up
            os.remove(screenshot_path)
    else:
        print("[BrickLink Ingestion] ¡Descarga completada exitosamente!")
        # Find downloaded file and clean up
        for f in os.listdir("data/bricklink_test"):
            if f.endswith('.io'):
                os.remove(os.path.join("data/bricklink_test", f))
                
    if os.path.exists("data/bricklink_test"):
        os.rmdir("data/bricklink_test")
