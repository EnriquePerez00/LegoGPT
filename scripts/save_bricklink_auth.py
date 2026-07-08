import os
import sys
import time
from playwright.sync_api import sync_playwright

def save_auth():
    os.makedirs("scratch", exist_ok=True)
    auth_path = "scratch/auth_state.json"
    
    print("==================================================================")
    print("INSTRUCCIONES DE INICIO DE SESIÓN:")
    print("1. Se abrirá una ventana de navegador Chromium visible en tu Mac.")
    print("2. Por favor, inicia sesión en tu cuenta de BrickLink.")
    print("3. Una vez iniciada la sesión, la ventana se cerrará sola y guardará las cookies.")
    print("==================================================================")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        page.goto("https://www.bricklink.com/v3/studio/gallery.page")
        
        print("\nEsperando inicio de sesión activo...")
        
        try:
            # Poll every 2 seconds to check if logged in
            for sec in range(150): # 5 minutes timeout
                if page.is_closed():
                    print("\n[Auth] El navegador fue cerrado por el usuario.")
                    break
                
                # We search for "Sign Out" or "Log Out" in the text or profile links
                logged_in = page.evaluate("""
                () => {
                    const text = document.body.innerText || "";
                    const html = document.body.innerHTML || "";
                    
                    // Check if 'Sign Out', 'Log Out' or user menu is rendered
                    return text.includes('Sign Out') || 
                           text.includes('Log Out') || 
                           html.includes('Sign Out') || 
                           html.includes('Log Out') ||
                           document.querySelector('#profile-menu') !== null ||
                           document.querySelector('.header-profile') !== null;
                }
                """)
                
                if logged_in:
                    print("\n[Auth] ¡Sesión de usuario detectada con éxito!")
                    time.sleep(2)
                    context.storage_state(path=auth_path)
                    print(f"[Auth] Estado de autenticación guardado correctamente en: {auth_path}")
                    break
                    
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\nGuardando estado actual antes de salir...")
            context.storage_state(path=auth_path)
            
        browser.close()

if __name__ == "__main__":
    save_auth()
