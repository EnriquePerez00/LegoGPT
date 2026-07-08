import os
import re
import requests
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://library.ldraw.org"

# We cache the indexed sets: { "31054": "https://library.ldraw.org/omr/sets/..." }
_sets_index: Dict[str, str] = {}

def index_single_page(page: int) -> List[tuple[str, str]]:
    """
    Indexes a single page of sets and returns list of (set_number, set_page_url)
    """
    url = f"{BASE_URL}/omr/sets?page={page}"
    results = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            content = response.text
            # Filament table links look like:
            # <a href="https://library.ldraw.org/omr/sets/657" class="fi-ta-col"...
            # Followed by text content inside table cells.
            # To be robust, let's extract all hrefs to "/omr/sets/[0-9]+"
            links = re.findall(r'href="(https://library\.ldraw\.org/omr/sets/([0-9]+))"', content)
            links = list(set(links)) # Deduplicate URL and ID tuples
            
            # Let's match all rows or table cells to associate set numbers
            # A set number in LDraw OMR is typically like "31054-1", "8094-1", etc.
            # Let's find matches in the HTML:
            # We can search for the set number string (digits followed by dash and digits)
            # inside the page, and find which set URL it corresponds to.
            # A simple way: find all set links and extract the HTML blocks between them.
            # Let's write a regex that matches the set numbers and names:
            # LDraw set number is usually in the format: \b[0-9]{3,6}-[0-9]\b
            for href, set_id in links:
                # Let's grab the set number by fetching or by a closer regex match in content.
                # In Filament table, the set number is usually in a text column of that row.
                # Let's do a search in the text block around the href.
                pos = content.find(href)
                if pos != -1:
                    # Look at the next 1000 characters
                    sub = content[pos:pos+1500]
                    # Find set numbers like 31054-1 or 31054
                    set_num_match = re.search(r'\b([0-9]{3,6}-[0-9]+)\b', sub)
                    if set_num_match:
                        set_num = set_num_match.group(1).split('-')[0] # "31054"
                        results.append((set_num, href))
                        # Also register the full set number with suffix
                        results.append((set_num_match.group(1), href))
                    else:
                        # Fallback: search for any number in the next few characters
                        number_match = re.search(r'>\s*([0-9]{3,6})\s*<', sub)
                        if number_match:
                            results.append((number_match.group(1), href))
    except Exception as e:
        print(f"Error indexando página {page}: {e}")
    return results

def build_sets_index():
    """
    Builds the global index of all OMR sets by scraping pages 1 to 60 in parallel.
    """
    global _sets_index
    if _sets_index:
        return
        
    print("Construyendo el índice de todos los sets OMR en paralelo (páginas 1-60)...")
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(index_single_page, p): p for p in range(1, 61)}
        
        for future in as_completed(futures):
            page_results = future.result()
            for set_num, href in page_results:
                _sets_index[set_num] = href
                
    print(f"Índice construido. Encontrados {len(_sets_index)} sets/referencias únicas en OMR.")

def search_set_metadata(set_number: str) -> Optional[dict]:
    """
    Locates the download URL, source page URL, and image URL for the given set number.
    """
    build_sets_index()
    
    # Try direct match
    set_page_url = _sets_index.get(set_number)
    
    if not set_page_url:
        # Try finding key that starts with set_number (e.g. "31054" matches "31054-1")
        for key, url in _sets_index.items():
            if key.startswith(set_number):
                set_page_url = url
                break
                
    if not set_page_url:
        print(f"Set {set_number} no encontrado en el índice OMR.")
        return None
        
    print(f"Set {set_number} encontrado en: {set_page_url}")
    
    # Get the set page and find the download URL and image preview URL
    try:
        response = requests.get(set_page_url, timeout=10)
        if response.status_code == 200:
            content = response.text
            match = re.search(r'href="(https://library\.ldraw\.org/library/omr/[^"]+\.mpd)"', content)
            img_match = re.search(r'src="(https://library\.ldraw\.org/media/omr_models/[^"]+)"', content)
            
            if match:
                return {
                    "download_url": match.group(1),
                    "image_url": img_match.group(1) if img_match else None,
                    "source_url": set_page_url
                }
    except Exception as e:
        print(f"Error accediendo a la página del set: {e}")
        
    return None

def search_set_download_url(set_number: str) -> Optional[str]:
    """
    Locates the download URL of the .mpd file for the given set number using the index.
    """
    meta = search_set_metadata(set_number)
    return meta["download_url"] if meta else None

def download_set_by_number(set_number: str, output_dir: str = "data/omr_raw") -> Optional[str]:
    """
    Downloads a set by its number directly and processes it through the ingestion pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = search_set_metadata(set_number)
    
    if not metadata or not metadata.get("download_url"):
        print(f"No se pudo encontrar el enlace de descarga para el set {set_number} en el OMR.")
        return None
        
    download_url = metadata["download_url"]
    print(f"Descargando {download_url}...")
    try:
        response = requests.get(download_url, timeout=20)
        if response.status_code == 200:
            filename = os.path.basename(download_url)
            if not filename.endswith('.mpd'):
                filename = f"{set_number}.mpd"
                
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Guardado en {file_path}")
            
            # Trigger ingestion pipeline!
            from src.ingestion_pipeline import process_and_register_downloaded_model
            print(f"[OMR Downloader] Iniciando pipeline de ingesta para {set_number}...")
            process_and_register_downloaded_model(
                file_path=file_path,
                source="OMR",
                source_url=metadata.get("source_url"),
                image_url=metadata.get("image_url")
            )
            
            return file_path
        else:
            print(f"Error descargando (HTTP {response.status_code})")
    except Exception as e:
        print(f"Fallo en la descarga: {e}")
        
    return None

def download_bulk_omr(max_pages: int = 5, output_dir: str = "data/omr_raw") -> List[str]:
    """
    Bulk downloads all sets on the first N pages.
    """
    os.makedirs(output_dir, exist_ok=True)
    downloaded_files = []
    
    print(f"Iniciando descarga masiva desde OMR (primeras {max_pages} páginas)...")
    
    # Get set URLs for pages
    set_links = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/omr/sets?page={page}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                links = re.findall(r'href="(https://library\.ldraw\.org/omr/sets/[0-9]+)"', res.text)
                set_links.extend(links)
        except Exception as e:
            print(f"Error leyendo página {page}: {e}")
            
    set_links = list(set(set_links))
    print(f"Encontrados {len(set_links)} sets únicos para descargar en bulk.")
    
    # Download helper
    def download_link(link):
        try:
            res = requests.get(link, timeout=10)
            if res.status_code == 200:
                mpd_urls = re.findall(r'href="(https://library\.ldraw\.org/library/omr/[^"]+\.mpd)"', res.text)
                local_downloads = []
                for dl_url in mpd_urls:
                    filename = os.path.basename(dl_url)
                    file_path = os.path.join(output_dir, filename)
                    if os.path.exists(file_path):
                        local_downloads.append(file_path)
                        continue
                        
                    dl_res = requests.get(dl_url, timeout=20)
                    if dl_res.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(dl_res.content)
                        local_downloads.append(file_path)
                        print(f"Descargado: {filename}")
                return local_downloads
        except Exception as e:
            print(f"Error descargando set de {link}: {e}")
        return []

    with ThreadPoolExecutor(max_workers=5) as download_executor:
        futures = [download_executor.submit(download_link, link) for link in set_links]
        for future in as_completed(futures):
            downloaded_files.extend(future.result())
            
    return downloaded_files

if __name__ == "__main__":
    # Test downloading set 31054
    download_set_by_number("31054")
