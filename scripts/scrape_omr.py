import os
import requests
import json

def download_omr_set(set_id: str, output_dir: str = "data/omr_raw") -> str:
    """
    Downloads a single LDraw set from the official OMR repository.
    Example: set_id = "75078"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard OMR filename naming format: <set_id>-1.mpd
    filename = f"{set_id}-1.mpd"
    url = f"http://omr.ldraw.org/files/omr/{filename}"
    output_path = os.path.join(output_dir, filename)
    
    print(f"Downloading from OMR: {url}")
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(res.text)
            print(f"Successfully saved to {output_path}")
            return output_path
        else:
            # Try alternate naming: <set_id>.mpd
            filename = f"{set_id}.mpd"
            url = f"http://omr.ldraw.org/files/omr/{filename}"
            output_path = os.path.join(output_dir, filename)
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(res.text)
                print(f"Successfully saved to {output_path}")
                return output_path
            else:
                print(f"Set {set_id} not found in OMR (Status code: {res.status_code})")
                return ""
    except Exception as e:
        print(f"Network error downloading {set_id}: {e}")
        return ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OMR Scraper CLI")
    parser.add_argument("--set_id", type=str, default="75078", help="LEGO set number")
    args = parser.parse_args()
    
    download_omr_set(args.set_id)
