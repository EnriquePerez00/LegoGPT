import os
import requests
import json

REBRICKABLE_API_KEY = os.getenv("REBRICKABLE_API_KEY", "")

def fetch_vehicle_mocs(api_key: str, min_parts: int = 75, max_parts: int = 200) -> list:
    """
    Queries Rebrickable API for vehicle MOCs with a part count between min_parts and max_parts.
    """
    headers = {"Authorization": f"key {api_key}"}
    
    # Theme ID 318 is "Creator", 365 is "Technic", 3 is "Star Wars" (heavy on vehicle MOCs)
    # We query for MOCs with the 'vehicle' tag or similar
    url = "https://rebrickable.com/api/v3/lego/mocs/"
    params = {
        "min_parts": min_parts,
        "max_parts": max_parts,
        "page_size": 20
    }
    
    print(f"Querying Rebrickable MOCs: {url}")
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data.get("results", [])
        else:
            print(f"API Error: Status {res.status_code} - {res.text}")
            return []
    except Exception as e:
        print(f"Connection error to Rebrickable: {e}")
        return []

def download_moc_file(api_key: str, moc_id: str, output_dir: str = "data/mocs") -> str:
    """
    Downloads the 3D model file associated with a MOC.
    """
    os.makedirs(output_dir, exist_ok=True)
    headers = {"Authorization": f"key {api_key}"}
    url = f"https://rebrickable.com/api/v3/lego/mocs/{moc_id}/"
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            moc_data = res.json()
            # If the user uploaded an LDraw or Studio file as instructions
            # It will be listed under the files or download URLs (if public)
            # Rebrickable also provides part inventory:
            inv_url = f"https://rebrickable.com/api/v3/lego/mocs/{moc_id}/parts/"
            part_res = requests.get(inv_url, headers=headers, timeout=10)
            if part_res.status_code == 200:
                parts_list = part_res.json().get("results", [])
                output_path = os.path.join(output_dir, f"{moc_id}_parts.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(parts_list, f, indent=2)
                print(f"Saved MOC {moc_id} inventory to {output_path}")
                return output_path
        return ""
    except Exception as e:
        print(f"Error querying MOC {moc_id}: {e}")
        return ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebrickable Scraper")
    parser.add_argument("--api_key", type=str, default=REBRICKABLE_API_KEY, help="Rebrickable API Key")
    args = parser.parse_args()
    
    if args.api_key:
        mocs = fetch_vehicle_mocs(args.api_key)
        print(f"Found {len(mocs)} MOCs.")
    else:
        print("Please provide a Rebrickable API Key via --api_key or REBRICKABLE_API_KEY env var.")
