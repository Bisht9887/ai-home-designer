"""
⚠️ DEPRECATED - Use generate_interiors.py instead

This script is kept for backward compatibility.
The new unified script (generate_interiors.py) supports multiple datasets
and has better configuration options.

---

Generate 2D Interior Plans from Pre-configured Variations

For each variation folder:
- Reads the PNG file (any name - only one PNG per folder)
- Reads configuration.json (design parameters)
- Uses global master_prompt_2D-only.json (from parent directory)

Generates:
- 2D_plan.png (furnished floor plan)
"""

import requests
import base64
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Paths - relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "Data"

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# API Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("API_URL", "https://openrouter.ai/api/v1/chat/completions")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Data paths
SOURCE_DIR = DATA_DIR / "D01_120_variations"
MASTER_PROMPT_PATH = SCRIPT_DIR / "master_prompt_2D-only.json"


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config_into_prompt(config, prompt_template):
    """Replace placeholders in prompt with config values"""
    prompt_str = json.dumps(prompt_template)
    
    # Build replacements from config
    replacements = {
        "{{rooms_count}}": str(config.get('rooms_count', 1)),
        "{{household}}": config.get('household', ''),
        "{{persona_lifestyle}}": config.get('persona_lifestyle', ''),
        "{{hobby_tags_json_array}}": json.dumps(config.get('hobby_tags_json_array', [])),
        "{{personality_social}}": config.get('personality_social', ''),
        "{{routine}}": config.get('routine', ''),
        "{{needs_json_array}}": json.dumps(config.get('needs_json_array', [])),
        "{{desires_json_array}}": json.dumps(config.get('desires_json_array', [])),
        "{{budget}}": config.get('budget', ''),
        "{{storage_intensity}}": config.get('storage_intensity', ''),
        "{{cabinet_target_guide}}": json.dumps(config.get('cabinet_target_guide', {})),
        "{{style_id}}": str(config.get('style_id', '')),
        "{{interior_style}}": config.get('interior_style', ''),
        "{{palette}}": config.get('palette', ''),
        "{{materials}}": json.dumps(config.get('materials', []))
    }
    
    for key, value in replacements.items():
        prompt_str = prompt_str.replace(key, value)
    
    return prompt_str


def call_api(png_path, prompt_text):
    """Call the API with image and prompt"""
    with open(png_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "google/gemini-3-pro-image-preview",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                },
                {
                    "type": "text",
                    "text": f"Follow these instructions exactly:\n\n{prompt_text}"
                }
            ]
        }],
        "modalities": ["image", "text"],
        "max_tokens": 8192
    }
    
    response = requests.post(API_URL, headers=headers, json=data, timeout=120)
    return response.json()


def extract_images(result):
    """Extract base64 images from API response"""
    images = []
    
    if "error" in result:
        return images
    
    message = result.get("choices", [{}])[0].get("message", {})
    
    for img in message.get("images", []):
        if isinstance(img, dict):
            image_url = img.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            elif isinstance(image_url, str):
                url = image_url
            else:
                continue
            
            if url.startswith("data:image") and "," in url:
                images.append(url.split(",", 1)[1])
    
    return images


def find_floor_plan_png(folder_path):
    """Find the single PNG file in the folder (any name)"""
    # Find any PNG that's not an output file
    for file in folder_path.glob("*.png"):
        # Skip output files we may have generated
        if not file.name.startswith(("2D_plan", "output_", "semi3D", "3D_")):
            return file
    return None


def process_variation(folder_path):
    """Process a single variation folder"""
    folder_name = folder_path.name
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")
    
    # Find floor plan PNG
    png_path = find_floor_plan_png(folder_path)
    config_path = folder_path / "configuration.json"
    
    if png_path is None:
        print(f"  [SKIP] No floor plan PNG found")
        return False
    
    if not config_path.exists():
        print(f"  [SKIP] No configuration.json found")
        return False
    
    # Check if already processed
    output_path = folder_path / "2D_plan.png"
    if output_path.exists():
        print(f"  [SKIP] Already processed (2D_plan.png exists)")
        return True
    
    # Load configuration
    print(f"  1. Loading configuration...")
    config = load_json(config_path)
    print(f"     Style: {config.get('interior_style', 'Unknown')}")
    print(f"     PNG: {png_path.name}")
    
    # Load global master prompt and merge with config
    print(f"  2. Merging prompt with config...")
    master_prompt = load_json(MASTER_PROMPT_PATH)
    prompt_text = merge_config_into_prompt(config, master_prompt)
    
    # Call API
    print(f"  3. Calling API...")
    result = call_api(png_path, prompt_text)
    
    if "error" in result:
        print(f"     ERROR: {result['error']}")
        return False
    
    # Extract and save images
    images = extract_images(result)
    print(f"     Received {len(images)} image(s)")
    
    if len(images) == 0:
        print(f"     ERROR: No images in response")
        # Show more details about what went wrong
        if "error" in result:
            print(f"     API Error: {result['error']}")
        else:
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            if content:
                print(f"     Model response: {content[:200]}...")
            else:
                print(f"     Response keys: {list(result.keys())}")
        return False
    
    # Save only the first image as 2D_plan.png (ignore any extra images)
    img_bytes = base64.b64decode(images[0])
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"     Saved: 2D_plan.png ({len(img_bytes):,} bytes)")
    
    if len(images) > 1:
        print(f"     (Ignored {len(images) - 1} extra image(s) from API)")
    
    print(f"  [OK] Complete!")
    return True


def main():
    """Main entry point"""
    print("=" * 60)
    print("Generate 2D Plans from Pre-configured Variations")
    print("=" * 60)
    
    # Check required paths
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return
    
    if not MASTER_PROMPT_PATH.exists():
        print(f"ERROR: Master prompt not found: {MASTER_PROMPT_PATH}")
        return
    
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found in .env file")
        return
    
    print(f"\nUsing master prompt: {MASTER_PROMPT_PATH.name}")
    
    # Get all variation folders
    folders = sorted([f for f in SOURCE_DIR.iterdir() if f.is_dir()])
    
    # Allow limiting via command line
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    folders = folders[:limit]
    
    print(f"\nProcessing {len(folders)} variations")
    print(f"Source: {SOURCE_DIR}")
    
    success = 0
    for folder in folders:
        try:
            if process_variation(folder):
                success += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(folders)} variations processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
