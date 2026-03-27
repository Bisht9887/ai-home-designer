"""
SVG Floor Plan → Interior Design Generator

For each SVG floor plan, this script:
1. Creates configuration.json (design parameters)
2. Creates master_prompt.json (prompt template)  
3. Creates empty_plan.png (from SVG)
4. Merges config + prompt and calls Nano Banana Pro
5. Saves generated 2D plan and semi-3D view
"""

import requests
import base64
import json
import random
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from dotenv import load_dotenv

# Paths - relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "Data"
SOURCE_DIR = DATA_DIR / "high_quality_architectural"
OUTPUT_DIR = SCRIPT_DIR / "generated_interiors"

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# API Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("API_URL", "https://openrouter.ai/api/v1/chat/completions")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Master prompt template (same as your examples)
MASTER_PROMPT = {
    "task": "Using the provided empty floorplan image as strict layout conditioning, generate exactly ONE finished interior design and return exactly TWO SEPARATE IMAGES (not a grid, not a collage). Image 1 must be a strictly 2D top-down orthographic floor plan (flat vector/CAD-style). Image 2 must be a semi-3D top-down dollhouse-style view. Keep the architecture identical to the input.",
    "input_image_role": "empty_floorplan_layout_conditioning",
    "num_outputs": 2,
    "global_constraints": {
        "do_not_change_architecture": True,
        "keep_walls_doors_windows_fixed": True,
        "keep_room_count_exact": "{{rooms_count}}",
        "do_not_add_or_remove_rooms": True,
        "keep_scale_realistic": True,
        "no_furniture_overlap": True,
        "respect_door_swings": True,
        "keep_clear_walkways": ">= 80cm where possible"
    },
    "design_brief": {
        "household": "{{household}}",
        "persona_lifestyle": "{{persona_lifestyle}}",
        "hobby_tags": "{{hobby_tags_json_array}}",
        "personality_social": "{{personality_social}}",
        "routine": "{{routine}}",
        "needs_must_implement": "{{needs_json_array}}",
        "desires_implement_if_feasible": "{{desires_json_array}}",
        "budget_level": "{{budget}}",
        "storage_intensity": "{{storage_intensity}}",
        "cabinet_target_guide": "{{cabinet_target_guide}}"
    },
    "style_pack": {
        "interior_style": "{{interior_style}}",
        "palette_description": "{{palette}}",
        "materials_description": "{{materials}}"
    },
    "outputs": [
        {
            "output_index": 1,
            "name": "2D_topview_plan",
            "camera": "top_down_orthographic",
            "render_mode": "flat_2D_vector_diagram",
            "requirements": [
                "STRICTLY 2D: no depth, no perspective, no isometric angle",
                "NO shadows, NO ambient occlusion, NO bevels, NO extrusion",
                "clean wall outlines exactly matching the input",
                "flat color blocks for floors/rugs/material zones",
                "simple filled shapes for furniture",
                "crisp edges, diagrammatic look"
            ]
        },
        {
            "output_index": 2,
            "name": "semi3D_topview",
            "camera": "top_down_semi_3D_dollhouse",
            "render_mode": "semi3D_with_depth",
            "requirements": [
                "visible depth and soft shadows",
                "top-down dollhouse/cutaway feel",
                "same exact furniture placement and materials as Image 1 (same design), only different view style"
            ],
            "style_note": "Do not copy the decor style from any reference image; only use the semi-3D viewpoint concept."
        }
    ],
    "negative_prompt": "grid, collage, split-panel, two views in one image, extra panels, multiple layouts, changing walls, moving doors, changing windows, adding/removing rooms, perspective in 2D plan, 3D in 2D plan, shadows in 2D plan, ambient occlusion in 2D plan, isometric in 2D plan, watermark, logo, unreadable text"
}

# Configuration options for random generation
HOUSEHOLDS = ["solo", "couple", "family", "multi-generational", "roommates"]
LIFESTYLES = ["minimalist", "maximalist", "retired", "work-from-home", "frequent traveler", "entertainer"]
HOBBY_OPTIONS = ["cooking", "reading", "gaming", "yoga", "art", "music", "plants", "collecting", "travel", "fitness"]
SOCIAL_OPTIONS = ["introvert", "balanced", "extrovert", "likes big parties", "small gatherings only"]
ROUTINES = ["morning workout", "night owl", "reading nights", "early riser", "work from home"]
NEEDS_OPTIONS = ["sound reduction", "pet friendly", "child safe", "wheelchair accessible", "multi-use furniture", "extra storage"]
DESIRES_OPTIONS = ["zen", "creative", "organized", "calm", "vibrant", "cozy", "luxurious"]
BUDGETS = ["low", "mid", "high", "luxury"]
STORAGE_LEVELS = ["minimal", "medium", "high", "maximum"]
STYLES = ["Scandinavian", "Industrial", "Modern", "Minimalist", "Bohemian", "Mid-Century Modern", "Japanese", "Contemporary"]
PALETTES = ["warm earth tones", "neutral whites", "cool blues", "monochrome", "forest greens", "desert sunset", "ocean vibes"]
MATERIALS_OPTIONS = ["white quartz", "leather accents", "black granite", "concrete", "light oak wood", "walnut", "marble", "brass", "velvet", "linen"]


def analyze_svg_for_rooms(svg_path):
    """Parse SVG and count rooms by unique IDs"""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Track unique room IDs to avoid double-counting
        room_ids = set()
        kitchen_count = 0
        bath_count = 0
        closet_count = 0
        
        # Only count elements that have an ID (unique rooms)
        for elem in root.iter():
            class_attr = elem.get('class', '')
            elem_id = elem.get('id', '')
            
            # Only count if it's a Space with a unique ID
            if 'Space' in class_attr and elem_id and elem_id not in room_ids:
                room_ids.add(elem_id)
                
                if 'Kitchen' in class_attr:
                    kitchen_count += 1
                elif 'Bath' in class_attr:
                    bath_count += 1
                elif 'Closet' in class_attr:
                    closet_count += 1
        
        room_count = len(room_ids)
        
        # If no rooms found with IDs, estimate from polygon count
        if room_count == 0:
            polygon_count = sum(1 for elem in root.iter() if 'polygon' in elem.tag.lower())
            room_count = max(polygon_count // 10, 3)  # Rough estimate
        
        return {
            'rooms_count': max(room_count, 2),  # At least 2 rooms
            'kitchen_cabinet_areas': max(kitchen_count * 3, 2),
            'bath_vanities': max(bath_count, 1),
            'closet_count': max(closet_count, 1)
        }
    except Exception as e:
        print(f"    Warning: Could not parse SVG: {e}")
        return {
            'rooms_count': 3,
            'kitchen_cabinet_areas': 4,
            'bath_vanities': 1,
            'closet_count': 2
        }


def generate_configuration(variation_id, room_info):
    """Generate a random configuration.json"""
    return {
        "variation_id": variation_id,
        "rooms_count": room_info['rooms_count'],
        "household": random.choice(HOUSEHOLDS),
        "persona_lifestyle": random.choice(LIFESTYLES),
        "hobby_tags_json_array": random.sample(HOBBY_OPTIONS, k=random.randint(1, 3)),
        "personality_social": random.choice(SOCIAL_OPTIONS),
        "routine": random.choice(ROUTINES),
        "needs_json_array": random.sample(NEEDS_OPTIONS, k=random.randint(1, 2)),
        "desires_json_array": random.sample(DESIRES_OPTIONS, k=random.randint(1, 2)),
        "budget": random.choice(BUDGETS),
        "storage_intensity": random.choice(STORAGE_LEVELS),
        "cabinet_target_guide": {
            "kitchen_cabinet_areas": room_info['kitchen_cabinet_areas'],
            "bath_vanities": room_info['bath_vanities'],
            "closet_count": room_info['closet_count']
        },
        "interior_style": random.choice(STYLES),
        "palette": random.choice(PALETTES),
        "materials": random.sample(MATERIALS_OPTIONS, k=random.randint(2, 4))
    }


def svg_to_png(svg_path, png_path):
    """Convert SVG to PNG - tries multiple methods"""
    
    # Method 1: Try cairosvg (needs Cairo/GTK installed on Windows)
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1024)
        return True
    except Exception:
        pass
    
    # Method 2: Try Wand/ImageMagick
    try:
        from wand.image import Image as WandImage
        with WandImage(filename=str(svg_path)) as img:
            img.format = 'png'
            img.save(filename=str(png_path))
        return True
    except Exception:
        pass
    
    # Method 3: Try Inkscape via command line
    try:
        import subprocess
        result = subprocess.run(
            ['inkscape', str(svg_path), '--export-filename=' + str(png_path), '-w', '1024'],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and png_path.exists():
            return True
    except Exception:
        pass
    
    # Method 4: Render SVG elements to PNG using Pillow (basic)
    try:
        from PIL import Image, ImageDraw
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Get dimensions from viewBox or width/height
        viewbox = root.get('viewBox')
        if viewbox:
            parts = viewbox.split()
            width = float(parts[2])
            height = float(parts[3])
        else:
            width = float(root.get('width', '1024').replace('px', '').replace('pt', ''))
            height = float(root.get('height', '768').replace('px', '').replace('pt', ''))
        
        # Scale to max 1024
        scale = 1024 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Create white background image
        img = Image.new('RGB', (new_width, new_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw basic shapes from SVG (walls, polygons)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        for polygon in root.iter():
            if 'polygon' in polygon.tag.lower():
                points_str = polygon.get('points', '')
                if points_str:
                    try:
                        coords = []
                        for pair in points_str.strip().split():
                            if ',' in pair:
                                x, y = pair.split(',')
                                coords.append((float(x) * scale, float(y) * scale))
                        if len(coords) >= 3:
                            # Check if it's a wall (black fill)
                            fill = polygon.get('fill', '#ffffff')
                            if fill == '#000000' or 'Wall' in polygon.get('class', ''):
                                draw.polygon(coords, fill='black', outline='black')
                            else:
                                draw.polygon(coords, outline='gray')
                    except:
                        pass
        
        img.save(png_path, 'PNG')
        print(f"    Note: Basic SVG render (install Inkscape or GTK for better quality)")
        return True
        
    except Exception as e:
        print(f"    ERROR: All SVG conversion methods failed")
        print(f"    Install one of: GTK3 runtime, Inkscape, or ImageMagick")
        print(f"    Details: {e}")
        return False


def merge_config_into_prompt(config):
    """Merge configuration values into the master prompt template"""
    prompt_str = json.dumps(MASTER_PROMPT)
    
    replacements = {
        "{{rooms_count}}": str(config['rooms_count']),
        "{{household}}": config['household'],
        "{{persona_lifestyle}}": config['persona_lifestyle'],
        "{{hobby_tags_json_array}}": json.dumps(config['hobby_tags_json_array']),
        "{{personality_social}}": config['personality_social'],
        "{{routine}}": config['routine'],
        "{{needs_json_array}}": json.dumps(config['needs_json_array']),
        "{{desires_json_array}}": json.dumps(config['desires_json_array']),
        "{{budget}}": config['budget'],
        "{{storage_intensity}}": config['storage_intensity'],
        "{{cabinet_target_guide}}": json.dumps(config['cabinet_target_guide']),
        "{{interior_style}}": config['interior_style'],
        "{{palette}}": config['palette'],
        "{{materials}}": json.dumps(config['materials'])
    }
    
    for key, value in replacements.items():
        prompt_str = prompt_str.replace(key, value)
    
    return prompt_str


def call_nano_banana(png_path, merged_prompt):
    """Call Nano Banana Pro API"""
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
                    "text": f"Follow these instructions exactly:\n\n{merged_prompt}"
                }
            ]
        }],
        "modalities": ["image", "text"]
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()


def extract_images(result):
    """Extract images from API response"""
    images = []
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


def copy_original_files(source_folder, output_folder):
    """Copy all original files from source to output/originals folder"""
    import shutil
    
    originals_folder = output_folder / "originals"
    originals_folder.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for file in source_folder.iterdir():
        if file.is_file():
            dest = originals_folder / file.name
            shutil.copy2(file, dest)
            copied += 1
    
    return copied


def process_svg_folder(folder_path):
    """Process a single SVG folder"""
    folder_name = folder_path.name
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")
    
    svg_path = folder_path / "model.svg"
    if not svg_path.exists():
        print(f"  Skipping: No model.svg")
        return False
    
    # Create output folder
    output_folder = OUTPUT_DIR / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Step 0: Copy original files
    print(f"  0. Copying original files...")
    num_copied = copy_original_files(folder_path, output_folder)
    print(f"     Copied {num_copied} files to originals/")
    
    # Step 1: Analyze SVG for room info
    print(f"  1. Analyzing SVG...")
    room_info = analyze_svg_for_rooms(svg_path)
    print(f"     Found {room_info['rooms_count']} rooms")
    
    # Step 2: Generate configuration.json
    print(f"  2. Creating configuration.json...")
    config = generate_configuration(folder_name, room_info)
    config_path = output_folder / "configuration.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"     Style: {config['interior_style']}, Household: {config['household']}")
    
    # Step 3: Save master_prompt.json
    print(f"  3. Creating master_prompt.json...")
    prompt_path = output_folder / "master_prompt.json"
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(MASTER_PROMPT, f, indent=2)
    
    # Step 4: Create empty_plan.png
    print(f"  4. Creating empty_plan.png...")
    png_path = output_folder / "empty_plan.png"
    if not svg_to_png(svg_path, png_path):
        print(f"     ERROR: Could not create PNG")
        return False
    
    # Step 5: Merge and call API
    print(f"  5. Calling Nano Banana Pro API...")
    merged_prompt = merge_config_into_prompt(config)
    result = call_nano_banana(png_path, merged_prompt)
    
    if "error" in result:
        print(f"     ERROR: {result['error']}")
        return False
    
    # Step 6: Save generated images
    images = extract_images(result)
    print(f"     Received {len(images)} image(s)")
    
    for i, img_base64 in enumerate(images):
        name = "2D_plan.png" if i == 0 else "semi3D_view.png" if i == 1 else f"output_{i+1}.png"
        out_path = output_folder / name
        img_bytes = base64.b64decode(img_base64)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"     Saved: {name} ({len(img_bytes):,} bytes)")
    
    print(f"  ✓ Complete!")
    return True


def main():
    """Main entry point"""
    print("=" * 60)
    print("SVG Floor Plan → Interior Design Generator")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get folders to process (limit for testing)
    folders = sorted([f for f in SOURCE_DIR.iterdir() if f.is_dir()])[:10]
    
    print(f"\nProcessing {len(folders)} folders (limited to 10)")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    success = 0
    for folder in folders:
        try:
            if process_svg_folder(folder):
                success += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
    
    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(folders)} folders processed successfully")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

