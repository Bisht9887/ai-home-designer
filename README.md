# HomAIker
AI Masterclass Capstone Project HomAIker

## Setup

### 1. Install Dependencies

```bash
cd dataset_generation
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_actual_api_key_here
API_URL=https://openrouter.ai/api/v1/chat/completions
```

Get your API key from: [https://openrouter.ai/keys](https://openrouter.ai/keys)

### 3. Available Scripts

- **`generate_interiors.py`** - ⭐ **Main script** - Generate 2D plans from any dataset with variations
- **`generate_interior_from_svg.py`** - Generate from SVG floor plans
- **`generate_from_variations.py`** - ⚠️ *Legacy* - Use `generate_interiors.py` instead
- **`generate_bauhaus_pairs.py`** - ⚠️ *Legacy* - Use `generate_interiors.py` instead

### 4. Project Structure

```
HomAIker/
├── .env                          # Your API keys (DO NOT COMMIT)
├── .env.example                  # Template for .env
├── dataset_generation/
│   ├── generate_interiors.py     # ⭐ Main generation script
│   ├── generate_interior_from_svg.py
│   ├── master_prompt_2D-only.json
│   ├── requirements.txt
│   ├── Data/                     # All datasets go here (gitignored)
│   │   ├── README.md            # Data folder guide
│   │   ├── D01_120_variations/  # Pre-configured variations
│   │   ├── pairs_7_Bauhaus_w-image/  # Bauhaus dataset
│   │   └── high_quality_architectural/  # SVG floor plans
│   └── generated_interiors/      # Output folder (gitignored)
├── training_pipeline/            # Vision pipeline: layout → OpenCV → (future: CNN, model)
│   ├── opencv_preprocess.py     # Part 1: layout preprocessing
│   ├── run_preprocess.py        # Run preprocessing on a dataset
│   └── README.md                # Pipeline step 1 docs
```

### 5. Add Your Datasets

Place your dataset folders inside `dataset_generation/Data/`:

```bash
cd dataset_generation/Data
# Copy or move your datasets here:
# - D01_120_variations/
# - pairs_7_Bauhaus_w-image/
# - high_quality_architectural/
```

See `dataset_generation/Data/README.md` for details.

## Usage

Make sure your datasets are in `dataset_generation/Data/` before running.

### Generate Interior Designs (Unified Script)

**New recommended script** - works with any dataset:

```bash
cd dataset_generation

# Use default dataset (D01_120_variations) with default limit (3)
python generate_interiors.py

# Specify dataset and limit
python generate_interiors.py D01_120_variations 10
python generate_interiors.py pairs_7_Bauhaus_w-image 5

# Process all variations in a dataset
python generate_interiors.py D01_120_variations 999999
```

**Configuration:**
Edit the top of `generate_interiors.py` to change defaults:
```python
DEFAULT_DATASET = "D01_120_variations"  # Change this
DEFAULT_LIMIT = 3                        # Change this
API_TIMEOUT = 180                        # Adjust if needed
```

### Generate from SVG Floor Plans

```bash
cd dataset_generation
python generate_interior_from_svg.py
```

## Security Notes

- **Never commit `.env`** - it's already in `.gitignore`
- If you accidentally commit API keys, **rotate them immediately** at OpenRouter
- Use `.env.example` to share configuration templates with others
