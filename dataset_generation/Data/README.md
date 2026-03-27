# Data Folder

This folder contains all the dataset files used by the generation scripts.

**⚠️ This folder is gitignored** - datasets are too large for git.

## Expected Structure

```
Data/
├── D01_120_variations/          # Pre-configured variations with PNG floor plans
│   ├── 1/
│   │   ├── *.png               # Floor plan image
│   │   └── configuration.json
│   ├── 2/
│   │   ├── *.png
│   │   └── configuration.json
│   └── ...
│
├── pairs_7_Bauhaus_w-image/    # Bauhaus style dataset
│   └── ...
│
└── high_quality_architectural/  # SVG floor plans for generation
    ├── 1/
    │   └── model.svg
    ├── 2/
    │   └── model.svg
    └── ...
```

## Setup

1. **Place your datasets here:**
   - Download or copy your dataset folders into this `Data/` directory
   
2. **Verify structure:**
   ```
   dataset_generation/
   └── Data/
       ├── D01_120_variations/      ← For generate_from_variations.py
       ├── pairs_7_Bauhaus_w-image/ ← Additional dataset
       └── high_quality_architectural/ ← For generate_interior_from_svg.py
   ```

3. **Run scripts:**
   - Scripts automatically look for data in this folder
   - No need to change paths in the code

## Sharing Datasets

Since datasets are gitignored, share them via:
- Google Drive / Dropbox
- AWS S3 / Azure Blob Storage
- Hugging Face Datasets
- DVC (Data Version Control)
