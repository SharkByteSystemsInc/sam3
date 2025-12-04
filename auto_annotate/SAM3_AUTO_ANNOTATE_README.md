# SAM3 Auto-Annotation Script

## Overview

`sam3_auto_annotate.py` is a command-line tool that uses SAM3 (Segment Anything Model 3) to automatically generate instance segmentation masks for images. The script processes images with corresponding JSON metadata files and detects four classes:
- **car**
- **license_plate**
- **person** (people outside vehicles)
- **passenger** (people inside vehicles with closed doors)

The results are saved in the GVA metadata format with NPZ mask files.

## Features

- ✅ **Multi-class segmentation**: Detects cars, license plates, persons, and passengers in a single pass
- ✅ **Passenger detection**: Automatically classifies persons inside vehicles as passengers
- ✅ **Confidence scores**: Provides per-instance confidence values (0.4-1.0) saved in JSON metadata
- ✅ **Text-prompted inference**: Uses SAM3's open-vocabulary capabilities
- ✅ **Overlap filtering**: Removes duplicate masks with high IoU
- ✅ **Directory recursion**: Processes nested directory structures
- ✅ **Preserves folder structure**: Recreates input directory structure in output
- ✅ **Real-time progress bar**: Shows processing progress with success/fail counts
- ✅ **GVA metadata format**: Compatible with existing GVA data pipeline
- ✅ **NPZ compression**: Efficient storage of segmentation masks

## Installation

### Prerequisites

1. **Conda environment** (sam3):
   ```bash
   conda activate sam3
   ```

2. **SAM3 repository**:
   - The script must be run from the SAM3 repository directory
   - SAM3 model will be automatically downloaded from HuggingFace on first run

3. **CUDA GPU** (recommended):
   - For fast inference (falls back to CPU if not available)

## Usage

### Basic Command

```bash
# Option 1: Using conda run (recommended)
conda run -n sam3 python -u sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output

# Option 2: Activating environment directly (better for interactive use)
conda activate sam3
python sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output
```

**Note**: The `-u` flag (unbuffered mode) ensures the progress bar displays in real-time when using `conda run`.

### Full Command-Line Options

```bash
python sam3_auto_annotate.py \
    --input_top_dir INPUT_DIR \          # Required: Input directory with JPG and JSON files
    --out_top_dir OUTPUT_DIR \            # Required: Output directory for results
    [--override_previous_annotations] \   # Optional: Reprocess existing annotations
    [--device {cuda,cpu}] \               # Optional: Device to use (default: cuda if available)
    [--log_level {DEBUG,INFO,WARNING,ERROR}]  # Optional: Logging verbosity (default: INFO)
```

### Examples

#### Process a single directory:
```bash
conda run -n sam3 python -u sam3_auto_annotate.py \
    --input_top_dir /tmp/sam3_test_input \
    --out_top_dir /tmp/sam3_test_output
```

#### Process with recursion and verbose logging:
```bash
conda run -n sam3 python -u sam3_auto_annotate.py \
    --input_top_dir /ssd/sbs/gpu2/annotation/negatively_reviewed_train/gva \
    --out_top_dir /ssd/sbs/output/sam3_annotations \
    --log_level DEBUG
```

#### Re-process existing annotations:
```bash
conda run -n sam3 python -u sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output \
    --override_previous_annotations
```

## Input Format

### Required Files

For each image to be processed, you need:
1. **JPG image file**: e.g., `entry1_1743414465802_1743480007459.jpg`
2. **JSON metadata file**: e.g., `entry1_1743414465802_1743480007459.json`

### Input Directory Structure

```
input_top_dir/
├── subdir1/
│   ├── image1.jpg
│   ├── image1.json
│   ├── image2.jpg
│   └── image2.json
└── subdir2/
    ├── image3.jpg
    └── image3.json
```

### JSON Metadata Format

The script reads standard GVA metadata format. Example:
```json
{
  "image": {
    "dimensions": {
      "width": 1920,
      "height": 1080
    },
    "timestamp_ms": 1743414465802
  },
  "source": {
    "source_type": "GVA",
    "camera": "entry1",
    "garage": "cen005"
  },
  ...
}
```

## Output Format

### Output Directory Structure

The script preserves the input directory structure:

```
out_top_dir/
├── subdir1/
│   ├── image1.jpg          # Copied from input
│   ├── image1.json         # Updated with segmentation metadata
│   ├── image1.npz          # Encoded instance masks
│   ├── image2.jpg
│   ├── image2.json
│   └── image2.npz
└── subdir2/
    ├── image3.jpg
    ├── image3.json
    └── image3.npz
```

### NPZ File Format

Compressed numpy array with a single key:
- **Key**: `'mask'`
- **Shape**: `(height, width)`
- **dtype**: `uint8`
- **Values**:
  - `0`: Background
  - `36, 72, 108, 144, 180, 216, 252...`: Instance masks (evenly distributed values)

Example:
```python
import numpy as np
data = np.load('image.npz')
mask = data['mask']  # Shape: (1080, 1920), dtype: uint8
unique_values = np.unique(mask)  # [0, 36, 72, 108, 144, 180, 216, 252]
# 0 = background, other values = individual instances
```

### Updated JSON Metadata

The script adds/updates the `annotation.segmentation` field:

```json
{
  "image": { ... },
  "source": { ... },
  "annotation": {
    "segmentation": [
      {
        "annotation_date": "2025-12-03",
        "considered_classes": ["car", "license_plate", "person", "passenger"],
        "annotator": {
          "annotator_type": "Model",
          "model_id": "SAM3",
          "model_metadata": {
            "model_type": "segmentation",
            "model_name": "SAM3",
            "classes": {
              "0": "car",
              "1": "license_plate",
              "2": "person",
              "3": "passenger"
            }
          }
        },
        "labels": {
          "annotation_format": "all_masks_on_single_array",
          "annotation_format_version": "1.0",
          "path": "image.npz",
          "properties": {
            "car": [180, 144, 72, 36, 108],
            "license_plate": [252, 216]
          },
          "confidences": {
            "car": [0.666, 0.842, 0.919, 0.895, 0.634],
            "license_plate": [0.856, 0.799]
          }
        },
        "review": null
      }
    ],
    ...
  }
}
```

**Key Fields:**
- `considered_classes`: List of classes the model was prompted with
- `labels.properties`: Maps class names to list of mask values in the NPZ file
- `labels.confidences`: Maps class names to list of confidence scores (3 decimal places), in the same order as properties
- `labels.path`: Name of the NPZ file containing masks

## How It Works

### Processing Pipeline

1. **Model Initialization**:
   - Loads SAM3 model once at startup
   - Creates processor for image inference

2. **For Each JSON File**:
   ```
   a. Load metadata from JSON
   b. Find corresponding JPG image
   c. Run SAM3 inference 3 times:
      - Prompt: "car"
      - Prompt: "license plate"
      - Prompt: "person"
   d. Collect all detected masks with class labels
   e. Detect passengers: classify persons inside cars as passengers
      - Check if person center is inside car bounding box
      - Calculate person-car mask overlap (containment ratio)
      - Person is passenger if >60% overlap OR >40% overlap + well inside car bbox
      - Remove classified passengers from "person" class
   f. Filter overlapping masks (IoU > 0.5)
   g. Encode masks into single uint8 array
   h. Create output directory structure
   i. Copy JPG to output directory
   j. Save NPZ with encoded masks
   k. Update JSON metadata with segmentation info
   l. Save updated JSON to output directory
   ```

3. **Mask Encoding**:
   - Masks sorted by area (largest first)
   - Unique values evenly distributed: `255 / num_masks`
   - Example: 7 masks → values [36, 72, 108, 144, 180, 216, 252]

4. **Overlap Filtering**:
   - Computes IoU between all mask pairs
   - Removes masks with IoU > 0.5 (keeps first occurrence)

### SAM3 Inference Details

- **Text Prompts**: Separate inference for each class
  - "car"
  - "license plate"
  - "person"

- **Passenger Detection Algorithm**:
  - **Input**: Detected person and car masks
  - **Criteria for passenger classification**:
    1. Person center must be inside car bounding box
    2. Calculate containment ratio: `(person ∩ car) / person_area`
    3. Check if person bbox is well inside car bbox (10px tolerance from edges)
    4. Classify as passenger if:
       - Containment ratio > 60%, OR
       - Containment ratio > 40% AND person bbox well inside car bbox
  - **Result**: Persons classified as passengers are removed from "person" class

- **Confidence Scores**: SAM3 provides confidence scores for each detected instance
  - Confidence threshold: 0.4 (masks with lower scores are discarded)
  - Confidence scores are preserved in ModelResult objects
  - Typical range: 0.4-1.0

- **Resolution**: 1008×1008 (SAM3 internal resolution, automatically resized)

- **Output**: Binary masks resized to original image dimensions

## Performance

### Speed Estimates

- **GPU (CUDA)**: ~0.5 seconds per image (3 classes)
- **CPU**: ~5-10 seconds per image (3 classes)

### Memory Requirements

- **GPU**: ~4-6 GB VRAM
- **RAM**: ~8 GB recommended

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU instead:
```bash
python sam3_auto_annotate.py --device cpu ...
```

#### 2. No JSON files found
```
WARNING: No JSON files found in /path/to/input
```
**Solution**: Check that input directory contains `.json` files

#### 3. Image not found
```
WARNING: No image found at /path/to/file.json
```
**Solution**: Ensure corresponding `.jpg` file exists for each `.json` file

#### 4. Import errors
```
ModuleNotFoundError: No module named 'sam3'
```
**Solution**: Make sure to run from SAM3 repository directory and use conda environment:
```bash
cd /ssd/sbs/repos/sam3
conda run -n sam3 python -u sam3_auto_annotate.py ...
```

## Technical Details

### Mask Encoding Algorithm

The `encode_binary_masks()` function combines multiple binary masks into a single uint8 array:

1. **Input**: List of N binary masks (H×W boolean arrays)
2. **Processing**:
   - Sort masks by area (descending)
   - Assign evenly distributed values: `step_size = 255 // N`
   - Values: `[step_size, 2*step_size, 3*step_size, ...]`
   - Apply masks in order (largest first, so smaller masks override)
3. **Output**:
   - Combined mask array (H×W uint8)
   - List of mask values (in original input order)

### Class Name Mapping

Text prompts are converted to class names for metadata:
- `"car"` → `"car"`
- `"license plate"` → `"license_plate"` (spaces replaced with underscores)
- `"person"` → `"person"` (people outside vehicles)
- Detected passengers → `"passenger"` (people inside vehicles with closed doors)

## Example Test Run

```bash
# Setup test data
mkdir -p /tmp/sam3_test_input
cp /ssd/sbs/gpu2/annotation/negatively_reviewed_train/gva/cen005/entry1/entry1_*.{jpg,json} \
   /tmp/sam3_test_input/

# Run annotation
conda run -n sam3 python -u sam3_auto_annotate.py \
    --input_top_dir /tmp/sam3_test_input \
    --out_top_dir /tmp/sam3_test_output \
    --log_level INFO

# Check results
ls -lh /tmp/sam3_test_output/
```

**Expected Output**:
```
2025-12-03 08:58:45,123 - INFO - Input directory: /tmp/sam3_test_input
2025-12-03 08:58:45,124 - INFO - Output directory: /tmp/sam3_test_output
2025-12-03 08:58:45,124 - INFO - Device: cuda
2025-12-03 08:58:45,125 - INFO - Loading SAM3 model...
2025-12-03 08:58:50,456 - INFO - SAM3 model loaded successfully
2025-12-03 08:58:50,457 - INFO - Found 2 JSON files to process
Processing images:  50%|█████████████▌              | 1/2 [00:00<00:00, 2.00img/s, ✓=1, ✗=0]
Processing images: 100%|████████████████████████████| 2/2 [00:00<00:00, 2.20img/s, ✓=2, ✗=0]
2025-12-03 08:58:51,001 - INFO -
============================================================
2025-12-03 08:58:51,001 - INFO - Processing complete!
2025-12-03 08:58:51,001 - INFO - Successfully processed: 2
2025-12-03 08:58:51,001 - INFO - Failed: 0
2025-12-03 08:58:51,001 - INFO -
============================================================
```

The progress bar now shows:
- **Real-time updates** during processing
- **Success count (✓)**: Number of successfully processed images
- **Fail count (✗)**: Number of failed images
- **Processing speed** (images/second)

## Verification

### Check NPZ Contents

```python
import numpy as np
import json

# Load mask
data = np.load('output/image.npz')
mask = data['mask']
print(f"Mask shape: {mask.shape}")
print(f"Unique values: {np.unique(mask)}")

# Load metadata
with open('output/image.json', 'r') as f:
    metadata = json.load(f)

# Print detected instances
properties = metadata['annotation']['segmentation'][0]['labels']['properties']
for class_name, mask_values in properties.items():
    print(f"{class_name}: {len(mask_values)} instances - {mask_values}")
```

### Visualize Masks (Optional)

```python
import cv2
import numpy as np

# Load image and mask
image = cv2.imread('output/image.jpg')
mask = np.load('output/image.npz')['mask']

# Visualize each instance
unique_values = np.unique(mask)[1:]  # Exclude background (0)
for value in unique_values:
    instance_mask = (mask == value).astype(np.uint8) * 255
    cv2.imshow(f'Instance {value}', instance_mask)
cv2.waitKey(0)
```

## Limitations

1. **Fixed classes**: Only detects car, license_plate, person, and passenger
2. **No tracking**: Processes each image independently (no temporal consistency)
3. **Memory intensive**: Loads entire model into memory
4. **Sequential processing**: Processes images one at a time (no batching)

## Future Improvements

- [ ] Add batched inference for multiple images
- [ ] Support custom class lists via command-line
- [ ] Add visualization output option
- [ ] Implement multi-GPU support
- [ ] Add resume capability (skip already processed files automatically)
- [ ] Add detailed per-image statistics logging

## References

- **SAM3 Paper**: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- **SAM3 Repository**: https://github.com/facebookresearch/sam3
- **GVA Segmentation**: /raid/michal/sbs/repos/gva_segmentation

## License

This script is part of the SAM3 project. See the main SAM3 LICENSE file for details.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review SAM3 documentation
3. Check example data format: `/ssd/sbs/gpu2/annotation/negatively_reviewed_train/gva/cen005/entry1`

---

**Last Updated**: 2025-12-03
**Script Version**: 1.0
**SAM3 Version**: As per repository
