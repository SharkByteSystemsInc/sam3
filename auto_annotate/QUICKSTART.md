# SAM3 Auto-Annotation Quick Start Guide

## Installation

1. Install SAM3 package:
```bash
cd /path/to/sam3
pip install -e .
```

2. For agent mode, install additional dependencies:
```bash
# Install pycocotools (for RLE mask decoding)
pip install pycocotools

# Install vLLM (in separate environment recommended)
conda create -n vllm python=3.12
conda activate vllm
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

## Basic Usage

### Standard Mode (Simple and Fast)

```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output \
    --device cuda
```

**What it does:**
- Segments cars, license plates, and persons in images
- Uses direct SAM3 inference
- Fast processing (~1-2 seconds per image)
- Outputs in GVA metadata format

### Agent Mode (Advanced and Accurate)

**Step 1:** Start vLLM server (in separate terminal)
```bash
conda activate vllm
vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --max-model-len 16000 \
    --tensor-parallel-size 4 \
    --allowed-local-media-path / \
    --enforce-eager \
    --port 8001
```

**Step 2:** Run annotation with agent
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output \
    --device cuda \
    --agent
```

**What it does:**
- Uses MLLM to iteratively refine segmentation
- Better handling of complex queries
- Special prompt for persons: "all people, except those displayed on the posters"
- Slower (~30-60 seconds per image)
- Same output format as standard mode

## Input Format

Your input directory should contain:
- **Images**: `.jpg` files
- **Metadata**: Corresponding `.json` files (GVA format)

```
input_dir/
├── video_001/
│   ├── frame_001.jpg
│   ├── frame_001.json
│   ├── frame_002.jpg
│   ├── frame_002.json
│   └── ...
└── video_002/
    └── ...
```

## Output Format

The output directory mirrors the input structure:

```
output_dir/
├── video_001/
│   ├── frame_001.jpg       # Copy of original image
│   ├── frame_001.json      # Updated metadata with annotations
│   ├── frame_001.npz       # Encoded masks
│   └── ...
└── video_002/
    └── ...
```

### Metadata JSON Structure

```json
{
  "annotation": {
    "segmentation": [
      {
        "annotation_date": "2024-12-04",
        "considered_classes": ["car", "license_plate", "person", "passenger"],
        "annotator": {
          "annotator_type": "Model",
          "model_id": "SAM3"
        },
        "labels": {
          "annotation_format": "all_masks_on_single_array",
          "path": "frame_001.npz",
          "properties": {
            "car": [51, 102],
            "person": [153],
            "license_plate": [204]
          },
          "confidences": {
            "car": [0.876, 0.923],
            "person": [0.812],
            "license_plate": [0.734]
          }
        }
      }
    ]
  }
}
```

### NPZ Mask File

- Single uint8 array with shape (height, width)
- Background pixels = 0
- Each instance has a unique pixel value (51, 102, 153, etc.)
- Values correspond to `properties` in metadata

## Command Line Options

### Required Arguments
- `--input_top_dir`: Path to input directory with images and metadata
- `--out_top_dir`: Path to output directory

### Optional Arguments
- `--device`: Device to run on (`cuda` or `cpu`, default: `cuda`)
- `--agent`: Enable agent mode (default: `False`)
- `--llm_server_url`: LLM server URL (default: `http://0.0.0.0:8001/v1`)
- `--llm_model`: LLM model name (default: `Qwen/Qwen3-VL-8B-Thinking`)
- `--override_previous_annotations`: Re-process existing annotations
- `--log_level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## Examples

### Example 1: Process with debug logging
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir ./data/input \
    --out_top_dir ./data/output \
    --log_level DEBUG
```

### Example 2: Agent mode with custom LLM
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir ./data/input \
    --out_top_dir ./data/output \
    --agent \
    --llm_server_url http://custom-server:8080/v1 \
    --llm_model custom-model-name
```

### Example 3: Override existing annotations
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir ./data/input \
    --out_top_dir ./data/output \
    --override_previous_annotations
```

### Example 4: CPU-only processing
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir ./data/input \
    --out_top_dir ./data/output \
    --device cpu
```

## Detected Classes

The tool automatically detects and segments:

1. **car**: All vehicles
2. **license_plate**: License plates on vehicles
3. **person**: People in the scene (excluding those on posters in agent mode)
4. **passenger**: People detected inside cars (derived from person + car)

## Features

### Passenger Detection
- Automatically identifies persons inside cars
- Reclassifies them as "passenger"
- Uses containment ratio and bounding box analysis
- Configurable thresholds in code

### Overlap Removal
- Filters out significantly overlapping masks
- Uses IoU threshold of 0.5
- Keeps higher-confidence detections

### Progress Tracking
- Real-time progress bar
- Shows success/failure counts
- Logs processing time and statistics

## Troubleshooting

### Issue: "No JSON files found"
**Solution**: Ensure input directory contains `.json` metadata files

### Issue: "Failed to load image"
**Solution**: Check that `.jpg` files exist alongside `.json` files

### Issue: Agent mode not working
**Solution**:
1. Verify vLLM server is running: `curl http://0.0.0.0:8001/v1/models`
2. Check server URL matches `--llm_server_url`
3. Ensure model is loaded in vLLM

### Issue: CUDA out of memory
**Solution**:
1. Use smaller batch size (modify code)
2. Reduce image resolution
3. Use CPU: `--device cpu`

### Issue: Slow processing
**Solution**:
- Use standard mode instead of agent mode
- Check GPU utilization
- Ensure CUDA is properly configured

## Performance Tips

### For Speed
- Use standard mode (no `--agent` flag)
- Ensure GPU is being utilized
- Process multiple images in parallel (modify code)

### For Accuracy
- Use agent mode (`--agent`)
- Adjust confidence thresholds in `model_inference.py`
- Customize prompts in `agent_inference.py`

## Output Validation

Verify your outputs:

```bash
# Check structure
ls -R output_dir/

# Count processed files
find output_dir/ -name "*.npz" | wc -l

# View metadata
python -c "import json; print(json.dumps(json.load(open('output_dir/video/frame.json')), indent=2))"

# Load and inspect masks
python -c "import numpy as np; masks = np.load('output_dir/video/frame.npz')['mask']; print(masks.shape, masks.dtype, masks.min(), masks.max())"
```

## Next Steps

- **Visualization**: Use `sam3.agent.viz.visualize()` to visualize masks
- **Evaluation**: Compare annotations against ground truth
- **Fine-tuning**: Adjust confidence thresholds and prompts
- **Integration**: Integrate outputs into your ML pipeline

## Support

For issues or questions:
- Check `AGENT_INTEGRATION_SUMMARY.md` for detailed documentation
- Review `ARCHITECTURE.md` for technical details
- See SAM3 repository: https://github.com/facebookresearch/sam3
