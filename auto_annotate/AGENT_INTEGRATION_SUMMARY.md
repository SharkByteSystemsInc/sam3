# SAM3 Agent Integration Summary

## Overview

This document describes the integration of SAM3 Agent mode into the auto-annotation script. The agent mode uses an MLLM (Multi-modal Large Language Model) to iteratively refine segmentation results, providing more accurate annotations for complex queries.

## Architecture Changes

### File Structure

The codebase has been refactored into four modular files:

```
auto_annotate/
├── utils.py                    # Shared utilities and data classes
├── model_inference.py          # Standard SAM3 inference
├── agent_inference.py          # SAM3 Agent inference (MLLM-based)
└── sam3_auto_annotate.py      # Main orchestration script
```

### Module Descriptions

#### 1. `utils.py`
- **Purpose**: Shared utilities and data classes
- **Contents**:
  - `ModelResult`: Container class for segmentation results (class_name, confidence, mask)

#### 2. `model_inference.py`
- **Purpose**: Standard (non-agent) SAM3 inference
- **Key Components**:
  - `SAM3Annotator`: Wrapper class for SAM3 model
    - Loads SAM3 image model with configurable device
    - `segment_image()`: Runs inference with text prompts
    - Applies 0.4 confidence threshold
  - `run_standard_inference()`: Processes multiple classes and returns results

#### 3. `agent_inference.py`
- **Purpose**: Agent-based SAM3 inference using MLLM
- **Key Components**:
  - `setup_agent_components()`: Initializes LLM and SAM3 clients
  - `run_agent_inference_for_class()`: Runs agent inference for a single class
    - Uses `sam3.agent.inference.run_single_image_inference()`
    - Decodes RLE masks from agent output
  - `run_agent_inference()`: Main entry point for agent-based processing
    - Handles multiple classes
    - **Special prompt for "person" class**: "all people, except those displayed on the posters"

#### 4. `sam3_auto_annotate.py`
- **Purpose**: Main orchestration script
- **Changes**:
  - Imports from modular files
  - Added CLI arguments for agent mode
  - Updated `process_image()` to support both standard and agent modes
  - Maintains all existing functionality (passenger detection, mask encoding, etc.)

## New CLI Arguments

### `--agent`
- **Type**: Boolean flag
- **Default**: False
- **Description**: Enable SAM3 agent mode with MLLM for iterative refinement
- **Usage**: `--agent`

### `--llm_server_url`
- **Type**: String
- **Default**: `http://0.0.0.0:8001/v1`
- **Description**: URL of the LLM server for agent mode
- **Usage**: `--llm_server_url http://localhost:8001/v1`

### `--llm_model`
- **Type**: String
- **Default**: `Qwen/Qwen3-VL-8B-Thinking`
- **Description**: LLM model name/identifier for agent mode
- **Usage**: `--llm_model Qwen/Qwen3-VL-8B-Thinking`

## Usage Examples

### Standard Mode (Default)
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output \
    --device cuda
```

### Agent Mode
```bash
# 1. Start vLLM server (in separate terminal/environment)
conda activate vllm
vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --max-model-len 16000 \
    --tensor-parallel-size 4 \
    --allowed-local-media-path / \
    --enforce-eager \
    --port 8001

# 2. Run annotation with agent mode
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output \
    --device cuda \
    --agent \
    --llm_server_url http://0.0.0.0:8001/v1 \
    --llm_model Qwen/Qwen3-VL-8B-Thinking
```

### Agent Mode with Custom LLM
```bash
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir /path/to/input \
    --out_top_dir /path/to/output \
    --agent \
    --llm_server_url http://custom-server:8000/v1 \
    --llm_model custom/model-name
```

## Key Features

### 1. Specialized Prompts
- **Standard classes** (car, license plate): Use class name as prompt
- **Person class**: Uses refined prompt "all people, except those displayed on the posters"
  - This helps the agent distinguish real people from images/posters

### 2. Agent Workflow
The agent mode follows an iterative refinement process:
1. **Initial Segmentation**: Agent calls `segment_phrase` tool with text prompt
2. **Examination**: Agent can call `examine_each_mask` to filter masks individually
3. **Selection**: Agent calls `select_masks_and_return` with final mask selection
4. **Alternative**: Agent can call `report_no_mask` if no valid results

### 3. Output Format
- **Identical to standard mode**: GVA metadata format
- **NPZ masks**: Encoded binary masks with unique values
- **JSON metadata**: Includes segmentation annotations with:
  - Class names
  - Mask values (pixel values in NPZ)
  - Confidence scores
  - Annotation metadata

### 4. Backward Compatibility
- All existing functionality preserved
- Standard mode works exactly as before
- Agent mode is opt-in via `--agent` flag

## Implementation Details

### Agent Integration Points

1. **LLM Client** (`sam3.agent.client_llm`):
   - Sends multimodal requests to LLM server
   - Converts images to base64
   - Uses OpenAI-compatible API

2. **SAM3 Client** (`sam3.agent.client_sam3`):
   - Wraps SAM3 processor
   - Formats outputs for agent consumption
   - Removes overlapping masks
   - Sorts by confidence

3. **Agent Core** (`sam3.agent.agent_core`):
   - Orchestrates LLM-SAM3 conversation
   - Manages tool calls
   - Prunes message history to conserve context

### RLE Mask Handling

Agent mode outputs use RLE (Run-Length Encoding) format:
- Agent saves masks as RLE strings
- `agent_inference.py` decodes RLE to binary masks using `sam3.train.masks_ops.rle_decode`
- Masks are resized to match image dimensions if needed
- Converted to boolean numpy arrays for compatibility

### Error Handling

- Agent failures fall back gracefully (log error, return False)
- Standard mode remains unaffected by agent code
- Import errors handled with fallback definitions

## Testing Recommendations

### Test Standard Mode
```bash
# Should work exactly as before
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir test_data/input \
    --out_top_dir test_data/output_standard \
    --device cuda
```

### Test Agent Mode
```bash
# Requires vLLM server running
python auto_annotate/sam3_auto_annotate.py \
    --input_top_dir test_data/input \
    --out_top_dir test_data/output_agent \
    --device cuda \
    --agent
```

### Verify Outputs
- Compare output formats between standard and agent modes
- Check NPZ mask files are identical in structure
- Verify JSON metadata follows same schema
- Ensure passenger detection works in both modes

## Dependencies

### Standard Mode (No Changes)
- torch
- numpy
- opencv-python (cv2)
- Pillow
- tqdm
- sam3 (core package)

### Agent Mode (Additional)
- openai (for LLM client)
- pycocotools (for RLE mask decoding)
- vLLM server (running separately)
- sam3.agent.* modules

## Notes

1. **LLM Server Requirement**: Agent mode requires a running vLLM server. The server must be started separately before running the script.

2. **Performance**: Agent mode is significantly slower than standard mode due to iterative LLM calls. Recommended for complex queries or high-precision requirements.

3. **Cost**: If using API-based LLMs (GPT, Gemini), agent mode will incur API costs per image.

4. **Prompts**: The person class prompt can be customized in `agent_inference.py:153` if needed.

5. **Output Directory**: Agent mode creates additional subdirectories under output for agent artifacts (sam3_output/, agent_output/).

## Future Enhancements

Possible improvements:
- Make person prompt configurable via CLI
- Add support for custom class prompts
- Cache agent results to avoid re-processing
- Add agent mode batch processing optimization
- Support for additional LLM providers (Anthropic, Google, etc.)
