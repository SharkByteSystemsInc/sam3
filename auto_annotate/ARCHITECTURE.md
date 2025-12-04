# SAM3 Auto-Annotation Architecture

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                   sam3_auto_annotate.py                          │
│                   (Main Orchestrator)                            │
│                                                                   │
│  • CLI argument parsing                                          │
│  • Image file discovery                                          │
│  • Progress tracking                                             │
│  • Metadata I/O                                                  │
│  • Passenger detection                                           │
│  • Mask encoding/overlapping removal                             │
└────────┬────────────────────────────────────┬────────────────────┘
         │                                    │
         │ imports                            │ imports
         ▼                                    ▼
┌─────────────────────────┐     ┌──────────────────────────────────┐
│   model_inference.py     │     │     agent_inference.py           │
│   (Standard Mode)        │     │     (Agent Mode)                 │
│                          │     │                                  │
│  • SAM3Annotator class   │     │  • setup_agent_components()      │
│  • segment_image()       │     │  • run_agent_inference()         │
│  • run_standard_inference│     │  • Uses sam3.agent.* modules     │
└────────┬─────────────────┘     └────────┬─────────────────────────┘
         │                                │
         │ imports                        │ imports
         ▼                                ▼
┌────────────────────────────────────────────────────────────────┐
│                        utils.py                                 │
│                   (Shared Utilities)                            │
│                                                                  │
│  • ModelResult class                                            │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Standard Mode Flow

```
Input Image (.jpg) + Metadata (.json)
    ↓
sam3_auto_annotate.py:process_image()
    ↓
model_inference.py:run_standard_inference()
    ↓
SAM3Annotator.segment_image() [for each class: car, license plate, person]
    ↓
List[ModelResult] with masks and scores
    ↓
detect_passengers() [identifies persons inside cars]
    ↓
filter_out_overlapping_masks() [removes overlaps]
    ↓
encode_binary_masks() [creates single NPZ file]
    ↓
Output: .jpg, .json (metadata), .npz (masks)
```

### Agent Mode Flow

```
Input Image (.jpg) + Metadata (.json)
    ↓
sam3_auto_annotate.py:process_image() [use_agent=True]
    ↓
agent_inference.py:run_agent_inference()
    ↓
setup_agent_components() [initialize LLM + SAM3 clients]
    ↓
For each class (car, license plate, person):
    ↓
    run_agent_inference_for_class()
        ↓
        sam3.agent.inference.run_single_image_inference()
            ↓
            ┌─────────────────────────────────────────┐
            │   Agent Iterative Loop                  │
            │                                          │
            │  1. LLM generates tool call             │
            │     ↓                                    │
            │  2. segment_phrase(text_prompt)         │
            │     ↓                                    │
            │  3. SAM3 returns masks                  │
            │     ↓                                    │
            │  4. LLM examines results                │
            │     ↓                                    │
            │  5. examine_each_mask (optional)        │
            │     ↓                                    │
            │  6. select_masks_and_return             │
            └─────────────────────────────────────────┘
        ↓
        Decode RLE masks to numpy arrays
    ↓
List[ModelResult] with masks and scores
    ↓
detect_passengers() [identifies persons inside cars]
    ↓
filter_out_overlapping_masks() [removes overlaps]
    ↓
encode_binary_masks() [creates single NPZ file]
    ↓
Output: .jpg, .json (metadata), .npz (masks)
```

## Class Diagram

```
┌────────────────────────┐
│    ModelResult         │
├────────────────────────┤
│ + class_name: str      │
│ + confidence: float    │
│ + mask: np.ndarray     │
└────────────────────────┘
            △
            │ uses
            │
┌───────────┴────────────────────────┐
│                                    │
│                                    │
┌────────────────────────┐  ┌───────────────────────┐
│   SAM3Annotator        │  │  Agent Components     │
├────────────────────────┤  ├───────────────────────┤
│ - model                │  │ + send_generate_req() │
│ - processor            │  │ + call_sam_service()  │
│ - device               │  │ + llm_config: dict    │
│ - logger               │  └───────────────────────┘
├────────────────────────┤
│ + segment_image()      │
│   → (masks, scores)    │
└────────────────────────┘
```

## Execution Modes Comparison

| Aspect | Standard Mode | Agent Mode |
|--------|---------------|------------|
| **Speed** | Fast (~1-2 sec/image) | Slow (~30-60 sec/image) |
| **Accuracy** | Good for simple objects | Better for complex queries |
| **Dependencies** | SAM3 only | SAM3 + vLLM server |
| **Prompts** | Direct class names | Natural language descriptions |
| **Person Class** | "person" | "all people, except those on posters" |
| **Iterative** | No | Yes (LLM refines results) |
| **Output Format** | GVA metadata | GVA metadata (identical) |

## Configuration Points

### Command Line Arguments
```
sam3_auto_annotate.py
├── --input_top_dir (required)
├── --out_top_dir (required)
├── --device [cuda|cpu]
├── --agent [flag]
├── --llm_server_url (agent mode)
├── --llm_model (agent mode)
├── --override_previous_annotations [flag]
└── --log_level [DEBUG|INFO|WARNING|ERROR]
```

### Code Customization Points

1. **Confidence Threshold** (model_inference.py:45)
   ```python
   confidence_threshold = 0.4
   ```

2. **Person Class Prompt** (agent_inference.py:153)
   ```python
   if class_name == "person":
       text_prompt = "all people, except those displayed on the posters"
   ```

3. **Classes to Segment** (sam3_auto_annotate.py:373)
   ```python
   classes_to_segment = ["car", "license plate", "person"]
   ```

4. **Passenger Detection Thresholds** (sam3_auto_annotate.py:273-275)
   ```python
   if containment_ratio > 0.60 or (containment_ratio > 0.40 and is_well_inside):
   ```

5. **LLM Default Config** (agent_inference.py:40-50)
   ```python
   llm_config = {
       "provider": "vllm",
       "model": llm_model,
       "name": llm_model.split("/")[-1]
   }
   ```

## External Dependencies

### SAM3 Package Modules
```
sam3/
├── model_builder.py
│   └── build_sam3_image_model()
├── model/
│   └── sam3_image_processor.py
│       └── Sam3Processor
└── agent/
    ├── inference.py
    │   └── run_single_image_inference()
    ├── client_llm.py
    │   └── send_generate_request()
    ├── client_sam3.py
    │   └── call_sam_service()
    └── agent_core.py
        └── agent_inference()
```

### Third-Party Libraries
```
torch          - PyTorch for model inference
numpy          - Array operations
opencv-python  - Image I/O and processing
Pillow         - Image format conversion
tqdm           - Progress bars
openai         - LLM client (agent mode)
```

## Error Handling Strategy

```
try:
    process_image()
        ↓
        try:
            if use_agent:
                run_agent_inference()
            else:
                run_standard_inference()
        except Exception:
            log error, return False
        ↓
        detect_passengers()
        filter_overlapping_masks()
        encode_masks()
        save_outputs()
except Exception:
    log error, skip image
    ↓
continue to next image
```

## Performance Considerations

### Standard Mode
- **Bottleneck**: SAM3 model inference
- **Optimization**: Batch processing possible
- **Memory**: ~8GB GPU for 1008x1008 images
- **Throughput**: ~30-60 images/minute

### Agent Mode
- **Bottleneck**: LLM round trips
- **Optimization**: Cache results, reduce iterations
- **Memory**: SAM3 (8GB) + LLM server (varies)
- **Throughput**: ~2-4 images/minute

## Testing Strategy

### Unit Tests (Recommended)
```python
# test_model_inference.py
def test_sam3_annotator_initialization()
def test_segment_image_output_format()
def test_run_standard_inference()

# test_agent_inference.py
def test_setup_agent_components()
def test_run_agent_inference_for_class()
def test_rle_decode()

# test_utils.py
def test_model_result_creation()
```

### Integration Tests
```python
# test_integration.py
def test_standard_mode_end_to_end()
def test_agent_mode_end_to_end()
def test_output_format_consistency()
def test_passenger_detection()
```

### Manual Testing Checklist
- [ ] Standard mode on sample dataset
- [ ] Agent mode on sample dataset
- [ ] Output format validation
- [ ] Error handling (missing files, corrupted images)
- [ ] LLM server connection failures
- [ ] Large dataset processing
