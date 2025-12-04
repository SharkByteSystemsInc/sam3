# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3 (Segment Anything Model 3) is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts (points, boxes, masks). The key innovation is the ability to exhaustively segment all instances of open-vocabulary concepts specified by text phrases or exemplars.

**Key capabilities:**
- Text-prompted segmentation (open vocabulary)
- Visual prompting (boxes, points, masks)
- Video object tracking and segmentation
- Instance-level exhaustive segmentation
- Interactive refinement

**Model architecture:**
- 848M parameters total
- Decoupled detector-tracker design
- Shared vision encoder (ViT-based, 1008x1008 resolution)
- DETR-based detector conditioned on text, geometry, and image exemplars
- SAM 2-inherited tracker architecture for video segmentation

## Commands

### Installation
```bash
# Basic installation
pip install -e .

# For running example notebooks
pip install -e ".[notebooks]"

# For development (includes formatting tools)
pip install -e ".[dev]"

# For training
pip install -e ".[train,dev]"
```

### Code Formatting
```bash
# Format all code using ufmt (uses ruff-api formatter)
ufmt format .
```

### Testing
```bash
# Run tests (when tests exist)
pytest tests/
```

### Model Checkpoints
SAM 3 checkpoints require authentication:
1. Request access at https://huggingface.co/facebook/sam3
2. Authenticate: `huggingface-cli login` or use environment variable

### Training & Evaluation

**Train on Roboflow dataset:**
```bash
# Local single GPU
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Local multi-GPU
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4

# On cluster (SLURM)
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1
```

**Train on ODinW13 dataset:**
```bash
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml --use-cluster 0 --num-gpus 1
```

**Evaluate on SA-Co/Gold benchmarks:**
```bash
# Before running, edit sam3/train/configs/eval_base.yaml with image/annotation paths

# MetaCLIP captioner NPs
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_metaclip_nps.yaml --use-cluster 0 --num-gpus 1

# SA-1B captioner NPs
python sam3/train/train.py -c configs/gold_image_evals/sam3_gold_image_sa1b_nps.yaml --use-cluster 0 --num-gpus 1

# Other subsets: attributes, crowded, wiki_common, fg_food, fg_sports
```

**Evaluate on SA-Co/Silver benchmarks:**
```bash
python sam3/train/train.py -c configs/silver_image_evals/sam3_silver_image_bdd100k.yaml --use-cluster 0 --num-gpus 1
```

**Offline cgF1 evaluation:**
```bash
# For SA-Co/Gold (with 3 GT annotations)
python scripts/eval/standalone_cgf1.py \
  --pred_file /path/to/coco_predictions_segm.json \
  --gt_files /path/to/gold_metaclip_merged_a_release_test.json \
             /path/to/gold_metaclip_merged_b_release_test.json \
             /path/to/gold_metaclip_merged_c_release_test.json

# For SA-Co/Silver (single GT annotation)
python scripts/eval/standalone_cgf1.py \
  --pred_file /path/to/coco_predictions_segm.json \
  --gt_files /path/to/silver_bdd100k_merged_test.json
```

## Architecture & Code Structure

### Core Model Components (sam3/model/)

**Vision Encoder (`vitdet.py`, `encoder.py`):**
- ViT backbone: 1024 embed_dim, 32 depth, 16 heads
- Input size: 1008x1008, patch size: 14x14
- Uses RoPE (Rotary Position Embeddings) and interpolated RoPE
- Global attention at layers 7, 15, 23, 31
- Returns multi-scale features via neck

**Text Encoder (`text_encoder_ve.py`, `tokenizer_ve.py`):**
- Based on CLIP-style architecture
- 1024 width, 16 heads, 24 layers
- Projects to 256-dim shared embedding space
- BPE tokenizer with 16k vocab

**Visual-Language Backbone (`vl_combiner.py`):**
- `SAM3VLBackbone`: Combines visual (ViT neck) and text encoders
- Shared embedding space (256-dim)

**Transformer Encoder/Decoder (`decoder.py`, `encoder.py`):**
- `TransformerEncoderFusion`: 6 layers, fuses image and text features
- `TransformerDecoder`: 6 layers, 200 queries, box refinement
- Includes "presence token" for discriminating similar prompts
- Deformable Attention with Content-aware (DAC)
- Box Relative Position Bias (RPB) with log encoding

**Geometry Encoders (`geometry_encoders.py`):**
- `SequenceGeometryEncoder`: Encodes points, boxes, masks
- 3 layers with cross-attention to image features
- Adds positional encoding for spatial context

**Segmentation Head (`maskformer_segmentation.py`):**
- `UniversalSegmentationHead`: Generates instance masks
- `PixelDecoder`: 3 upsampling stages for high-res masks
- Cross-attends to prompt embeddings

**Tracker (`sam3_tracker_base.py`, `sam3_tracking_predictor.py`):**
- Memory-based tracking with mask memory encoder
- 7 memory frames (configurable)
- RoPE attention for temporal modeling
- 4-layer transformer encoder

### Main Model Classes

**Image Model (`sam3_image.py`):**
- `Sam3Image`: Main detector for single-image inference
- `Sam3ImageOnVideoMultiGPU`: Detector adapted for per-frame video inference
- Outputs: masks, boxes, scores, embeddings

**Video Model (`sam3_video_inference.py`, `sam3_video_predictor.py`):**
- `Sam3VideoInferenceWithInstanceInteractivity`: Full video model (detector + tracker)
- `Sam3VideoPredictorMultiGPU`: Multi-GPU wrapper with session management
- Association logic: IoU-based matching, NMS, hotstart delay
- Temporal disambiguation options (controllable heuristics)

**Model Builders (`model_builder.py`):**
- `build_sam3_image_model()`: Builds image detector
- `build_sam3_video_model()`: Builds full video model (detector + tracker)
- `build_sam3_video_predictor()`: Builds multi-GPU video predictor
- `build_tracker()`: Builds standalone tracker

### Processors & Interfaces

**Image Processor (`sam3_image_processor.py`):**
- `Sam3Processor`: High-level API for image inference
- Methods: `set_image()`, `set_text_prompt()`, `set_box_prompt()`

**Video Predictor (`sam3_video_predictor.py`):**
- Request-based API with session management
- Request types: `start_session`, `add_prompt`, `add_points`, etc.

**SAM 1 Task Support (`sam1_task_predictor.py`):**
- `SAM3InteractiveImagePredictor`: Interactive instance segmentation
- Compatible with SAM 1 API for backward compatibility

### Agent System (sam3/agent/)

The agent system provides an MLLM-based interface for complex segmentation tasks:
- `agent_core.py`: Core agentic loop
- `client_sam3.py`: SAM 3 client interface
- `client_llm.py`: LLM client interface
- `helpers/`: Utilities for masks, boxes, visualization, zoom-in, etc.

### Evaluation (sam3/eval/)

**Metrics:**
- `cgf1_eval.py`: Concept-granularity F1 (cgF1) - main metric for SA-Co
- `coco_eval.py`: Standard COCO AP/AR metrics
- `hota_eval_toolkit/`: HOTA metric for video tracking
- `teta_eval_toolkit/`: TETA metric for video tracking

**SA-Co Evaluators:**
- `saco_veval_evaluators.py`: Video evaluation for SA-Co/VEval
- `saco_veval_eval.py`: Standalone script for offline evaluation

**COCO Writers:**
- `coco_writer.py`: Converts predictions to COCO format
- `coco_reindex.py`: Reindexes COCO annotations

### Training (sam3/train/)

Training infrastructure uses Hydra for config management:
- `train/train.py`: Main training script (also used for eval when `trainer.mode = val`)
- `train/configs/`: YAML configs for datasets and experiments
- `train/matcher.py`: Hungarian matcher for training
- `train/losses.py`: Loss functions (focal, dice, giou, etc.)

**Key config files:**
- `eval_base.yaml`: Base config with paths (edit before running evals)
- `gold_image_evals/`: Configs for SA-Co/Gold subsets
- `silver_image_evals/`: Configs for SA-Co/Silver subsets
- `saco_video_evals/`: Configs for SA-Co/VEval
- `odinw13/`: Configs for ODinW13 dataset
- `roboflow_v100/`: Configs for Roboflow-100 dataset

### Performance Optimizations (sam3/perflib/)

Custom CUDA/triton kernels for performance:
- `masks_ops.py`: Fast mask operations
- `connected_components.py`: Connected components analysis
- `nms.py`: Non-maximum suppression
- `associate_det_trk.py`: Fast detection-track association
- `fa3.py`: FlashAttention 3 integration
- `compile.py`: torch.compile utilities

## Important Implementation Details

### Coordinate Systems
- **Boxes**: Normalized [x, y, w, h] format in range [0, 1]
- **Image resolution**: Models expect 1008x1008 input
- **Patch size**: 14x14 pixels
- **Feature stride**: 14 pixels

### Data Format
- **Annotations**: Extended COCO format with `text_input` field per image
- **RLE masks**: Standard COCO RLE format via pycocotools
- **Video annotations**: YTVIS-like format with `video_np_pairs` field
- **Negative prompts**: Present in `images` but not in `annotations`

### Model Loading
- Checkpoints use nested dict structure: `checkpoint['model']`
- Detector keys: `detector.*`
- Tracker keys: `tracker.*`
- Load from HuggingFace by default if no checkpoint path provided

### Compilation
- TF32 automatically enabled for Ampere+ GPUs (major >= 8)
- torch.compile support via `compile_mode="default"` parameter
- Selective compilation of ViT backbone and pixel decoder

### Activation Checkpointing
- Enabled by default in transformer encoder/decoder for memory efficiency
- Controlled via `use_act_checkpoint` parameters

### Video Processing
- Supports both JPEG folders and MP4 files
- Session-based API for stateful video processing
- Memory management: 7 memory frames, configurable keep-alive

### Training Notes
- Uses submitit for cluster job submission
- Supports SLURM job arrays for multi-dataset training (e.g., 100 Roboflow datasets)
- Checkpoint saving controlled by `skip_checkpointing` flag
- TensorBoard logging to `experiment_log_dir/tensorboard/`

## Dataset Information

### SA-Co/Gold
7 image subsets with triple annotations per datapoint:
- MetaCLIP captioner NPs
- SA-1B captioner NPs
- Attributes
- Crowded Scenes
- Wiki-Common1K
- Wiki-Food/Drink
- Wiki-Sports Equipment

**Metric**: cgF1 (concept-granularity F1)
**Evaluation**: Oracle setting - best of 3 annotations per sample

### SA-Co/Silver
10 image subsets with single annotation:
- BDD100k, DROID, Ego4D, MyFoodRepo-273, GeoDE
- iNaturalist-2017, National Gallery of Art
- SA-V, YT-Temporal-1B, Fathomnet

**Note**: Performance may be underestimated due to single annotation

### SA-Co/VEval
3 video domains (val + test splits):
- SA-V (24fps frames)
- YT-Temporal-1B (6fps frames) - **Note**: Frame alignment issues possible
- SmartGlasses (6fps frames)

**Metrics**: cgF1, pHOTA (promptable HOTA)

## Key Files to Reference

When implementing features:
- Model architecture: `sam3/model_builder.py` (canonical model construction)
- Image inference: `sam3/model/sam3_image_processor.py`
- Video inference: `sam3/model/sam3_video_predictor.py`
- Geometry encoding: `sam3/model/geometry_encoders.py`
- Training: `sam3/train/train.py`

When debugging:
- Check `sam3/__init__.py` for package version
- See example notebooks in `examples/` for usage patterns
- Review eval scripts in `scripts/eval/` for metric computation

## Development Notes

- Code formatting uses `ufmt` with `ruff-api` formatter (not black directly)
- Python 3.12+ required (originally 3.8+ but 3.12 recommended)
- PyTorch 2.7+, CUDA 12.6+ for GPU support
- No explicit test suite currently exists
- Package structure uses setuptools with dynamic version from `sam3.__version__`
