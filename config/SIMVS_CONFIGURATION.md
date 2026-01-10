# SimVS Configuration Guide

This guide explains how to configure NoPo-Avatar for SimVS-style training with inconsistent views.

## Overview

The SimVS implementation adds three main features:
1. **Inconsistent Image Loading**: Load pre-generated inconsistent images from disk
2. **Harmonization**: Mix original and inconsistent images based on reference masks
3. **Loss Weighting**: Apply per-view weights to the projection loss (SimVS-style)

## Dataset Configuration

### Basic Setup

Add the following fields to your dataset configuration (e.g., `config/dataset/thuman.yaml`):

```yaml
# Enable loading of inconsistent images
load_inconsistent_images: true
inconsistent_images_path: datasets/thuman2.0_inconsistent  # Path to inconsistent images
inconsistent_frame_name_format: "frame_{:06d}"  # Format for frame names

# Enable harmonization
use_harmonization: true
harmonization_default_reference_indices: [0]  # Default reference view indices
```

### Folder Structure

The inconsistent images should be organized as follows:

```
inconsistent_images_path/
├── train/
│   ├── subject_001/
│   │   └── images/
│   │       ├── frame_000000.png
│   │       ├── frame_000001.png
│   │       └── ...
│   ├── subject_002/
│   │   └── images/
│   │       └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Configuration Options

#### `load_inconsistent_images` (bool)
- **Default**: `false`
- **Description**: Enable/disable loading inconsistent images from disk
- **Required**: `true` to use SimVS features

#### `inconsistent_images_path` (Path | None)
- **Default**: `null`
- **Description**: Base path to the inconsistent images folder
- **Required**: Must be set when `load_inconsistent_images: true`
- **Example**: `datasets/thuman2.0_inconsistent`

#### `inconsistent_frame_name_format` (str)
- **Default**: `"frame_{:06d}"`
- **Description**: Format string for frame names (uses Python string formatting)
- **Examples**:
  - `"frame_{:06d}"` → `frame_000000.png`
  - `"view_{:04d}"` → `view_0000.png`
  - `"{:d}"` → `0.png`

#### `use_harmonization` (bool)
- **Default**: `false`
- **Description**: Enable harmonization (mixing original and inconsistent images)
- **Behavior**:
  - **Context (input)**: Reference views use original images, non-reference views use inconsistent images
  - **Target (output)**: All views use original images (ground truth)

#### `harmonization_default_reference_indices` (list[int] | None)
- **Default**: `null` (which defaults to `[0]`)
- **Description**: Default reference view indices when `reference_mask` is not provided in the batch
- **Examples**:
  - `[0]` → First view is reference
  - `[0, 2]` → Views 0 and 2 are reference
  - `null` → Uses `reference_mask` from batch if available, otherwise defaults to `[0]`

## Loss Configuration

The projection loss automatically applies per-view weights when `reference_mask` is available in the batch. The weighting strategy is configured in `src/model/model_wrapper.py`:

```python
# Default: SimVSWeighting with max_weight=5.0
self.loss_weighting = get_loss_weighting(strategy="simvs", max_weight=5.0)
```

### Weighting Strategy

- **Reference frames**: Weight = 0 (they're clean/ground truth)
- **Non-reference frames**: Weighted by distance from input frames
- **Farther frames**: Get higher weight (up to `max_weight=5.0`)

## Example Configuration Files

### Minimal Example

```yaml
# config/dataset/thuman_simvs.yaml
defaults:
  - base_dataset
  - view_sampler: uniform

name: thuman
roots: [datasets/thuman2.0]

# ... other dataset config ...

# SimVS settings
load_inconsistent_images: true
inconsistent_images_path: datasets/thuman2.0_inconsistent
use_harmonization: true
harmonization_default_reference_indices: [0]
```

### Full Training Example

```yaml
# config/experiment/simvs_training.yaml
defaults:
  - /model/encoder: noposplat
  - /model/decoder: splatting_cuda
  - /dataset: thuman_simvs  # Your SimVS-enabled dataset config
  - /loss: [mse, lpips, ssim, projection]

wandb:
  project: noposplat_simvs
  name: simvs_experiment
  mode: online

# ... rest of training config ...
```

## Running Training

```bash
python src/main.py \
  experiment=simvs_training \
  dataset.inconsistent_images_path=/path/to/inconsistent/images \
  dataset.use_harmonization=true
```

## Troubleshooting

### Inconsistent images not found
- **Symptom**: Training continues but uses original images as fallback
- **Solution**: Check that:
  1. `inconsistent_images_path` is correct
  2. Folder structure matches expected format
  3. Frame names match `inconsistent_frame_name_format`
  4. File extensions are `.png`, `.jpg`, or `.jpeg`

### Reference mask not working
- **Symptom**: Harmonization doesn't respect reference views
- **Solution**: 
  1. Ensure `reference_mask` is in the batch (check dataset)
  2. Or set `harmonization_default_reference_indices` explicitly
  3. Check that `use_harmonization: true` is set

### Loss weights not applied
- **Symptom**: Projection loss behaves the same with/without weights
- **Solution**:
  1. Ensure `reference_mask` is in `batch["context"]`
  2. Check that projection loss is included in loss config
  3. Verify `compute_loss_weights()` is called in `model_wrapper.py`

## Disabling SimVS Features

To disable SimVS features and return to standard training:

```yaml
load_inconsistent_images: false
use_harmonization: false
```

The code will fall back to standard behavior (no inconsistent images, uniform loss weighting).
