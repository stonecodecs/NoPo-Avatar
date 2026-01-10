# SimVS Implementation Tests

This directory contains tests for the SimVS (Simulating World Inconsistencies for Robust View Synthesis) implementation in NoPo-Avatar.

## Test Files

### `test_inconsistent_image_shim.py`
Tests for loading inconsistent images from disk:
- Loading single inconsistent images
- Loading inconsistent images for multiple views
- Handling missing images (fallback to original)
- Full shim integration

### `test_harmonization_shim.py`
Tests for harmonization (mixing original and inconsistent images):
- Harmonization with explicit reference masks
- Default reference indices
- Batched examples
- Handling missing inconsistent images

### `test_projection_loss_weighting.py`
Tests for per-view loss weighting in projection loss:
- Loss computation without weights (original behavior)
- Loss computation with weights
- Zero-weight reference frames
- Batched weights

## Running Tests

### Run all tests:
```bash
python tests/run_tests.py
```

### Run individual test files:
```bash
python tests/test_inconsistent_image_shim.py
python tests/test_harmonization_shim.py
python tests/test_projection_loss_weighting.py
```

### Run with pytest (if installed):
```bash
pytest tests/
```

## Test Requirements

The tests require:
- `torch` (PyTorch)
- `PIL` (Pillow)
- `numpy`
- Standard library: `tempfile`, `pathlib`

All dependencies should already be available if the main project dependencies are installed.

## Test Coverage

The tests cover:
- ✅ Image loading from disk with various formats
- ✅ Handling missing files gracefully
- ✅ Reference mask handling
- ✅ Batched and unbatched examples
- ✅ Loss weighting computation
- ✅ Edge cases (missing data, invalid inputs)

## Notes

- Tests use temporary directories for file I/O tests
- Mock data is used for loss computation tests
- Tests are designed to run quickly without requiring actual dataset files
