#!/usr/bin/env python3
"""
Test runner for SimVS implementation tests.
Run with: python tests/run_tests.py
"""
import sys
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_suite(test_module_name, test_functions):
    """Run a suite of test functions and report results."""
    print(f"\n{'='*60}")
    print(f"Running {test_module_name} tests...")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    errors = []
    
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✓ {test_name} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            errors.append((test_name, e, traceback.format_exc()))
            failed += 1
    
    print(f"\n{test_module_name} Results: {passed} passed, {failed} failed")
    return passed, failed, errors


def main():
    """Main test runner."""
    print("="*60)
    print("Running SimVS Implementation Tests")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    all_errors = []
    
    # Test suite 1: Inconsistent Image Shim
    try:
        from tests.test_inconsistent_image_shim import (
            test_load_inconsistent_image,
            test_load_inconsistent_image_not_found,
            test_load_inconsistent_views,
            test_apply_inconsistent_image_shim,
            test_apply_inconsistent_image_shim_missing_images,
        )
        
        passed, failed, errors = run_test_suite(
            "inconsistent_image_shim",
            [
                test_load_inconsistent_image,
                test_load_inconsistent_image_not_found,
                test_load_inconsistent_views,
                test_apply_inconsistent_image_shim,
                test_apply_inconsistent_image_shim_missing_images,
            ]
        )
        total_passed += passed
        total_failed += failed
        all_errors.extend(errors)
    except ImportError as e:
        print(f"✗ Failed to import inconsistent_image_shim tests: {e}")
        total_failed += 1
        all_errors.append(("import", e, traceback.format_exc()))
    
    # Test suite 2: Harmonization Shim
    try:
        from tests.test_harmonization_shim import (
            test_apply_harmonization_to_views_with_reference_mask,
            test_apply_harmonization_to_views_default_reference,
            test_apply_harmonization_to_views_no_inconsistent_images,
            test_apply_harmonization_shim,
            test_apply_harmonization_shim_batched,
        )
        
        passed, failed, errors = run_test_suite(
            "harmonization_shim",
            [
                test_apply_harmonization_to_views_with_reference_mask,
                test_apply_harmonization_to_views_default_reference,
                test_apply_harmonization_to_views_no_inconsistent_images,
                test_apply_harmonization_shim,
                test_apply_harmonization_shim_batched,
            ]
        )
        total_passed += passed
        total_failed += failed
        all_errors.extend(errors)
    except ImportError as e:
        print(f"✗ Failed to import harmonization_shim tests: {e}")
        total_failed += 1
        all_errors.append(("import", e, traceback.format_exc()))
    
    # Test suite 3: Projection Loss Weighting
    try:
        from tests.test_projection_loss_weighting import (
            test_projection_loss_without_weights,
            test_projection_loss_with_weights,
            test_projection_loss_zero_weight_reference,
            test_projection_loss_batched_weights,
        )
        
        passed, failed, errors = run_test_suite(
            "projection_loss_weighting",
            [
                test_projection_loss_without_weights,
                test_projection_loss_with_weights,
                test_projection_loss_zero_weight_reference,
                test_projection_loss_batched_weights,
            ]
        )
        total_passed += passed
        total_failed += failed
        all_errors.extend(errors)
    except ImportError as e:
        print(f"✗ Failed to import projection_loss_weighting tests: {e}")
        total_failed += 1
        all_errors.append(("import", e, traceback.format_exc()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total: {total_passed + total_failed} tests")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if all_errors:
        print("\n" + "="*60)
        print("Error Details")
        print("="*60)
        for test_name, error, trace in all_errors:
            print(f"\n{test_name}:")
            print(f"  Error: {error}")
            if len(all_errors) <= 3:  # Only show full traceback for few errors
                print(f"  Traceback:\n{trace}")
    
    if total_failed > 0:
        print("\n" + "="*60)
        print("❌ Some tests failed!")
        print("="*60)
        sys.exit(1)
    else:
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()
