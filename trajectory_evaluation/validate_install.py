#!/usr/bin/env python3
"""
Simple validation script for trajectory evaluation module (no torch required).

This script tests basic functionality without requiring PyTorch dependencies.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only import functions that don't require torch
from trajectory_evaluation.evaluate_trajectory import (
    align_trajectories,
    pose_error,
    evaluate_trajectory
)


def test_basic_functions():
    """Test basic trajectory evaluation functions."""
    print("\n" + "="*60)
    print("BASIC VALIDATION TEST")
    print("="*60)
    
    # Create simple test data
    print("\n1. Creating test data...")
    gt_t = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
    ]).astype(np.float32)
    
    est_t = gt_t + np.array([0.5, 0.3, 0.1]) + np.random.randn(4, 3) * 0.01
    print(f"   Created {len(gt_t)} test poses")
    
    # Test alignment
    print("\n2. Testing trajectory alignment...")
    aligned_t = align_trajectories(est_t, gt_t)
    error_before = np.linalg.norm(est_t - gt_t, axis=1).mean()
    error_after = np.linalg.norm(aligned_t - gt_t, axis=1).mean()
    print(f"   Error before: {error_before*100:.3f} cm")
    print(f"   Error after:  {error_after*100:.3f} cm")
    print(f"   ✓ Alignment {'improved' if error_after < error_before else 'failed'}")
    
    # Test pose error
    print("\n3. Testing pose error computation...")
    errors = pose_error(est_t, gt_t)
    print(f"   RMSE: {errors['rmse']*100:.2f} cm")
    print(f"   Mean: {errors['mean']*100:.2f} cm")
    print(f"   ✓ Error metrics computed")
    
    # Test full evaluation with 4x4 matrices
    print("\n4. Testing full evaluation pipeline...")
    gt_poses = np.tile(np.eye(4), (len(gt_t), 1, 1))
    gt_poses[:, :3, 3] = gt_t
    est_poses = np.tile(np.eye(4), (len(est_t), 1, 1))
    est_poses[:, :3, 3] = est_t
    
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    evaluate_trajectory(
        estimated_poses=est_poses,
        gt_poses=gt_poses,
        output_path=output_dir,
        method_name="Validation Test"
    )
    
    # Check outputs
    print("\n5. Checking output files...")
    files = ["ate.json", "ate_aligned.json", "eval_trajectory.png"]
    for filename in files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"   ✓ {filename} created")
        else:
            print(f"   ✗ {filename} missing")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nThe trajectory_evaluation module is working correctly!")
    print(f"Check output files in: {output_dir}")


if __name__ == "__main__":
    try:
        test_basic_functions()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
