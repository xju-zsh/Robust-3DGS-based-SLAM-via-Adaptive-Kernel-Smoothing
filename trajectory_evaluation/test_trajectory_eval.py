#!/usr/bin/env python3
"""
Test script for trajectory evaluation module.

This script creates synthetic data to test the trajectory evaluation functions.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_evaluation.evaluate_trajectory import (
    align_trajectories,
    pose_error,
    plot_2d,
    evaluate_trajectory
)


def create_synthetic_trajectory(num_frames=100):
    """Create synthetic camera trajectory data for testing."""
    print("Creating synthetic trajectory data...")
    
    # Create a circular trajectory
    t = np.linspace(0, 2 * np.pi, num_frames)
    radius = 2.0
    
    # Ground truth trajectory (perfect circle)
    gt_positions = np.zeros((num_frames, 3))
    gt_positions[:, 0] = radius * np.cos(t)
    gt_positions[:, 1] = radius * np.sin(t)
    gt_positions[:, 2] = 0.1 * t  # slight upward motion
    
    # Estimated trajectory (circle with noise)
    noise_level = 0.05
    est_positions = gt_positions + np.random.randn(num_frames, 3) * noise_level
    
    # Convert positions to 4x4 transformation matrices
    gt_poses = []
    est_poses = []
    
    for i in range(num_frames):
        # Ground truth pose
        gt_pose = np.eye(4)
        gt_pose[:3, 3] = gt_positions[i]
        # Add rotation (camera looking at center)
        direction = -gt_positions[i] / np.linalg.norm(gt_positions[i][:2] + 1e-6)
        gt_pose[:3, 0] = np.array([direction[1], -direction[0], 0])
        gt_pose[:3, 1] = np.array([0, 0, 1])
        gt_pose[:3, 2] = np.array([direction[0], direction[1], 0])
        gt_poses.append(gt_pose)
        
        # Estimated pose
        est_pose = np.eye(4)
        est_pose[:3, 3] = est_positions[i]
        est_pose[:3, :3] = gt_pose[:3, :3]  # Same rotation for simplicity
        est_poses.append(est_pose)
    
    gt_poses = np.stack(gt_poses, axis=0)
    est_poses = np.stack(est_poses, axis=0)
    
    print(f"Created {num_frames} synthetic poses")
    print(f"GT trajectory shape: {gt_poses.shape}")
    print(f"Est trajectory shape: {est_poses.shape}")
    
    return gt_poses, est_poses


def test_alignment():
    """Test trajectory alignment function."""
    print("\n" + "="*60)
    print("TEST 1: Trajectory Alignment")
    print("="*60)
    
    # Create simple test data
    gt_t = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
    ]).astype(np.float32)
    
    # Estimated with offset and noise
    est_t = gt_t + np.array([0.5, 0.3, 0.1]) + np.random.randn(4, 3) * 0.01
    
    # Align
    aligned_t = align_trajectories(est_t, gt_t)
    
    # Check alignment improved
    error_before = np.linalg.norm(est_t - gt_t, axis=1).mean()
    error_after = np.linalg.norm(aligned_t - gt_t, axis=1).mean()
    
    print(f"Average error before alignment: {error_before*100:.3f} cm")
    print(f"Average error after alignment: {error_after*100:.3f} cm")
    print(f"Improvement: {(error_before - error_after)*100:.3f} cm")
    
    if error_after < error_before:
        print("✓ TEST PASSED: Alignment reduced error")
    else:
        print("✗ TEST FAILED: Alignment did not reduce error")
    
    return error_after < error_before


def test_pose_error():
    """Test pose error computation."""
    print("\n" + "="*60)
    print("TEST 2: Pose Error Computation")
    print("="*60)
    
    gt_t = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ]).astype(np.float32)
    
    est_t = gt_t + 0.05  # 5cm error
    
    errors = pose_error(est_t, gt_t)
    
    print(f"RMSE: {errors['rmse']*100:.2f} cm")
    print(f"Mean: {errors['mean']*100:.2f} cm")
    print(f"Median: {errors['median']*100:.2f} cm")
    print(f"Std: {errors['std']*100:.2f} cm")
    print(f"Min: {errors['min']*100:.2f} cm")
    print(f"Max: {errors['max']*100:.2f} cm")
    
    # Check if RMSE is approximately 5cm (with sqrt(3) factor for 3D)
    expected_error = 0.05 * np.sqrt(3)
    actual_error = errors['rmse']
    
    if abs(actual_error - expected_error) < 0.01:
        print(f"✓ TEST PASSED: RMSE is approximately {expected_error*100:.2f} cm")
        return True
    else:
        print(f"✗ TEST FAILED: Expected ~{expected_error*100:.2f} cm, got {actual_error*100:.2f} cm")
        return False


def test_full_evaluation():
    """Test full evaluation pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Full Evaluation Pipeline")
    print("="*60)
    
    # Create synthetic data
    gt_poses, est_poses = create_synthetic_trajectory(num_frames=50)
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    try:
        evaluate_trajectory(
            estimated_poses=est_poses,
            gt_poses=gt_poses,
            output_path=output_dir,
            method_name="Test Method"
        )
        
        # Check if output files exist
        files_created = []
        for filename in ["ate.json", "ate_aligned.json", "eval_trajectory.png"]:
            filepath = output_dir / filename
            if filepath.exists():
                files_created.append(filename)
                print(f"✓ Created: {filename}")
            else:
                print(f"✗ Missing: {filename}")
        
        if len(files_created) == 3:
            print("✓ TEST PASSED: All output files created")
            return True
        else:
            print(f"✗ TEST FAILED: Only {len(files_created)}/3 files created")
            return False
            
    except Exception as e:
        print(f"✗ TEST FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRAJECTORY EVALUATION MODULE - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Trajectory Alignment", test_alignment),
        ("Pose Error Computation", test_pose_error),
        ("Full Evaluation Pipeline", test_full_evaluation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ TEST CRASHED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
