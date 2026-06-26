#!/usr/bin/env python3
"""
Compare trajectory evaluation with ground truth from traj.txt

This script directly loads the ground truth from traj.txt (c2w format)
and compares it with the estimated trajectory.
"""

import argparse
import numpy as np
from pathlib import Path

from extract_trajectory import load_saved_params, extract_estimated_poses_from_params
from evaluate_trajectory import evaluate_trajectory


def pose_matrix_from_quaternion(pvec):
    """Convert quaternion to 4x4 pose matrix.
    
    Args:
        pvec: [tx, ty, tz, qx, qy, qz, qw]
        
    Returns:
        pose: 4x4 pose matrix (c2w)
    """
    from scipy.spatial.transform import Rotation
    
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose


def load_traj_txt(traj_path: str) -> np.ndarray:
    """Load ground truth trajectory from traj.txt file.
    
    Supports two formats:
    1. Replica format: 16 values per line (4x4 matrix flattened)
    2. TUM format: timestamp tx ty tz qx qy qz qw
    
    Args:
        traj_path: Path to traj.txt or groundtruth.txt file
        
    Returns:
        poses: (num_frames, 4, 4) array of c2w poses
    """
    print(f"Loading ground truth from: {traj_path}")
    poses = []
    
    with open(traj_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        values = line.split()
        
        # Detect format based on number of values
        if len(values) == 16:
            # Replica format: 4x4 matrix flattened
            c2w = np.array(list(map(float, values))).reshape(4, 4)
            poses.append(c2w)
        elif len(values) == 8:
            # TUM format: timestamp tx ty tz qx qy qz qw
            pose_vec = np.array(list(map(float, values[1:])))  # Skip timestamp
            c2w = pose_matrix_from_quaternion(pose_vec)
            poses.append(c2w)
        else:
            print(f"Warning: Unexpected format with {len(values)} values, skipping line")
            continue
    
    poses = np.stack(poses, axis=0)
    print(f"Loaded {len(poses)} ground truth poses (c2w)")
    return poses


def main():
    parser = argparse.ArgumentParser(
        description='Compare trajectory with ground truth from traj.txt')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory containing params.npz')
    parser.add_argument('--traj_txt', type=str, required=True,
                       help='Path to traj.txt file')
    parser.add_argument('--method_name', type=str, default='SplaTAM-CB-KNN',
                       help='Method name for labeling')
    parser.add_argument('--plot_unaligned', action='store_true',
                       help='Force plot unaligned trajectory even if error is large')
    parser.add_argument('--no_plot_unaligned', action='store_true',
                       help='Never plot unaligned trajectory')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    params_path = experiment_dir / "params.npz"
    
    print("\n" + "="*70)
    print("TRAJECTORY COMPARISON WITH TRAJ.TXT")
    print("="*70)
    print(f"\nExperiment: {experiment_dir}")
    print(f"GT Source: {args.traj_txt}")
    
    # Load parameters
    params = load_saved_params(params_path)
    
    # Extract estimated poses (c2w to match traj.txt)
    estimated_poses = extract_estimated_poses_from_params(params, return_c2w=True)
    
    # Load ground truth from traj.txt (already c2w)
    gt_poses = load_traj_txt(args.traj_txt)
    
    # Truncate to match shorter length
    min_len = min(len(estimated_poses), len(gt_poses))
    print(f"\nUsing {min_len} frames (estimated: {len(estimated_poses)}, gt: {len(gt_poses)})")
    
    estimated_poses = estimated_poses[:min_len]
    gt_poses = gt_poses[:min_len]
    
    # Determine plot_unaligned setting
    plot_unaligned = None
    if args.plot_unaligned:
        plot_unaligned = True
    elif args.no_plot_unaligned:
        plot_unaligned = False
    
    # Evaluate
    print("\n" + "="*70)
    evaluate_trajectory(
        estimated_poses=estimated_poses,
        gt_poses=gt_poses,
        output_path=experiment_dir,
        method_name=f"{args.method_name} (vs traj.txt)",
        plot_unaligned=plot_unaligned
    )
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
