#!/usr/bin/env python3
"""
Evaluate trajectory using GT from params.npz (same as training script).

This script replicates the exact evaluation logic from eval_helpers.py
but uses the trajectory_evaluation module's RMSE calculation for consistency.
"""

import argparse
import numpy as np
from pathlib import Path

from extract_trajectory import load_saved_params, extract_estimated_poses_from_params, extract_gt_poses_from_params
from evaluate_trajectory import evaluate_trajectory


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trajectory using GT from params.npz')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory containing params.npz')
    parser.add_argument('--method_name', type=str, default='SplaTAM-CB-KNN',
                       help='Method name for labeling')
    parser.add_argument('--use_c2w', action='store_true',
                       help='Convert to c2w format (for visualization consistency)')
    parser.add_argument('--plot_unaligned', action='store_true',
                       help='Force plot unaligned trajectory')
    parser.add_argument('--no_plot_unaligned', action='store_true',
                       help='Never plot unaligned trajectory')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    params_path = experiment_dir / "params.npz"
    
    print("\n" + "="*70)
    print("TRAJECTORY EVALUATION (Using GT from params.npz)")
    print("="*70)
    print(f"\nExperiment: {experiment_dir}")
    print(f"GT Source: params.npz (gt_w2c_all_frames)")
    
    # Load parameters
    params = load_saved_params(params_path)
    
    # Extract estimated and GT poses
    # Use the same coordinate system for both
    use_c2w = args.use_c2w
    estimated_poses = extract_estimated_poses_from_params(params, return_c2w=use_c2w)
    gt_poses = extract_gt_poses_from_params(params, return_c2w=use_c2w)
    
    # Truncate to match shorter length (handle NaN frames)
    min_len = min(len(estimated_poses), len(gt_poses))
    print(f"\nUsing {min_len} frames (estimated: {len(estimated_poses)}, gt: {len(gt_poses)})")
    
    estimated_poses = estimated_poses[:min_len]
    gt_poses = gt_poses[:min_len]
    
    # Determine plot_unaligned flag
    plot_unaligned_flag = None  # Let evaluate_trajectory decide automatically
    if args.plot_unaligned:
        plot_unaligned_flag = True
    elif args.no_plot_unaligned:
        plot_unaligned_flag = False
    
    # Evaluate trajectory
    output_dir = experiment_dir
    evaluate_trajectory(
        estimated_poses=estimated_poses,
        gt_poses=gt_poses,
        output_path=output_dir,
        method_name=f"{args.method_name} (params GT)",
        plot_unaligned=plot_unaligned_flag
    )
    
    print(f"\n✓ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - {output_dir}/ate.json")
    print(f"  - {output_dir}/ate_aligned.json")
    print(f"  - {output_dir}/eval_trajectory.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
