#!/usr/bin/env python3
"""
Standalone Trajectory Evaluation Script for SplaTAM-CB-KNN

This script evaluates the trajectory from a completed SplaTAM experiment by:
1. Loading saved parameters from the experiment directory
2. Extracting estimated camera poses
3. Extracting ground truth poses (from params or dataset)
4. Computing alignment and ATE metrics
5. Generating visualization plots

Usage:
    python eval_trajectory.py --experiment_dir <path/to/experiment> [options]

Example:
    python eval_trajectory.py --experiment_dir experiments/Replica/room0 --method_name "SplaTAM-CB-KNN"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import trajectory_evaluation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_evaluation.evaluate_trajectory import evaluate_trajectory
from trajectory_evaluation.extract_trajectory import (
    load_saved_params,
    extract_estimated_poses_from_params,
    extract_gt_poses_from_params,
    convert_w2c_to_c2w
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate camera trajectory from SplaTAM-CB-KNN experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--experiment_dir', 
        type=str, 
        required=True,
        help='Path to experiment directory containing params.npz'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: same as experiment_dir)'
    )
    
    parser.add_argument(
        '--method_name',
        type=str,
        default='SplaTAM-CB-KNN',
        help='Method name for plot labels'
    )
    
    parser.add_argument(
        '--params_file',
        type=str,
        default='params.npz',
        help='Name of the parameters file'
    )
    
    parser.add_argument(
        '--use_c2w',
        action='store_true',
        help='Use c2w poses (camera-to-world) instead of w2c. This matches final_recon.py visualization.'
    )
    
    parser.add_argument(
        '--match_final_recon',
        action='store_true',
        help='Match the visualization style of final_recon.py (uses c2w poses)'
    )
    
    parser.add_argument(
        '--smooth_sigma',
        type=float,
        default=2.0,
        help='Sigma parameter for Gaussian smoothing of aligned trajectory. Higher = smoother. Set to 0 to disable.'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("SplaTAM-CB-KNN Trajectory Evaluation")
    print("="*80)
    
    # Setup paths
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    params_path = experiment_dir / args.params_file
    if not params_path.exists():
        print(f"Error: Parameters file not found: {params_path}")
        print(f"Please make sure {args.params_file} exists in the experiment directory.")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Parameters file: {params_path}")
    print(f"Output directory: {output_dir}")
    print(f"Method name: {args.method_name}")
    print()
    
    # Load parameters
    try:
        params = load_saved_params(params_path)
        print(f"✓ Successfully loaded parameters")
        print(f"  Available keys: {list(params.keys())}")
    except Exception as e:
        print(f"✗ Error loading parameters: {e}")
        sys.exit(1)
    
    # Determine if we should use c2w
    use_c2w = args.use_c2w or args.match_final_recon
    
    # Extract estimated poses
    try:
        estimated_poses = extract_estimated_poses_from_params(params, return_c2w=use_c2w)
        print(f"✓ Successfully extracted estimated poses")
    except Exception as e:
        print(f"✗ Error extracting estimated poses: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract ground truth poses
    try:
        gt_poses = extract_gt_poses_from_params(params)
        print(f"✓ Successfully extracted ground truth poses")
    except Exception as e:
        print(f"✗ Error extracting ground truth poses: {e}")
        print(f"   Ground truth poses must be saved in params['gt_w2c_all_frames']")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Convert GT to c2w if using c2w for estimated
    if use_c2w:
        print("Converting ground truth from w2c to c2w...")
        gt_poses = convert_w2c_to_c2w(gt_poses)
    
    # Run evaluation
    try:
        print("\nRunning trajectory evaluation...")
        evaluate_trajectory(
            estimated_poses=estimated_poses,
            gt_poses=gt_poses,
            output_path=output_dir,
            method_name=args.method_name,
            smooth_sigma=args.smooth_sigma
        )
        print(f"\n✓ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
