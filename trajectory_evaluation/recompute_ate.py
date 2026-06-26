#!/usr/bin/env python3
"""
Recompute ATE metrics for existing experiments using the fixed RMSE calculation.

This script reads params.npz and recomputes the ATE metrics using the corrected
RMSE formula, updating the eval/ate.json and eval/ate_aligned.json files.
"""

import argparse
import numpy as np
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.append('.')

from utils.slam_external import build_rotation
from utils.eval_helpers import align


def recompute_ate_from_params(params_path, eval_dir):
    """Recompute ATE metrics from params.npz using fixed RMSE calculation."""
    
    print(f"\nLoading parameters from: {params_path}")
    params = np.load(params_path, allow_pickle=True)
    
    # Extract estimated trajectory
    cam_unnorm_rots = params['cam_unnorm_rots']
    cam_trans = params['cam_trans']
    gt_w2c_all_frames = params['gt_w2c_all_frames']
    
    num_frames = cam_unnorm_rots.shape[-1]
    print(f"Number of frames: {num_frames}")
    
    # Build estimated trajectory
    first_frame_w2c = torch.eye(4).cuda().float()
    latest_est_w2c_list = [first_frame_w2c]
    valid_gt_w2c_list = [torch.from_numpy(gt_w2c_all_frames[0]).cuda().float()]
    
    for idx in range(1, num_frames):
        # Check if GT pose is valid
        gt_pose = torch.from_numpy(gt_w2c_all_frames[idx]).cuda().float()
        if torch.isnan(gt_pose).sum() > 0:
            continue
        
        # Build estimated pose
        interm_cam_rot = torch.from_numpy(cam_unnorm_rots[..., idx]).cuda().float()
        interm_cam_rot = F.normalize(interm_cam_rot)
        interm_cam_trans = torch.from_numpy(cam_trans[..., idx]).cuda().float()
        
        intermrel_w2c = torch.eye(4).cuda().float()
        intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
        intermrel_w2c[:3, 3] = interm_cam_trans
        
        latest_est_w2c_list.append(intermrel_w2c)
        valid_gt_w2c_list.append(gt_pose)
    
    print(f"Valid poses: {len(valid_gt_w2c_list)} / {num_frames}")
    
    # Convert w2c to c2w for proper trajectory comparison (same as eval_helpers.py)
    # c2w gives us the camera center positions in world coordinates
    latest_est_c2w_list = [torch.linalg.inv(w2c) for w2c in latest_est_w2c_list]
    valid_gt_c2w_list = [torch.linalg.inv(w2c) for w2c in valid_gt_w2c_list]
    
    # Extract translation components (camera center positions)
    gt_traj_pts = torch.stack([pose[:3, 3] for pose in valid_gt_c2w_list]).detach().cpu().numpy()
    est_traj_pts = torch.stack([pose[:3, 3] for pose in latest_est_c2w_list]).detach().cpu().numpy()
    
    # Compute unaligned errors
    unaligned_errors = np.linalg.norm(est_traj_pts - gt_traj_pts, axis=1)
    ate_rmse_unaligned = np.sqrt(np.mean(unaligned_errors ** 2))
    
    ate_unaligned_dict = {
        "compared_pose_pairs": len(valid_gt_c2w_list),
        "rmse": float(ate_rmse_unaligned),
        "mean": float(np.mean(unaligned_errors)),
        "median": float(np.median(unaligned_errors)),
        "std": float(np.std(unaligned_errors)),
        "min": float(np.min(unaligned_errors)),
        "max": float(np.max(unaligned_errors))
    }
    
    # Compute aligned errors
    _, _, aligned_errors = align(gt_traj_pts.T, est_traj_pts.T)
    ate_rmse_aligned = np.sqrt(np.mean(aligned_errors ** 2))
    
    ate_aligned_dict = {
        "compared_pose_pairs": len(valid_gt_c2w_list),
        "rmse": float(ate_rmse_aligned),
        "mean": float(np.mean(aligned_errors)),
        "median": float(np.median(aligned_errors)),
        "std": float(np.std(aligned_errors)),
        "min": float(np.min(aligned_errors)),
        "max": float(np.max(aligned_errors))
    }
    
    # Save to eval/ directory
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(exist_ok=True, parents=True)
    
    with open(eval_dir / "ate.json", "w") as f:
        json.dump(ate_unaligned_dict, f, indent=2)
    
    with open(eval_dir / "ate_aligned.json", "w") as f:
        json.dump(ate_aligned_dict, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("RECOMPUTED ATE METRICS (Fixed RMSE + c2w format)")
    print("="*60)
    print(f"Compared pose pairs: {len(valid_gt_c2w_list)}")
    print(f"\nWithout Alignment:")
    print(f"  ATE RMSE:   {ate_rmse_unaligned*100:.2f} cm")
    print(f"  Mean:       {ate_unaligned_dict['mean']*100:.2f} cm")
    print(f"  Median:     {ate_unaligned_dict['median']*100:.2f} cm")
    print(f"  Std:        {ate_unaligned_dict['std']*100:.2f} cm")
    print(f"\nWith Alignment:")
    print(f"  ATE RMSE:   {ate_rmse_aligned*100:.2f} cm")
    print(f"  Mean:       {ate_aligned_dict['mean']*100:.2f} cm")
    print(f"  Median:     {ate_aligned_dict['median']*100:.2f} cm")
    print(f"  Std:        {ate_aligned_dict['std']*100:.2f} cm")
    print("="*60)
    
    print(f"\n✓ Updated files:")
    print(f"  - {eval_dir}/ate.json")
    print(f"  - {eval_dir}/ate_aligned.json")
    
    return ate_aligned_dict


def main():
    parser = argparse.ArgumentParser(
        description='Recompute ATE metrics with fixed RMSE calculation')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory containing params.npz')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    params_path = experiment_dir / "params.npz"
    eval_dir = experiment_dir / "eval"
    
    if not params_path.exists():
        print(f"❌ Error: {params_path} not found")
        return 1
    
    try:
        recompute_ate_from_params(params_path, eval_dir)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
