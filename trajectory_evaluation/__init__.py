"""
Trajectory Evaluation Module for SplaTAM-CB-KNN

This module provides tools for evaluating camera trajectory accuracy by comparing 
estimated camera poses with ground truth poses. It includes:
- Trajectory alignment using Horn's method
- ATE (Absolute Trajectory Error) computation
- 2D trajectory visualization
"""

# Lazy imports to avoid requiring torch at import time
def __getattr__(name):
    if name in ['align_trajectories', 'pose_error', 'plot_2d', 'evaluate_trajectory', 'align']:
        from .evaluate_trajectory import (
            align_trajectories,
            pose_error,
            plot_2d,
            evaluate_trajectory,
            align
        )
        return locals()[name]
    
    elif name in ['extract_estimated_poses_from_params', 'extract_gt_poses_from_dataset', 
                  'load_saved_params']:
        from .extract_trajectory import (
            extract_estimated_poses_from_params,
            extract_gt_poses_from_dataset,
            load_saved_params
        )
        return locals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'align_trajectories',
    'pose_error',
    'plot_2d',
    'evaluate_trajectory',
    'align',
    'extract_estimated_poses_from_params',
    'extract_gt_poses_from_dataset',
    'load_saved_params'
]
