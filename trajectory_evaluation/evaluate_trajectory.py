"""
Trajectory evaluation functions adapted from Gaussian-SLAM for SplaTAM-CB-KNN.

This module provides functions to:
1. Align estimated trajectories with ground truth using Horn's method (SVD-based)
2. Compute pose errors (ATE - Absolute Trajectory Error)
3. Visualize trajectories in 2D
4. Generate comprehensive trajectory evaluation reports
5. Smooth trajectories for visualization
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


class NumpyFloatValuesEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy float types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    
    This function finds the optimal rigid transformation (rotation + translation)
    that aligns the model trajectory to the data trajectory using SVD.

    Args:
        model: first trajectory (3xn) - estimated positions
        data: second trajectory (3xn) - ground truth positions

    Returns:
        rot: rotation matrix (3x3)
        trans: translation vector (3x1)
        trans_error: translational error per point (1xn)
    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def align_trajectories(t_pred: np.ndarray, t_gt: np.ndarray):
    """Align predicted trajectory to ground truth trajectory.
    
    Args:
        t_pred: (n, 3) predicted translations
        t_gt: (n, 3) ground truth translations
        
    Returns:
        t_align: (n, 3) aligned translations
    """
    t_align = np.matrix(t_pred).transpose()
    R, t, _ = align(t_align, np.matrix(t_gt).transpose())
    t_align = R * t_align + t
    t_align = np.asarray(t_align).T
    return t_align


def pose_error(t_pred: np.ndarray, t_gt: np.ndarray, align_flag=False):
    """Compute trajectory error metrics.
    
    Args:
        t_pred: (n, 3) predicted translations
        t_gt: (n, 3) ground truth translations
        align_flag: whether trajectories are already aligned (not used, kept for compatibility)
        
    Returns:
        dict: Dictionary containing error metrics:
            - compared_pose_pairs: number of poses compared
            - rmse: Root Mean Square Error
            - mean: mean error
            - median: median error
            - std: standard deviation
            - min: minimum error
            - max: maximum error
    """
    n = t_pred.shape[0]
    trans_error = np.linalg.norm(t_pred - t_gt, axis=1)
    return {
        "compared_pose_pairs": n,
        "rmse": np.sqrt(np.dot(trans_error, trans_error) / n),
        "mean": np.mean(trans_error),
        "median": np.median(trans_error),
        "std": np.std(trans_error),
        "min": np.min(trans_error),
        "max": np.max(trans_error)
    }


def smooth_trajectory(trajectory, sigma=2.0):
    """Smooth a 3D trajectory using Gaussian filtering.
    
    This function applies Gaussian smoothing independently to each coordinate (X, Y, Z)
    to reduce noise and make the trajectory visualization smoother.
    
    Args:
        trajectory: (n, 3) array of 3D points
        sigma: standard deviation for Gaussian kernel (higher = smoother)
        
    Returns:
        smoothed_trajectory: (n, 3) array of smoothed 3D points
    """
    if len(trajectory) < 3:
        return trajectory
    
    smoothed = np.zeros_like(trajectory)
    for i in range(3):  # Smooth X, Y, Z independently
        smoothed[:, i] = gaussian_filter1d(trajectory[:, i], sigma=sigma, mode='nearest')
    
    return smoothed


def plot_2d(pts, ax=None, color="green", label="None", title="3D Trajectory in 2D"):
    """Plot 3D trajectory projected onto 2D (X-Y plane).
    
    Args:
        pts: (n, 3) array of 3D points
        ax: matplotlib axis object (optional)
        color: line color
        label: legend label
        title: plot title
        
    Returns:
        ax: matplotlib axis object
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(pts[:, 0], pts[:, 1], color=color, label=label, linewidth=1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return ax


def evaluate_trajectory(estimated_poses: np.ndarray, gt_poses: np.ndarray, 
                       output_path: Path, method_name="SplaTAM-CB-KNN", 
                       plot_unaligned=None, smooth_sigma=2.0):
    """Evaluate estimated trajectory against ground truth.
    
    This function:
    1. Aligns estimated trajectory to ground truth
    2. Computes ATE metrics (with and without alignment)
    3. Saves metrics to JSON files
    4. Generates and saves trajectory visualization plot
    
    Args:
        estimated_poses: (n, 4, 4) estimated camera poses (c2w or w2c)
        gt_poses: (n, 4, 4) ground truth camera poses
        output_path: directory to save results
        method_name: name of the method for plot labels
        plot_unaligned: Whether to plot unaligned trajectory. 
                       If None, auto-decide based on error magnitude.
                       If True, always plot. If False, never plot.
        smooth_sigma: sigma parameter for Gaussian smoothing of aligned trajectory.
                     Set to 0 to disable smoothing. Default is 2.0.
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Truncate the ground truth trajectory if needed
    if gt_poses.shape[0] > estimated_poses.shape[0]:
        print(f"Warning: GT has {gt_poses.shape[0]} poses but estimated has {estimated_poses.shape[0]}. Truncating GT.")
        gt_poses = gt_poses[:estimated_poses.shape[0]]
    
    # Filter out invalid poses (NaN or Inf)
    valid = ~np.any(np.isnan(gt_poses) | np.isinf(gt_poses), axis=(1, 2))
    valid = valid & ~np.any(np.isnan(estimated_poses) | np.isinf(estimated_poses), axis=(1, 2))
    
    print(f"Valid poses: {valid.sum()} / {len(valid)}")
    
    gt_poses = gt_poses[valid]
    estimated_poses = estimated_poses[valid]

    # Extract translation vectors (camera positions)
    gt_t = gt_poses[:, :3, 3]
    estimated_t = estimated_poses[:, :3, 3]
    
    # Align estimated trajectory to ground truth
    estimated_t_aligned = align_trajectories(estimated_t, gt_t)
    
    # Compute ATE metrics
    ate = pose_error(estimated_t, gt_t)
    ate_aligned = pose_error(estimated_t_aligned, gt_t)

    # Save metrics to JSON
    with open(str(output_path / "ate.json"), "w") as f:
        f.write(json.dumps(ate, cls=NumpyFloatValuesEncoder, indent=2))

    with open(str(output_path / "ate_aligned.json"), "w") as f:
        f.write(json.dumps(ate_aligned, cls=NumpyFloatValuesEncoder, indent=2))

    # Generate visualization
    ate_rmse, ate_rmse_aligned = ate["rmse"], ate_aligned["rmse"]
    
    # Auto-decide whether to plot unaligned trajectory
    # If unaligned error is more than 10x the aligned error, skip it
    if plot_unaligned is None:
        plot_unaligned = (ate_rmse < ate_rmse_aligned * 10) and (ate_rmse < 1.0)  # 1 meter threshold
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plot_2d(gt_t, ax=ax, label="GT", color="green")
    
    if plot_unaligned:
        # Plot unaligned trajectory if error is reasonable
        ax = plot_2d(estimated_t, ax, 
                     label=f"ate-rmse ({method_name}): {ate_rmse*100:.2f} cm", 
                     color="orange")
    
    # Apply smoothing to aligned trajectory for visualization
    estimated_t_aligned_smooth = estimated_t_aligned
    if smooth_sigma > 0:
        estimated_t_aligned_smooth = smooth_trajectory(estimated_t_aligned, sigma=smooth_sigma)
        print(f"Applied Gaussian smoothing to aligned trajectory (sigma={smooth_sigma})")
    
    # Always plot aligned trajectory (smoothed version)
    ax = plot_2d(estimated_t_aligned_smooth, ax,
                 label=f"ate-rmse ({method_name} aligned): {ate_rmse_aligned*100:.2f} cm", 
                 color="lightskyblue")
    
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path / "eval_trajectory.png"), dpi=300)
    plt.close()
    
    if not plot_unaligned:
        print(f"\nNote: Unaligned trajectory not plotted (error too large: {ate_rmse*100:.2f} cm)")
        print(f"      Only showing GT and aligned trajectory for clarity.")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAJECTORY EVALUATION RESULTS")
    print("="*60)
    print(f"Method: {method_name}")
    print(f"Compared pose pairs: {ate['compared_pose_pairs']}")
    print(f"\nWithout Alignment:")
    print(f"  ATE RMSE:   {ate_rmse * 100:.2f} cm")
    print(f"  Mean:       {ate['mean'] * 100:.2f} cm")
    print(f"  Median:     {ate['median'] * 100:.2f} cm")
    print(f"  Std:        {ate['std'] * 100:.2f} cm")
    print(f"  Min:        {ate['min'] * 100:.2f} cm")
    print(f"  Max:        {ate['max'] * 100:.2f} cm")
    print(f"\nWith Alignment:")
    print(f"  ATE RMSE:   {ate_rmse_aligned * 100:.2f} cm")
    print(f"  Mean:       {ate_aligned['mean'] * 100:.2f} cm")
    print(f"  Median:     {ate_aligned['median'] * 100:.2f} cm")
    print(f"  Std:        {ate_aligned['std'] * 100:.2f} cm")
    print(f"  Min:        {ate_aligned['min'] * 100:.2f} cm")
    print(f"  Max:        {ate_aligned['max'] * 100:.2f} cm")
    print("="*60)
    print(f"Results saved to: {output_path}")
    print(f"  - {output_path / 'ate.json'}")
    print(f"  - {output_path / 'ate_aligned.json'}")
    print(f"  - {output_path / 'eval_trajectory.png'}")
    print("="*60 + "\n")
