"""
Utility functions to extract camera trajectories from SplaTAM-CB-KNN parameters and datasets.

This module provides functions to:
1. Load saved parameters from SplaTAM-CB-KNN experiments
2. Extract estimated camera poses from parameters
3. Extract ground truth poses from datasets
"""

import os
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_saved_params(params_path: Union[str, Path]) -> dict:
    """Load saved SplaTAM parameters from .npz file.
    
    Args:
        params_path: Path to the saved parameters (.npz file or directory containing params.npz)
        
    Returns:
        dict: Dictionary containing SplaTAM parameters
    """
    params_path = Path(params_path)
    
    if params_path.is_dir():
        params_path = params_path / "params.npz"
    
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    print(f"Loading parameters from: {params_path}")
    params = dict(np.load(str(params_path), allow_pickle=True))
    
    return params


def build_rotation(r):
    """Build rotation matrix from quaternion.
    
    Args:
        r: quaternion (4,) in format [w, x, y, z] or [x, y, z, w]
        
    Returns:
        Rotation matrix (3, 3)
    """
    if isinstance(r, np.ndarray):
        r = torch.from_numpy(r).float()
    
    # Ensure it's 1D
    if r.dim() > 1:
        r = r.squeeze()
    
    # Check dimension
    if r.shape[0] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got shape {r.shape}")
    
    # Normalize quaternion
    norm = torch.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3])
    q = r / norm

    R = torch.zeros((3, 3), dtype=torch.float32)

    # Assuming quaternion format is [w, x, y, z]
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - w*z)
    R[0, 2] = 2 * (x*z + w*y)
    R[1, 0] = 2 * (x*y + w*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - w*x)
    R[2, 0] = 2 * (x*z - w*y)
    R[2, 1] = 2 * (y*z + w*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    
    return R


def extract_estimated_poses_from_params(params: dict, device='cpu', return_c2w=False) -> np.ndarray:
    """Extract estimated camera poses from SplaTAM parameters.
    
    SplaTAM stores camera poses as:
    - cam_unnorm_rots: quaternion rotations (4, num_frames)
    - cam_trans: translations (3, num_frames)
    
    These are relative poses (w2c - world to camera).
    
    Args:
        params: Dictionary containing SplaTAM parameters
        device: Device to use for computation ('cpu' or 'cuda')
        return_c2w: If True, return c2w matrices instead of w2c
        
    Returns:
        poses: (num_frames, 4, 4) array of camera poses (w2c or c2w)
    """
    print("Extracting estimated camera poses from parameters...")
    
    # Load camera parameters
    cam_unnorm_rots = params['cam_unnorm_rots']
    cam_trans = params['cam_trans']
    
    # Convert to torch tensors
    if isinstance(cam_unnorm_rots, np.ndarray):
        cam_unnorm_rots = torch.from_numpy(cam_unnorm_rots).float().to(device)
    if isinstance(cam_trans, np.ndarray):
        cam_trans = torch.from_numpy(cam_trans).float().to(device)
    
    # Get first frame w2c (usually identity or stored separately)
    if 'w2c' in params:
        first_frame_w2c = params['w2c']
        if isinstance(first_frame_w2c, np.ndarray):
            first_frame_w2c = torch.from_numpy(first_frame_w2c).float().to(device)
    else:
        # Assume identity for first frame
        first_frame_w2c = torch.eye(4, dtype=torch.float32, device=device)
    
    # Handle different dimension formats
    # cam_unnorm_rots can be (4, num_frames) or (1, 4, num_frames)
    if cam_unnorm_rots.dim() == 3:
        cam_unnorm_rots = cam_unnorm_rots.squeeze(0)  # (1, 4, N) -> (4, N)
    
    if cam_trans.dim() == 3:
        cam_trans = cam_trans.squeeze(0)  # (1, 3, N) -> (3, N)
    
    num_frames = cam_unnorm_rots.shape[-1]
    print(f"Number of frames: {num_frames}")
    print(f"Quaternion shape: {cam_unnorm_rots.shape}")
    print(f"Translation shape: {cam_trans.shape}")
    
    # Extract poses for all frames
    poses = []
    poses.append(first_frame_w2c.cpu().numpy())
    
    for idx in range(1, num_frames):
        # Get quaternion and translation
        quat = cam_unnorm_rots[:, idx]  # Shape: (4,)
        trans = cam_trans[:, idx]  # Shape: (3,)
        
        # Normalize quaternion
        quat = F.normalize(quat.unsqueeze(0), dim=1).squeeze(0)
        
        # Build transformation matrix
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        w2c[:3, :3] = build_rotation(quat)
        w2c[:3, 3] = trans
        
        poses.append(w2c.cpu().numpy())
    
    poses = np.stack(poses, axis=0)
    
    # Convert to c2w if requested (matching final_recon.py behavior)
    if return_c2w:
        print("Converting w2c to c2w (camera-to-world) for visualization...")
        c2w_poses = []
        for w2c in poses:
            c2w = np.linalg.inv(w2c)
            c2w_poses.append(c2w)
        poses = np.stack(c2w_poses, axis=0)
        print(f"Extracted {len(poses)} estimated poses (c2w) with shape {poses.shape}")
    else:
        print(f"Extracted {len(poses)} estimated poses (w2c) with shape {poses.shape}")
    
    return poses


def extract_gt_poses_from_params(params: dict, return_c2w=False) -> np.ndarray:
    """Extract ground truth poses from saved parameters.
    
    Args:
        params: Dictionary containing SplaTAM parameters
        return_c2w: If True, convert from w2c to c2w format
        
    Returns:
        gt_poses: (num_frames, 4, 4) array of ground truth poses (w2c or c2w)
    """
    print("Extracting ground truth poses from parameters...")
    
    if 'gt_w2c_all_frames' not in params:
        raise KeyError("Ground truth poses not found in parameters. "
                      "Make sure the experiment was run with ground truth available.")
    
    gt_poses = params['gt_w2c_all_frames']
    
    if not isinstance(gt_poses, np.ndarray):
        gt_poses = np.array(gt_poses)
    
    print(f"Extracted {len(gt_poses)} ground truth poses (w2c) with shape {gt_poses.shape}")
    
    # Convert to c2w if requested
    if return_c2w:
        print("Converting w2c to c2w (camera-to-world)...")
        c2w_poses = []
        for w2c in gt_poses:
            c2w = np.linalg.inv(w2c)
            c2w_poses.append(c2w)
        gt_poses = np.stack(c2w_poses, axis=0)
        print(f"Converted to c2w format")
    
    return gt_poses


def extract_gt_poses_from_dataset(dataset) -> np.ndarray:
    """Extract ground truth poses from a dataset object.
    
    Args:
        dataset: SplaTAM dataset object (e.g., ReplicaDataset, TUMDataset, etc.)
        
    Returns:
        gt_poses: (num_frames, 4, 4) array of ground truth poses
    """
    print("Extracting ground truth poses from dataset...")
    
    num_frames = len(dataset)
    gt_poses = []
    
    for idx in range(num_frames):
        _, _, _, pose = dataset[idx]
        
        # Convert to w2c if needed
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        
        # Check if it's c2w or w2c
        # In most SplaTAM datasets, poses are c2w, so we need to invert
        w2c = np.linalg.inv(pose)
        gt_poses.append(w2c)
    
    gt_poses = np.stack(gt_poses, axis=0)
    print(f"Extracted {len(gt_poses)} ground truth poses with shape {gt_poses.shape}")
    
    return gt_poses


def convert_w2c_to_c2w(poses: np.ndarray) -> np.ndarray:
    """Convert world-to-camera poses to camera-to-world poses.
    
    Args:
        poses: (num_frames, 4, 4) array of w2c poses
        
    Returns:
        c2w_poses: (num_frames, 4, 4) array of c2w poses
    """
    c2w_poses = []
    for pose in poses:
        c2w = np.linalg.inv(pose)
        c2w_poses.append(c2w)
    return np.stack(c2w_poses, axis=0)


def convert_c2w_to_w2c(poses: np.ndarray) -> np.ndarray:
    """Convert camera-to-world poses to world-to-camera poses.
    
    Args:
        poses: (num_frames, 4, 4) array of c2w poses
        
    Returns:
        w2c_poses: (num_frames, 4, 4) array of w2c poses
    """
    return convert_w2c_to_c2w(poses)  # Same operation (inverse)
