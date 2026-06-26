import torch
import pytorch3d.ops as ops

def compute_local_density(xyz, K=8, radius=None):
    """
    Compute local density of a point cloud.

    Args:
        xyz: torch.Tensor, 3D coordinates of the point cloud, shape (N, 3)
        K: int, number of K-nearest neighbors
        radius: float, optional search radius

    Returns:
        torch.Tensor: local density per point
    """
    with torch.no_grad():
        # Compute K-nearest neighbors
        knn_result = ops.knn_points(
            xyz.unsqueeze(0),
            xyz.unsqueeze(0),
            K=K,
            return_sorted=True
        )
        dists = knn_result.dists.squeeze(0)  # (N, K)
        
        # Compute local density (inverse of mean distance)
        density = 1.0 / (dists.mean(dim=1) + 1e-6)
        return density

def get_adaptive_k(density, base_k=8, min_k=2, max_k=16):
    """
    Compute adaptive K values based on local density.

    Args:
        density: torch.Tensor, local density values
        base_k: int, base K value (unused, kept for backward compatibility)
        min_k: int, minimum K value (for sparse regions)
        max_k: int, maximum K value (for dense regions)

    Returns:
        torch.Tensor: adaptive K per point, range [min_k, max_k]

    Note:
        Points with lowest density use min_k, highest density use max_k;
        intermediate density points have K linearly distributed between min_k and max_k
    """
    # Normalize density to [0, 1]
    norm_density = (density - density.min()) / (density.max() - density.min() + 1e-6)
    
    # Map normalized density linearly to [min_k, max_k]
    # Higher density -> larger K (smoother)
    # Lower density -> smaller K (preserve detail)
    adaptive_k = min_k + (norm_density * (max_k - min_k))
    adaptive_k = adaptive_k.round()
    
    # Ensure within valid range
    adaptive_k = torch.clamp(adaptive_k, min_k, max_k)
    return adaptive_k.long()

def get_smoothed_attribute(attribute, xyz, K=8, dropout=0.5, variables=None, 
                          use_adaptive_k=True, multi_scale=False, 
                          min_k=2, max_k=16):
    """
    Random-sampling smoothing using adaptive K-nearest neighbors for point cloud data.

    Args:
        attribute: torch.Tensor, point cloud attribute to smooth
        xyz: torch.Tensor, 3D coordinates of the point cloud, shape (N, 3)
        K: int, base number of K-nearest neighbors
        dropout: float, random dropout ratio, range [0, 1]
        variables: dict, dictionary for storing timing statistics
        use_adaptive_k: bool, whether to use adaptive K values
        multi_scale: bool, whether to use multi-scale smoothing
        min_k: int, minimum K value (for adaptive K)
        max_k: int, maximum K value (for adaptive K)

    Returns:
        torch.Tensor: smoothed attribute values
    """
    import time
    
    # Record function start time
    start_time = time.time()
    
    if K <= 1:
        return attribute
    
    with torch.no_grad():
        # Record adaptive K computation start time
        adaptive_k_start_time = time.time()
        
        # Compute local density and adaptive K values
        if use_adaptive_k:
            density = compute_local_density(xyz, K=K)
            adaptive_k = get_adaptive_k(density, base_k=K, min_k=min_k, max_k=max_k)
        else:
            adaptive_k = torch.full((xyz.shape[0],), K, device=xyz.device, dtype=torch.long)
        
        # Record adaptive K computation time
        adaptive_k_time = time.time() - adaptive_k_start_time
        
        # Record KNN computation start time
        knn_start_time = time.time()
        
        # Multi-scale processing
        if multi_scale:
            scales = [1.0, 0.5, 0.25]  # Multiple scale levels
            smoothed = torch.zeros_like(attribute)
            weights = torch.tensor(scales, device=xyz.device)
            
            for scale_idx, scale in enumerate(scales):
                # Scale K values for current scale
                current_k = (adaptive_k.float() * scale).long()
                current_k = torch.clamp(current_k, min=2)  # Ensure K is at least 2
                max_current_k = current_k.max().item()
                
                # Compute K-nearest neighbors for current scale
                nearest_k_idx = ops.knn_points(
                    xyz.unsqueeze(0),
                    xyz.unsqueeze(0),
                    K=max_current_k,
                ).idx.squeeze()
                
                # Apply dropout and aggregation
                if dropout > 0 and dropout < 1:
                    k_after_dropout = (current_k.float() * (1 - dropout)).long()
                    k_after_dropout = torch.clamp(k_after_dropout, min=1)
                    
                    scale_smoothed = torch.zeros_like(attribute)
                    for i in range(len(xyz)):
                        k_i = k_after_dropout[i].item()
                        if k_i > 1:
                            idx_i = nearest_k_idx[i, :current_k[i]]
                            select_idx_i = idx_i[torch.randperm(len(idx_i))[:k_i]]
                            scale_smoothed[i] = attribute[select_idx_i].mean(dim=0)
                        else:
                            scale_smoothed[i] = attribute[i]
                else:
                    scale_smoothed = torch.zeros_like(attribute)
                    for i in range(len(xyz)):
                        idx_i = nearest_k_idx[i, :current_k[i]]
                        scale_smoothed[i] = attribute[idx_i].mean(dim=0)
                
                smoothed += scale_smoothed * weights[scale_idx]
            
            # Normalize weights
            ret = smoothed / weights.sum()
        else:
            # Single-scale processing (optimized version)
            max_k = adaptive_k.max().item()
            
            # Compute K-nearest neighbors
            nearest_k_idx = ops.knn_points(
                xyz.unsqueeze(0),
                xyz.unsqueeze(0),
                K=max_k,
            ).idx.squeeze()
            
            # Record KNN computation time
            knn_time = time.time() - knn_start_time
            
            # Record attribute aggregation start time
            attr_start_time = time.time()
            
            # Apply dropout and adaptive K
            if dropout > 0 and dropout < 1:
                k_after_dropout = (adaptive_k.float() * (1 - dropout)).long()
                k_after_dropout = torch.clamp(k_after_dropout, min=1)
                
                ret = torch.zeros_like(attribute)
                for i in range(len(xyz)):
                    k_i = k_after_dropout[i].item()
                    if k_i > 1 and adaptive_k[i] > 1:
                        idx_i = nearest_k_idx[i, :adaptive_k[i]]
                        select_idx_i = idx_i[torch.randperm(len(idx_i))[:k_i]]
                        ret[i] = attribute[select_idx_i].mean(dim=0)
                    else:
                        ret[i] = attribute[i]
            else:
                ret = torch.zeros_like(attribute)
                for i in range(len(xyz)):
                    if adaptive_k[i] > 1:
                        idx_i = nearest_k_idx[i, :adaptive_k[i]]
                        ret[i] = attribute[idx_i].mean(dim=0)
                    else:
                        ret[i] = attribute[i]
            
            # Record attribute aggregation time
            attr_time = time.time() - attr_start_time
        
        # Record total time
        total_time = time.time() - start_time
        
        # Update statistics
        if variables is not None:
            if "smoothing_time_stats" not in variables:
                variables["smoothing_time_stats"] = {
                    "total_time": 0.0,
                    "knn_time": 0.0,
                    "attr_time": 0.0,
                    "adaptive_k_time": 0.0,
                    "count": 0,
                    "use_adaptive_k": use_adaptive_k
                }
            
            variables["smoothing_time_stats"]["total_time"] += total_time
            if not multi_scale:
                variables["smoothing_time_stats"]["knn_time"] += knn_time
                variables["smoothing_time_stats"]["attr_time"] += attr_time
            variables["smoothing_time_stats"]["adaptive_k_time"] += adaptive_k_time
            variables["smoothing_time_stats"]["count"] += 1
            
            # Print statistics every 100 calls
            if variables["smoothing_time_stats"]["count"] % 100 == 0:
                count = variables["smoothing_time_stats"]["count"]
                avg_total = variables["smoothing_time_stats"]["total_time"] / count
                avg_adaptive_k = variables["smoothing_time_stats"]["adaptive_k_time"] / count
                
                print(f"\nPoint cloud smoothing stats (call count: {count}, adaptive K: {use_adaptive_k})")
                print(f"Average total time: {avg_total*1000:.2f} ms")
                print(f"Average adaptive K computation time: {avg_adaptive_k*1000:.2f} ms ({avg_adaptive_k/avg_total*100:.1f}%)")
                
                if not multi_scale:
                    avg_knn = variables["smoothing_time_stats"]["knn_time"] / count
                    avg_attr = variables["smoothing_time_stats"]["attr_time"] / count
                    print(f"Average KNN time: {avg_knn*1000:.2f} ms ({avg_knn/avg_total*100:.1f}%)")
                    print(f"Average attribute aggregation time: {avg_attr*1000:.2f} ms ({avg_attr/avg_total*100:.1f}%)")
        
        return ret
