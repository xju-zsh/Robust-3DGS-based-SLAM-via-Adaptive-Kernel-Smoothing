import faiss
import faiss.contrib.torch_utils
import torch


def torch_3d_knn(pts, num_knn, method="l2"):
    # Initialize FAISS index
    if method == "l2":
        index = faiss.IndexFlatL2(pts.shape[1])
    elif method == "cosine":
        index = faiss.IndexFlatIP(pts.shape[1])
    else:
        raise NotImplementedError(f"Method: {method}")

    # Convert FAISS index to GPU
    if pts.get_device() != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add points to index and compute distances
    index.add(pts)
    distances, indices = index.search(pts, num_knn)
    return distances, indices
    

def calculate_neighbors(params, variables, time_idx, num_knn=20):
    import time
    
    # Record KNN computation start time
    knn_start_time = time.time()
    
    if time_idx is None:
        pts = params['means3D'].detach()
    else:
        pts = params['means3D'][:, :, time_idx].detach()
    
    # Record KNN search start time
    knn_search_start_time = time.time()
    neighbor_dist, neighbor_indices = torch_3d_knn(pts.contiguous(), num_knn)
    knn_search_time = time.time() - knn_search_start_time
    
    # Record weight computation start time
    weight_calc_start_time = time.time()
    neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
    weight_calc_time = time.time() - weight_calc_start_time
    
    variables["neighbor_indices"] = neighbor_indices.long().contiguous()
    variables["neighbor_weight"] = neighbor_weight.float().contiguous()
    variables["neighbor_dist"] = neighbor_dist.float().contiguous()
    
    # Record total KNN computation time
    total_knn_time = time.time() - knn_start_time
    
    # Initialize knn_time_stats if not present in variables
    if "knn_time_stats" not in variables:
        variables["knn_time_stats"] = {
            "total_time": 0.0,
            "search_time": 0.0,
            "weight_calc_time": 0.0,
            "count": 0
        }
    
    # Update KNN timing statistics
    variables["knn_time_stats"]["total_time"] += total_knn_time
    variables["knn_time_stats"]["search_time"] += knn_search_time
    variables["knn_time_stats"]["weight_calc_time"] += weight_calc_time
    variables["knn_time_stats"]["count"] += 1
    
    # Print current KNN computation time
    if variables["knn_time_stats"]["count"] % 100 == 0:  # Print every 100 calls
        avg_total = variables["knn_time_stats"]["total_time"] / variables["knn_time_stats"]["count"]
        avg_search = variables["knn_time_stats"]["search_time"] / variables["knn_time_stats"]["count"]
        avg_weight = variables["knn_time_stats"]["weight_calc_time"] / variables["knn_time_stats"]["count"]
        print(f"\nKNN computation stats (call count: {variables['knn_time_stats']['count']})")
        print(f"Average total computation time: {avg_total*1000:.2f} ms")
        print(f"Average search time: {avg_search*1000:.2f} ms ({avg_search/avg_total*100:.1f}%)")
        print(f"Average weight computation time: {avg_weight*1000:.2f} ms ({avg_weight/avg_total*100:.1f}%)")
    
    return variables