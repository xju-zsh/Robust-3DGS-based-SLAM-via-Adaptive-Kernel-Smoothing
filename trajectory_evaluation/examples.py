#!/usr/bin/env python3
"""
Example: How to use trajectory evaluation in SplaTAM-CB-KNN experiments

This script demonstrates different ways to evaluate trajectories:
1. From saved experiment results (params.npz)
2. Directly from SplaTAM parameters during runtime
3. From custom pose arrays
"""

import sys
from pathlib import Path
import numpy as np

# Example 1: Evaluate from saved experiment
def example_evaluate_saved_experiment():
    """Evaluate trajectory from a completed SplaTAM experiment."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Evaluate Saved Experiment")
    print("="*70)
    
    # Path to your experiment directory
    experiment_dir = "experiments/Replica/room0"  # Change this to your experiment
    
    # You can either use the standalone script:
    print("\n1. Using standalone script (recommended):")
    print("   cd /path/to/SplaTAM-CB-KNN")
    print(f"   python trajectory_evaluation/eval_trajectory.py \\")
    print(f"       --experiment_dir {experiment_dir} \\")
    print(f"       --method_name 'SplaTAM-CB-KNN'")
    
    # Or use Python API:
    print("\n2. Using Python API:")
    print("""
    from trajectory_evaluation.extract_trajectory import (
        load_saved_params,
        extract_estimated_poses_from_params,
        extract_gt_poses_from_params
    )
    from trajectory_evaluation.evaluate_trajectory import evaluate_trajectory
    
    # Load parameters
    params = load_saved_params(f"{experiment_dir}/params.npz")
    
    # Extract poses
    estimated_poses = extract_estimated_poses_from_params(params)
    gt_poses = extract_gt_poses_from_params(params)
    
    # Evaluate
    evaluate_trajectory(
        estimated_poses=estimated_poses,
        gt_poses=gt_poses,
        output_path=experiment_dir,
        method_name="SplaTAM-CB-KNN"
    )
    """)


# Example 2: Add to existing SLAM script
def example_add_to_slam_script():
    """Show how to add trajectory evaluation to existing SLAM scripts."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Add to Existing SLAM Script")
    print("="*70)
    
    print("""
Add the following code at the end of your SLAM script (e.g., scripts/splatam.py):

```python
# ... your existing SLAM code ...

# After the main SLAM loop completes and parameters are saved:
if config.get('eval_trajectory', True):  # Add this config option
    print("\\nEvaluating camera trajectory...")
    
    from trajectory_evaluation.extract_trajectory import (
        extract_estimated_poses_from_params,
        extract_gt_poses_from_params
    )
    from trajectory_evaluation.evaluate_trajectory import evaluate_trajectory
    
    # Convert final parameters
    eval_params = convert_params_to_store(params)
    
    # Extract poses
    estimated_poses = extract_estimated_poses_from_params(eval_params)
    gt_poses = extract_gt_poses_from_params(eval_params)
    
    # Run evaluation
    evaluate_trajectory(
        estimated_poses=estimated_poses,
        gt_poses=gt_poses,
        output_path=output_dir,
        method_name=config.get('method_name', 'SplaTAM-CB-KNN')
    )
```

This will automatically generate trajectory evaluation results after each experiment.
    """)


# Example 3: Evaluate custom trajectory
def example_custom_trajectory():
    """Show how to evaluate custom trajectory arrays."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Evaluate Custom Trajectory")
    print("="*70)
    
    print("""
If you have your own pose arrays (not from SplaTAM parameters):

```python
import numpy as np
from trajectory_evaluation.evaluate_trajectory import evaluate_trajectory

# Your poses as (num_frames, 4, 4) arrays
# Can be either w2c or c2w, as long as both use same convention
estimated_poses = np.array([...])  # Shape: (N, 4, 4)
gt_poses = np.array([...])         # Shape: (N, 4, 4)

# Run evaluation
evaluate_trajectory(
    estimated_poses=estimated_poses,
    gt_poses=gt_poses,
    output_path="./my_results",
    method_name="My Method"
)
```

This will create:
- my_results/ate.json (error metrics without alignment)
- my_results/ate_aligned.json (error metrics with alignment)
- my_results/eval_trajectory.png (trajectory visualization)
    """)


# Example 4: Batch evaluation
def example_batch_evaluation():
    """Show how to evaluate multiple experiments."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Evaluation")
    print("="*70)
    
    print("""
Create a bash script to evaluate multiple experiments:

```bash
#!/bin/bash
# evaluate_all_experiments.sh

# Scenes to evaluate
SCENES="room0 room1 room2 office0 office1 office2 office3 office4"
DATASET="Replica"
METHOD="SplaTAM-CB-KNN"

for scene in $SCENES; do
    echo "=========================================="
    echo "Evaluating $scene"
    echo "=========================================="
    
    python trajectory_evaluation/eval_trajectory.py \\
        --experiment_dir experiments/$DATASET/$scene \\
        --method_name "$METHOD"
    
    echo ""
done

echo "All evaluations complete!"
```

Or use Python:

```python
import subprocess
from pathlib import Path

scenes = ["room0", "room1", "room2", "office0"]
base_dir = Path("experiments/Replica")

for scene in scenes:
    exp_dir = base_dir / scene
    if not exp_dir.exists():
        print(f"Skipping {scene}: directory not found")
        continue
    
    print(f"Evaluating {scene}...")
    subprocess.run([
        "python", "trajectory_evaluation/eval_trajectory.py",
        "--experiment_dir", str(exp_dir),
        "--method_name", "SplaTAM-CB-KNN"
    ])
```
    """)


# Example 5: Compare multiple methods
def example_compare_methods():
    """Show how to compare trajectories from different methods."""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Compare Multiple Methods")
    print("="*70)
    
    print("""
To compare multiple methods on the same scene:

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# List of methods and their experiment directories
methods = {
    "SplaTAM": "experiments/baseline/room0",
    "SplaTAM-CB": "experiments/with_cb/room0",
    "SplaTAM-CB-KNN": "experiments/with_cb_knn/room0",
}

# Collect ATE results
results = {}
for method_name, exp_dir in methods.items():
    ate_file = Path(exp_dir) / "ate_aligned.json"
    if ate_file.exists():
        with open(ate_file) as f:
            ate_data = json.load(f)
        results[method_name] = ate_data['rmse'] * 100  # Convert to cm
    else:
        print(f"Warning: {ate_file} not found")

# Plot comparison
plt.figure(figsize=(10, 6))
methods_list = list(results.keys())
rmse_values = list(results.values())

plt.bar(methods_list, rmse_values)
plt.ylabel('ATE RMSE (cm)')
plt.title('Trajectory Error Comparison - Replica Room0')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison.png', dpi=300)
plt.close()

print("Comparison saved to comparison.png")
```
    """)


def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print("TRAJECTORY EVALUATION - USAGE EXAMPLES")
    print("="*70)
    
    example_evaluate_saved_experiment()
    example_add_to_slam_script()
    example_custom_trajectory()
    example_batch_evaluation()
    example_compare_methods()
    
    print("\n" + "="*70)
    print("For more information, see:")
    print("  - trajectory_evaluation/README.md")
    print("  - trajectory_evaluation/eval_trajectory.py --help")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
