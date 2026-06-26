#!/usr/bin/env python3
"""
Verify that the ATE RMSE calculation fix is working correctly.
"""
import numpy as np
import json
from pathlib import Path

def verify_experiment(exp_dir):
    """Verify ATE calculation for an experiment."""
    exp_path = Path(exp_dir)
    
    # Load JSON files
    ate_aligned_path = exp_path / "ate_aligned.json"
    eval_ate_aligned_path = exp_path / "eval" / "ate_aligned.json"
    
    if not ate_aligned_path.exists():
        print(f"❌ {ate_aligned_path} not found")
        return
    
    with open(ate_aligned_path) as f:
        ate_data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_dir}")
    print(f"{'='*70}")
    
    mean = ate_data['mean']
    std = ate_data['std']
    rmse = ate_data['rmse']
    
    # Verify RMSE calculation
    expected_rmse = np.sqrt(mean**2 + std**2)
    
    print(f"\nFrom ate_aligned.json:")
    print(f"  MEAN:          {mean*100:.2f} cm")
    print(f"  STD:           {std*100:.2f} cm")
    print(f"  RMSE:          {rmse*100:.2f} cm")
    print(f"\nVerification:")
    print(f"  Expected RMSE: {expected_rmse*100:.2f} cm (from √(mean² + std²))")
    print(f"  Match:         {'✅' if abs(rmse - expected_rmse) < 0.0001 else '❌'}")
    
    # Check if eval/ directory exists (from training)
    if eval_ate_aligned_path.exists():
        with open(eval_ate_aligned_path) as f:
            eval_ate_data = json.load(f)
        
        eval_mean = eval_ate_data['mean']
        eval_rmse = eval_ate_data['rmse']
        
        print(f"\nFrom eval/ate_aligned.json (training output):")
        print(f"  MEAN:          {eval_mean*100:.2f} cm")
        print(f"  RMSE:          {eval_rmse*100:.2f} cm")
        print(f"  Consistency:   {'✅' if abs(rmse - eval_rmse) < 0.01 else '❌'}")
    
    # Relationship check
    print(f"\nRelationship Check:")
    print(f"  RMSE > MEAN:   {'✅' if rmse >= mean else '❌'} (expected for non-zero std)")
    print(f"  RMSE formula:  RMSE = √(MEAN² + STD²)")
    
    return ate_data

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = "experiments/Replica/room1_1"
    
    verify_experiment(exp_dir)
    print(f"\n{'='*70}\n")
