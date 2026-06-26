#!/usr/bin/env python3
"""
Regenerate metrics.png with updated ATE RMSE value from ate_aligned.json
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def regenerate_metrics_png(eval_dir):
    """Regenerate metrics.png with correct ATE RMSE value."""
    
    eval_path = Path(eval_dir)
    
    # Load ATE aligned metrics
    ate_aligned_path = eval_path / "ate_aligned.json"
    if not ate_aligned_path.exists():
        print(f"❌ Error: {ate_aligned_path} not found")
        return 1
    
    with open(ate_aligned_path) as f:
        ate_data = json.load(f)
    
    ate_rmse = ate_data['rmse']
    
    # Load existing metric files
    psnr_list = np.loadtxt(eval_path / "psnr.txt")
    l1_list = np.loadtxt(eval_path / "l1.txt")
    
    avg_psnr = psnr_list.mean()
    avg_l1 = l1_list.mean()
    
    print(f"\nMetrics for {eval_dir}:")
    print(f"  Average PSNR: {avg_psnr:.2f}")
    print(f"  Average Depth L1: {avg_l1*100:.2f} cm")
    print(f"  ATE RMSE (aligned): {ate_rmse*100:.2f} cm")
    
    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE (aligned): {:.2f} cm".format(
        avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    
    output_path = eval_path / "metrics.png"
    plt.savefig(str(output_path), bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Regenerated: {output_path}")
    print(f"  Now displays: ATE RMSE (aligned): {ate_rmse*100:.2f} cm\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate metrics.png with updated ATE RMSE')
    parser.add_argument('--eval_dir', type=str, required=True,
                       help='Path to eval directory containing ate_aligned.json')
    
    args = parser.parse_args()
    
    return regenerate_metrics_png(args.eval_dir)


if __name__ == "__main__":
    import sys
    sys.exit(main())
