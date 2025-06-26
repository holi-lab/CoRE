import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import random

from evaluator import evaluate_edited_model
from utils import set_seed


def load_data(
    data_path: str,
    start_idx: int,
    end_idx: Optional[int],
    ds_size: Optional[int] = None
) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if ds_size is not None:
        data = random.sample(data, min(ds_size, len(data)))
    return data[start_idx:end_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the pretrained edited model directory")
    parser.add_argument("--edit_data_dir", required=True, help="Path to the JSON test data file")
    parser.add_argument("--save_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--cuda_device", default="0", help="CUDA device index")
    parser.add_argument("--eval_max_length", type=int, default=50, help="Max generation length for evaluation")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--ds_size", type=int, default=None, help="Number of samples to randomly select from the dataset")
    parser.add_argument("--start_sample", type=int, default=1, help="1-based index of the first sample to evaluate")
    parser.add_argument("--end_sample", type=int, default=None, help="1-based index of the last sample to evaluate (inclusive)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    set_seed(args.seed)

    # Load and slice data
    start_idx = args.start_sample - 1
    end_idx = args.end_sample if args.end_sample is None else args.end_sample
    data = load_data(
        args.edit_data_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        ds_size=args.ds_size
    )

    print("=== Evaluation Only ===")
    start_time = time.time()
    results_by_type, avg_m = evaluate_edited_model(
        model_path=args.model_path,
        model_name=args.model_path,
        test_data=data,
        max_length=args.eval_max_length,
        batch_size=args.batch_size,
        cuda_device=args.cuda_device
    )
    elapsed = time.time() - start_time

    output = {
        "total_eval_time": elapsed,
        "total_samples": len(data),
        "avg_metrics": avg_m
    }

    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(
        args.save_dir,
        f"eval_results_{Path(args.model_path).name}_{args.start_sample}_{args.end_sample or 'end'}.json"
    )
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Evaluation results saved to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
