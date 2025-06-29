import os
import sys
import random
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EasyEdit')))

from easyeditor import (
    MEMITHyperParams, ROMEHyperParams, PMETHyperParams,
    EMMETHyperParams, FTHyperParams, COREHyperParams, AlphaEditHyperParams, BaseEditor
)
from evaluator import evaluate_edited_model
from utils import generate_unique_model_path, set_seed


SENTENCE_TYPES = [
    "REWRITE", "REPHRASE", "LOCALITY", "SBJ",
    "OBJ_NEW", "OBJ_OLD", "SBJ_HOP", "OBJ_NEW_HOP", "OBJ_OLD_HOP",
]

EDITING_HPARAMS = {
    "MEMIT": MEMITHyperParams,
    "ROME": ROMEHyperParams,
    "EMMET": EMMETHyperParams,
    "PMET": PMETHyperParams,
    "FT":    FTHyperParams,
    "CORE":  COREHyperParams,
    "AlphaEdit" : AlphaEditHyperParams
}


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


def prepare_editing_inputs(
    edit_data: List[Dict]
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    case_ids = [item["case_id"] for item in edit_data]
    prompts = [
        item["prompt"].format(item["subject"]) if "{}" in item["prompt"] else item["prompt"]
        for item in edit_data
    ]
    ground_truths = [item["fact_knowledge"] for item in edit_data]
    target_new = [item["edited_knowledge"] for item in edit_data]
    subjects = [item["subject"] for item in edit_data]
    return case_ids, prompts, ground_truths, target_new, subjects


def calculate_metrics(
    avg_metrics: Dict[str, Any],
    results_by_type: Any  # Changed from Dict to Any to handle both dict and list
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    
    scores = [m["editing_success"] for k, m in avg_metrics.items() if k in SENTENCE_TYPES and k != "LOCALITY"]
    sentence_editing_success = float(np.mean(scores)) if scores else 0.0

    # REWRITE repetition
    rep_metrics = {}
    if "REWRITE" in avg_metrics and "repetition_SRS" in avg_metrics["REWRITE"]:
        rep_metrics["AVG_SRS_REWRITE"] = float(avg_metrics["REWRITE"]["repetition_SRS"])

    # LOCALITY success
    loc_succ = 0.0
    if isinstance(results_by_type, dict):
        # Original logic for dict format
        locality_data = results_by_type.get("LOCALITY", {})
        if isinstance(locality_data, dict):
            loc_succ = float(locality_data.get("locality_success", 0.0))
    elif isinstance(results_by_type, list):
        # Handle list format - search for LOCALITY type in the list
        for item in results_by_type:
            if isinstance(item, dict) and item.get("type") == "LOCALITY":
                loc_succ = float(item.get("locality_success", 0.0))
                break
    else:
        if "LOCALITY" in avg_metrics and isinstance(avg_metrics["LOCALITY"], dict):
            loc_succ = float(avg_metrics["LOCALITY"].get("locality_success", 0.0))
    
    loc_metrics = {"LOCALITY/locality_success": loc_succ}
    return sentence_editing_success, rep_metrics, loc_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--editing_method", default="MEMIT")
    parser.add_argument("--hparams_dir", required=True)
    parser.add_argument("--edit_data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--ds_size", type=int, default=None)
    parser.add_argument("--start_sample", type=int, default=1)
    parser.add_argument("--end_sample", type=int, default=None)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--eval_max_length", type=int, default=50)
    parser.add_argument("--model_name", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    set_seed(args.seed)

    # 데이터 로드
    data = load_data(
        args.edit_data_dir,
        args.start_sample - 1,
        args.end_sample,
        args.ds_size
    )
    case_ids, prompts, gts, tgt, subjects = prepare_editing_inputs(data)

    # 편집 수행
    start = time.time()
    hparams_cls = EDITING_HPARAMS.get(args.editing_method)
    if not hparams_cls:
        raise NotImplementedError(f"Unknown editing method: {args.editing_method}")

    editor = BaseEditor.from_hparams(hparams_cls.from_hparams(args.hparams_dir))
    metrics, model, _ = editor.batch_edit(
        prompts=prompts,
        target_new=tgt,
        subject=subjects,
        ground_truth=gts,
        train_ds=None,
        keep_original_weight=True,
        sequential_edit=False
    )
    edit_time = time.time() - start


    for idx, cid in enumerate(case_ids):
        metrics[idx]["case_id"] = cid


    save_path = generate_unique_model_path(
        save_dir=args.save_dir,
        cuda_device=args.cuda_device,
        hparams_dir=args.hparams_dir,
        editing_method=args.editing_method,
        start_sample=args.start_sample,
        end_sample=args.end_sample
    )
    os.makedirs(save_path, exist_ok=True)

    try:
        model = model.to(torch.bfloat16)
        model.save_pretrained(save_path, torch_dtype=torch.bfloat16)
        editor.tok.save_pretrained(save_path)
        print(f"Saved model at {save_path}")
    finally:
        del model, editor
        torch.cuda.empty_cache()


    print("\n=== Evaluation ===")
    results_by_type, avg_m = evaluate_edited_model(
        model_path=save_path,
        model_name=args.model_name,
        test_data=data,
        max_length=args.eval_max_length,
        batch_size=256,
        cuda_device=args.cuda_device
    )
    sent_succ, rep_m, loc_m = calculate_metrics(avg_m, results_by_type)


    output = {"AVG_editing_success": sent_succ, "total_edit_time": edit_time, "total_samples": len(data)}
    output.update(rep_m)
    output.update(loc_m)
    for t, m in avg_m.items():
        if t == "LOCALITY":
            continue
        if "editing_success" in m:
            output[f"{t}/editing_success"] = m["editing_success"]
    os.makedirs(args.save_dir, exist_ok=True)
    json_file = os.path.join(args.save_dir, f"results_{args.editing_method}_{args.start_sample}_{args.end_sample or 'end'}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {json_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
